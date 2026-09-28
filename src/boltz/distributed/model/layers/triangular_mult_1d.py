# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Distributed triangular multiplicative update for 1D context parallelism.

Implements outgoing and incoming triangle multiplication over a 2D mesh
``(dp, cp)`` with 2-element placements.  Outgoing uses a 1D ring; incoming
forward uses a single tiled reduce-scatter (see below).

Under 1D CP, the pair tensor ``z [B, N, N, C_z]`` has placements
``(Shard(0), Shard(1))`` — row-slab ``[B, N/cp, N, C_z]``.

**Outgoing** (``z_out[i,j] = sum_k a[i,k] * b[j,k]``):
    After permute, ``a_local [B, c_h, N/cp, N]`` (i=local, k=full) and
    ``b_local [B, c_h, N/cp, N]`` (j=local, k=full).  The matmul
    ``a_local @ b_chunk^T`` contracts over k (full N on both sides) and
    produces ``[B, c_h, N/cp, N/cp]`` per ring step.  Ring rotates b
    through cp_size steps to cover all j-ranges.

**Incoming** (``z_out[i,j] = sum_k a[k,i] * b[k,j]``):
    After permute, ``a_local [B, c_h, N, N/cp]`` (i=full N, k=N/cp sharded)
    and ``b_local [B, c_h, N/cp, N]`` (k=N/cp sharded, j=full N).  The
    contraction axis ``k`` is the *sharded* axis, so each rank holds its
    k-chunk's contribution to every ``(i, j)``.  The forward computes per-rank
    output partials and lands the native row-slab via **reduce-scatter** over
    the cp axis (summing the k-reduction, scattering the output row axis ``i``),
    tiled along ``j`` into ``T`` tiles.  ``T`` is a collective-count vs
    peak-memory knob (default ``T = cp_size`` → O(N^2/cp) peak, ``T`` collectives;
    ``T = 1`` → 1 collective but O(N^2) peak).  The durable advantage over the
    prior 2-tensor operand-rotation ring is *per-collective* (not per-count):
    ~half the bytes (reduce one output vs rotate two operands), no send/recv
    double-buffer, partial computed once instead of recomputed per ring step.
    Identical FLOPs to the ring.  See ``_tiled_reduce_scatter_incoming`` and
    ``trimul-reduce-scatter-feasibility.md`` §1.

Communication budget:
    Forward (outgoing): cp_size ring steps, 1 tensor per step.
    Forward (incoming): T reduce-scatters over the cp axis (T = tile count,
        default cp_size; peak O(N^2/cp) at the default).
    Backward (outgoing): cp_size ring steps for dA + cp_size ring steps for dB.
    Backward (incoming): cp_size ring steps for dA + cp_size ring steps for dB
        (the backward retains the operand-rotation ring; its math is unchanged
        by the forward switch — see the backward docstring).
    All memory is O(N^2/cp) per rank at the default T.  No O(N^2) tensor is
    created at T = cp_size.
"""

from enum import Enum, auto

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard

from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.sigmoid_gate import sigmoid_gate
from boltz.distributed.model.layers.utils import _ring_p2p_send_recv
from boltz.distributed.utils import update_exhaustive_strides
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming as SerialTriangleMultiplicationIncoming,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationOutgoing as SerialTriangleMultiplicationOutgoing,
)


class _Direction(Enum):
    Outgoing = auto()
    Incoming = auto()


def _ring_rotate_b_assemble_cols(
    a_local: torch.Tensor,
    b_shard: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Ring matmul: rotate b, assemble column blocks in output.

    Computes ``a_local @ b_full`` where ``b_full`` is the concatenation of
    ``b_shard`` across all CP ranks along dim -1.

    Shapes::
        a_local : [B, c_h, M, K]
        b_shard : [B, c_h, K, N/cp]
        output  : [B, c_h, M, N]  (column blocks assembled by source rank)
    """
    n_local = b_shard.shape[-1]
    out = a_local.new_zeros(*a_local.shape[:-1], n_local * cp_size)
    buf = [b_shard.contiguous(), torch.empty_like(b_shard)]
    i_ready, i_recv = 0, 1

    send_to = (cp_rank - 1) % cp_size
    recv_from = (cp_rank + 1) % cp_size
    parity = cp_rank % 2 == 0

    for step in range(cp_size):
        if step < cp_size - 1:
            works = _ring_p2p_send_recv([buf[i_ready]], [buf[i_recv]], send_to, recv_from, cp_group, parity)

        source_rank = (cp_rank + step) % cp_size
        col_start = source_rank * n_local
        out[..., col_start : col_start + n_local] = torch.matmul(a_local, buf[i_ready])

        if step < cp_size - 1:
            for w in works:
                w.wait()
            i_ready ^= 1
            i_recv ^= 1

    return out


def _tiled_reduce_scatter_incoming(
    a_perm_local: torch.Tensor,
    b_perm_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
    num_tiles: int | None = None,
) -> torch.Tensor:
    """Incoming matmul via reduce-scatter, tiled along the j axis.

    Incoming's contraction axis ``k`` is the *sharded* (N/cp) axis, so each rank
    holds its k-chunk's contribution to *every* output ``(i, j)``.  The correct
    cross-rank reduction sums these per-k-chunk partials across the cp axis and
    scatters along the output **row axis ``i``** so each rank keeps its native
    ``N/cp`` i-slab — exactly the ``(Shard(0), Shard(1))`` row-slab the pipeline
    requires, with ``j`` full locally.  No transpose, no all-gather, no
    2D-sharded intermediate.

    This replaces the previous 2-tensor operand-rotation ring
    (``_ring_rotate_ab_accumulate_incoming``).  Both forms do identical FLOPs
    (``c_h * N^3 / cp`` per rank).  The reduce-scatter form's durable advantage
    over the ring (independent of the tile count) is **per-collective, not
    per-count**: each ``reduce_scatter`` moves ~half the bytes (it reduces one
    computed output instead of rotating two raw operands), there is **no
    send/recv double-buffer**, and each rank's partial is computed **once**
    instead of recomputed on every ring step.  See
    ``trimul-reduce-scatter-feasibility.md`` §1.

    **Tile count ``num_tiles`` (T) is a collective-count vs peak-memory knob, NOT
    a free "1 collective" win.**  The full per-rank partial ``[i=N, j=N]`` is
    O(N^2) if materialized whole.  We split the output ``j`` (full-N) axis into T
    tiles and issue one ``reduce_scatter`` per tile:

    - ``T = 1``  → a single collective, but the reduce-scatter input is the full
      ``[cp_size, N/cp, N]`` partial → **O(N^2) peak**.
    - ``T = cp_size`` (default) → ``cp_size`` collectives, each over an
      ``[cp_size, N/cp, N/cp]`` input → **O(N^2/cp) peak** (ring-parity peak).

    So at ring-parity peak the collective *count* is ~cp_size — comparable to the
    ring's cp_size steps.  The win is the per-collective structure above, not a
    reduction in collective count.  ``num_tiles`` defaults to ``None`` →
    ``cp_size``; callers may override to sweep the knee where per-collective
    savings beat tile-loop launch overhead (heavier on PCIe / small cp).

    Each tile's reduce-scatter input is a list of ``cp_size`` chunks ``[B, c_h,
    N/cp, j_tile]`` where chunk ``r`` is this rank's k-contribution to rank
    ``r``'s output i-slab; ``dist.reduce_scatter`` sums chunk ``r`` across ranks
    (the k-reduction) and lands it on rank ``r``.  On the **gloo** backend
    (which lacks reduce_scatter on some torch builds) the per-tile path falls
    back to ``all_reduce`` of the full-i partial then slices this rank's i-slab,
    mirroring ``triangular_attention_1d.py``; the per-tile transient is the same
    O(N^2 / num_tiles) on both backends.

    Note: the contraction axis must be divisible by cp — guaranteed by the
    even-sharding guard on ``N`` in the caller (``N % cp_size == 0``).

    Shapes::
        a_perm_local : [B, c_h, N, N/cp]  (i=full N, k=N/cp sharded)
        b_perm_local : [B, c_h, N/cp, N]  (k=N/cp sharded, j=full N)
        output       : [B, c_h, N/cp, N]  (i=local slab, j=full)

    Returns the reduce-scattered local i-slab in the same dtype as the matmul
    output (``a_perm_local``/``b_perm_local`` dtype, already >= fp32 in the
    forward path).
    """
    n_full = a_perm_local.shape[-2]  # i = full N
    n_local = n_full // cp_size  # N/cp

    if cp_size == 1:
        # Single rank: the local partial IS the output, no collective.
        return torch.matmul(a_perm_local, b_perm_local)

    if num_tiles is None:
        num_tiles = cp_size
    assert num_tiles >= 1, f"num_tiles must be >= 1, got {num_tiles}"
    # Tile the j (full-N) axis.  Bound num_tiles by N so each tile is non-empty;
    # use even tiling when it divides, else fall back to torch.tensor_split which
    # handles the remainder (still bounds peak to ~O(N^2 * ceil/N) per tile).
    num_tiles = min(num_tiles, n_full)

    out = a_perm_local.new_empty(*a_perm_local.shape[:2], n_local, n_full)

    # Gloo lacks reduce_scatter on some torch builds; mirror the sibling
    # fallback in triangular_attention_1d.py (all_reduce the full-i partial,
    # then slice this rank's i-slab).  The all_reduce path's per-tile transient
    # is the same O(cp_size * N/cp * j_tile) = O(N^2 / num_tiles) as the
    # reduce_scatter path (the full-i partial [B, c_h, N, j_tile] equals the
    # cp_size stacked i-slab chunks), so the peak budget is preserved on both
    # backends.
    use_gloo_fallback = dist.get_backend(cp_group) == "gloo"

    # Per-tile column ranges over the full-N j axis. int32 suffices: these are
    # token-index boundaries bounded by n_full (realistic N is O(10^4), far
    # below the int32 max ~2.1e9), and `.tolist()` converts them to Python ints
    # used only as slice bounds, so the tensor dtype never reaches a kernel.
    j_bounds = torch.linspace(0, n_full, steps=num_tiles + 1).round().to(torch.int32).tolist()
    for t in range(num_tiles):
        j0, j1 = j_bounds[t], j_bounds[t + 1]
        if j1 <= j0:
            continue
        b_tile = b_perm_local[..., j0:j1]  # [B, c_h, N/cp, j_tile]
        if use_gloo_fallback:
            # Full-i partial for this j-tile [B, c_h, N, j_tile] = this rank's
            # k-contribution to every output row; all_reduce sums across ranks
            # (the k-reduction), then slice this rank's i-slab.  .contiguous()
            # binds the buffer we reduce into and slice from to the same tensor.
            full_i = torch.matmul(a_perm_local, b_tile).contiguous()
            dist.all_reduce(full_i, op=dist.ReduceOp.SUM, group=cp_group)
            out[..., j0:j1] = full_i[..., cp_rank * n_local : (cp_rank + 1) * n_local, :]
        else:
            # Per-rank i-slab partials for this j-tile: chunk r = this rank's
            # k-contribution to rank r's output i-slab, [B, c_h, N/cp, j_tile].
            # dist.reduce_scatter (list form) sums chunk r across ranks (the
            # k-reduction) and returns rank r's chunk.  The list form is used
            # (rather than reduce_scatter_tensor) because it is backend-portable —
            # reduce_scatter_tensor enforces input.shape[0] == worldSize *
            # output.shape[0], which does not hold when the scattered cp axis is a
            # leading stack dim distinct from the batch dim.  All cp_size chunks
            # for one tile are alive at once: O(cp_size * N/cp * j_tile) =
            # O(N^2 / num_tiles) transient per tile.
            chunks = [
                torch.matmul(a_perm_local[..., r * n_local : (r + 1) * n_local, :], b_tile).contiguous()
                for r in range(cp_size)
            ]
            rs_out = torch.empty(chunks[0].shape, dtype=chunks[0].dtype, device=chunks[0].device)
            dist.reduce_scatter(rs_out, chunks, op=dist.ReduceOp.SUM, group=cp_group)
            out[..., j0:j1] = rs_out
    return out


def _ring_rotate_b_extract_cols_accumulate(
    a_local: torch.Tensor,
    b_shard: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Ring matmul: rotate b, extract matching columns from a, accumulate.

    Used for outgoing backward dA where we need:
        dA[i_local, k] = sum_j dX[i_local, j] * B[k, j]
    rewritten as ``dX @ B^T``, split over sharded j-ranges of B.

    Shapes::
        a_local : [B, c_h, M, N]     (e.g. dX: M=N/cp, last dim=N full)
        b_shard : [B, c_h, K, N/cp]  (e.g. B_perm: K=N, last dim=N/cp)
        output  : [B, c_h, M, K]     (accumulated)
    """
    n_local = b_shard.shape[-1]
    out = a_local.new_zeros(*a_local.shape[:3], b_shard.shape[-2])
    buf = [b_shard.contiguous(), torch.empty_like(b_shard)]
    i_ready, i_recv = 0, 1

    send_to = (cp_rank - 1) % cp_size
    recv_from = (cp_rank + 1) % cp_size
    parity = cp_rank % 2 == 0

    for step in range(cp_size):
        if step < cp_size - 1:
            works = _ring_p2p_send_recv([buf[i_ready]], [buf[i_recv]], send_to, recv_from, cp_group, parity)

        source_rank = (cp_rank + step) % cp_size
        j_start = source_rank * n_local
        out = out + torch.matmul(
            a_local[..., j_start : j_start + n_local],
            buf[i_ready].transpose(-1, -2),
        )

        if step < cp_size - 1:
            for w in works:
                w.wait()
            i_ready ^= 1
            i_recv ^= 1

    return out


class _TriangleMultiplication1DImpl(torch.autograd.Function):
    """Distributed triangle multiplication BMM via 1D ring communication.

    Handles the gated projection, masking, and distributed matmul.
    The surrounding layer norms and output gating are computed by the
    nn.Module wrapper so that PyTorch autograd handles their gradients.

    Inputs:
        x : DTensor  [B, N, N, 2*c_hidden], placements (Shard(0), Shard(1))
            Output of p_in linear projection (pre-gating).
        mask : DTensor  [B, N, N], placements (Shard(0), Shard(1))
        g : DTensor  [B, N, N, 2*c_hidden], placements (Shard(0), Shard(1))
            Pre-sigmoid gate tensor (output of g_in).
        cp_group : dist.ProcessGroup
        direction : _Direction
        incoming_rs_tiles : int | None
            Incoming-only reduce-scatter tile count (T). ``None`` → cp_size
            (O(N^2/cp) peak). See _tiled_reduce_scatter_incoming for the
            collective-count vs peak-memory tradeoff. Ignored for outgoing.

    Output:
        DTensor [B, N, N, c_hidden], placements (Shard(0), Shard(1))

    Communication (forward):
        Outgoing: cp_size ring steps, 1 tensor per step.
        Incoming: T reduce-scatters (T = incoming_rs_tiles, default cp_size).
    Communication (backward):
        2 * cp_size ring steps for both directions.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: DTensor,
        mask: DTensor,
        g: DTensor,
        cp_group: dist.ProcessGroup,
        direction: _Direction,
        incoming_rs_tiles: int | None = None,
    ) -> DTensor:
        _expected = (Shard(0), Shard(1))
        for name, t in [("x", x), ("mask", mask), ("g", g)]:
            assert isinstance(t, DTensor), f"{name} must be a DTensor, got {type(t)}"
            assert t.device_mesh is x.device_mesh, f"{name} device_mesh differs from x"
            plc = tuple(t.placements)
            assert plc == _expected, f"{name} placements must be {_expected}, got {plc}"

        assert x.shape[-1] % 2 == 0, f"x last dim must be even, got {x.shape[-1]}"
        assert x.shape == g.shape, f"x and g must have same shape, got {x.shape} vs {g.shape}"
        assert mask.shape == x.shape[:3], f"mask shape {mask.shape} must match x shape[:3] {x.shape[:3]}"

        cp_size = dist.get_world_size(cp_group)
        n_full = x.shape[1]
        assert n_full % cp_size == 0, f"N ({n_full}) must be evenly divisible by cp_size ({cp_size})"
        assert (
            x.shape[1] == x.shape[2]
        ), f"Pair tensor N dims must be square, got dim1={x.shape[1]} vs dim2={x.shape[2]}"

        device_mesh = x.device_mesh
        placements = x.placements
        cp_rank = dist.get_rank(cp_group)

        # Apply sigmoid gating and mask locally.
        # Gating and masking are fused into this autograd.Function (rather than
        # handled externally by PyTorch autograd as in the OpenFold reference)
        # to avoid materializing intermediate DTensors for the gated output,
        # saving one O(N^2/cp) DTensor allocation.
        mask_local = mask.to_local().unsqueeze(-1)  # [B, N/cp, N, 1]
        # Capture the original input dtype so saved-for-backward tensors can be
        # downcast back to it (bf16 under AMP) to halve the pair-tensor memory
        # footprint at the autograd boundary. Forward arithmetic still runs in
        # safe_dtype (>= fp32) for numerical accuracy.
        input_dtype = x.to_local().dtype
        safe_dtype = torch.promote_types(input_dtype, torch.float32)
        sig_g_local = g.to_local().sigmoid().to(dtype=safe_dtype)
        x_local = x.to_local().to(dtype=safe_dtype) * mask_local
        x_local = x_local * sig_g_local

        # Split into a, b projections
        c_hidden = x_local.shape[-1] // 2
        a_local = x_local[..., :c_hidden]
        b_local = x_local[..., c_hidden:]

        if direction == _Direction.Outgoing:
            # Serial einsum: "bikd,bjkd->bijd"
            # a_perm [B, c_h, N/cp, N] (i=local, k=full)
            # b_perm [B, c_h, N, N/cp] (k=full, j=local) -> transposed for matmul
            a_perm = a_local.permute(0, 3, 1, 2).contiguous()
            b_perm = b_local.permute(0, 3, 2, 1).contiguous()

            x_perm = _ring_rotate_b_assemble_cols(a_perm, b_perm, cp_group, cp_size, cp_rank)
            out_local = x_perm.permute(0, 2, 3, 1).contiguous()
        else:
            # Serial einsum: "bkid,bkjd->bijd"
            # a_perm [B, c_h, N, N/cp] (k=N/cp sharded, i=full N)
            # b_perm [B, c_h, N/cp, N] (k=N/cp sharded, j=full N)
            a_perm = a_local.permute(0, 3, 2, 1).contiguous()
            b_perm = b_local.permute(0, 3, 1, 2).contiguous()

            x_perm = _tiled_reduce_scatter_incoming(a_perm, b_perm, cp_group, cp_size, cp_rank, incoming_rs_tiles)
            out_local = x_perm.permute(0, 2, 3, 1).contiguous()

        # out_local carries safe_dtype = promote_types(input_dtype, fp32), exactly
        # mirroring the serial TriMul whose einsum output (also safe_dtype) is fed
        # straight into norm_out with NO downcast (triangular_mult.py:143-146 /
        # 232-235). Returning safe_dtype here is therefore FAITHFUL to serial
        # (CLAUDE.md "serial is ground truth"): it preserves the fp32 precision
        # serial keeps at the einsum->norm_out handoff (and preserves fp64 paths
        # via promote_types), and is a no-op under bf16-mixed autocast in
        # production (the matmul autocasts to bf16 anyway, as does serial's
        # einsum). A `.to(input_dtype)` downcast here would DIVERGE from serial by
        # clamping the fp32 intermediate to bf16 before norm_out. No silent fp32
        # *upcast* occurs: there is no hardcoded `.float()`, and the fp64 path is
        # preserved (verified by the safe_dtype assertion in
        # test_dtensor_triangular_mult_1d.py).

        if x.requires_grad:
            # Save a, b (masked+gated halves), mask, and post-sigmoid gate.
            # x_local = cat(a_local, b_local) is reconstructed in backward to
            # avoid saving redundant data at O(N^2/cp) scale.
            #
            # Memory optimisation: downcast a/b/sig_g to input_dtype (bf16 under
            # AMP) before saving — pair-tensor saves are halved (4B fp32 -> 2B
            # bf16). Backward re-promotes to fp32 at entry. The fp32->bf16->fp32
            # round-trip introduces ~eps_bf16 (~7.8e-3) relative error per saved
            # element; backward matmuls amplify this by sqrt(N_local). Layer
            # tests under bf16 use a tolerance derived from that bound (see
            # test_dtensor_triangular_mult_1d.py bf16 parametrization).
            ctx.save_for_backward(
                a_local.to(dtype=input_dtype),
                b_local.to(dtype=input_dtype),
                mask_local,
                sig_g_local.to(dtype=input_dtype),
            )
            ctx.cp_group = cp_group
            ctx.direction = direction
            ctx.placements = placements
            ctx.device_mesh = device_mesh
            ctx.shape_x = x.shape
            ctx.stride_x = x.stride()
            ctx.shape_g = g.shape
            ctx.stride_g = g.stride()
            ctx.cp_size = cp_size
            ctx.cp_rank = cp_rank

        c_hidden_out = out_local.shape[-1]
        shape_out = x.shape[:-1] + (c_hidden_out,)
        stride_out = update_exhaustive_strides(x.shape, x.stride(), shape_out)

        return DTensor.from_local(
            out_local,
            device_mesh=device_mesh,
            placements=placements,
            shape=shape_out,
            stride=stride_out,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        a_local, b_local, mask_local, sig_g_local = ctx.saved_tensors
        cp_group = ctx.cp_group
        direction = ctx.direction
        cp_size = ctx.cp_size
        cp_rank = ctx.cp_rank

        # Re-promote saved tensors (downcast to input_dtype at save time to
        # halve the saved-for-backward memory footprint) back to safe_dtype
        # for gradient arithmetic. promote_types preserves fp64 paths.
        compute_dtype = torch.promote_types(a_local.dtype, torch.float32)
        a_local = a_local.to(dtype=compute_dtype)
        b_local = b_local.to(dtype=compute_dtype)
        sig_g_local = sig_g_local.to(dtype=compute_dtype)

        grad_out_local = grad_output.to_local().to(dtype=compute_dtype)

        if direction == _Direction.Outgoing:
            # Forward was: X = A_perm @ B_perm_assembled
            a_perm = a_local.permute(0, 3, 1, 2).contiguous()
            b_perm = b_local.permute(0, 3, 2, 1).contiguous()
            grad_perm = grad_out_local.permute(0, 3, 1, 2).contiguous()

            n_local = a_local.shape[1]

            # dA: ring-rotate B, extract matching j-cols from grad
            grad_a_perm = _ring_rotate_b_extract_cols_accumulate(grad_perm, b_perm, cp_group, cp_size, cp_rank)
            grad_a_local = grad_a_perm.permute(0, 2, 3, 1).contiguous()

            # dB: ring-rotate both A and grad
            j_start = cp_rank * n_local

            send_to = (cp_rank - 1) % cp_size
            recv_from = (cp_rank + 1) % cp_size
            parity = cp_rank % 2 == 0

            grad_b_perm = a_perm.new_zeros(*a_perm.shape[:2], a_perm.shape[3], n_local)
            buf_a = [a_perm.contiguous(), torch.empty_like(a_perm)]
            buf_g = [grad_perm.contiguous(), torch.empty_like(grad_perm)]
            i_ready, i_recv = 0, 1

            for step in range(cp_size):
                if step < cp_size - 1:
                    works = _ring_p2p_send_recv(
                        [buf_a[i_ready], buf_g[i_ready]],
                        [buf_a[i_recv], buf_g[i_recv]],
                        send_to,
                        recv_from,
                        cp_group,
                        parity,
                    )

                grad_j_chunk = buf_g[i_ready][..., j_start : j_start + n_local]
                grad_b_perm = grad_b_perm + torch.matmul(buf_a[i_ready].transpose(-1, -2), grad_j_chunk)

                if step < cp_size - 1:
                    for w in works:
                        w.wait()
                    i_ready ^= 1
                    i_recv ^= 1

            grad_b_local = grad_b_perm.permute(0, 3, 2, 1).contiguous()

        else:
            # Incoming
            a_perm = a_local.permute(0, 3, 2, 1).contiguous()
            b_perm = b_local.permute(0, 3, 1, 2).contiguous()
            grad_perm = grad_out_local.permute(0, 3, 1, 2).contiguous()

            n_local = a_local.shape[1]

            # dA: ring-rotate dX^T, assemble col blocks
            dX_t = grad_perm.transpose(-1, -2).contiguous()
            grad_a_perm_t = _ring_rotate_b_assemble_cols(
                b_perm,
                dX_t,
                cp_group,
                cp_size,
                cp_rank,
            )
            grad_a_perm = grad_a_perm_t.transpose(-1, -2).contiguous()
            grad_a_local = grad_a_perm.permute(0, 3, 2, 1).contiguous()

            # dB: ring-rotate dX
            send_to = (cp_rank - 1) % cp_size
            recv_from = (cp_rank + 1) % cp_size
            parity = cp_rank % 2 == 0

            a_perm_t = a_perm.transpose(-1, -2).contiguous()
            grad_b_perm = torch.zeros_like(b_perm)
            buf = [grad_perm.contiguous(), torch.empty_like(grad_perm)]
            i_ready, i_recv_b = 0, 1

            for step in range(cp_size):
                if step < cp_size - 1:
                    works = _ring_p2p_send_recv([buf[i_ready]], [buf[i_recv_b]], send_to, recv_from, cp_group, parity)

                source_rank = (cp_rank + step) % cp_size
                n2_start = source_rank * n_local
                grad_b_perm = grad_b_perm + torch.matmul(
                    a_perm_t[..., n2_start : n2_start + n_local],
                    buf[i_ready],
                )

                if step < cp_size - 1:
                    for w in works:
                        w.wait()
                    i_ready ^= 1
                    i_recv_b ^= 1

            grad_b_local = grad_b_perm.permute(0, 2, 3, 1).contiguous()

        # Chain rule through mask and sigmoid gating:
        # forward: x_gated = x_raw * mask * sig(g)
        # d_x_raw = d_x_gated * mask * sig(g)
        # d_g = d_x_gated * x_raw * mask * sig(g) * (1 - sig(g))
        #      = d_x_gated * x_gated * (1 - sig(g))
        # where x_gated = cat(a_local, b_local)
        grad_ab_local = torch.cat([grad_a_local, grad_b_local], dim=-1)

        # d_x_raw (through mask * sig(g))
        grad_x_local = grad_ab_local * mask_local * sig_g_local

        # d_g (through sigmoid gate); reconstruct x_gated from saved halves
        x_gated_local = torch.cat([a_local, b_local], dim=-1)
        grad_g_local = grad_ab_local * x_gated_local * (1 - sig_g_local)

        grad_x = DTensor.from_local(
            grad_x_local,
            device_mesh=ctx.device_mesh,
            placements=ctx.placements,
            shape=ctx.shape_x,
            stride=ctx.stride_x,
        )
        grad_g = DTensor.from_local(
            grad_g_local,
            device_mesh=ctx.device_mesh,
            placements=ctx.placements,
            shape=ctx.shape_g,
            stride=ctx.stride_g,
        )

        # Grads for (x, mask, g, cp_group, direction, incoming_rs_tiles).
        return grad_x, None, grad_g, None, None, None


class TriangleMultiplication1D(nn.Module):
    """Distributed triangle multiplication for 1D CP (2D mesh ``(dp, cp)``).

    Wraps a serial ``TriangleMultiplicationOutgoing`` or
    ``TriangleMultiplicationIncoming`` and replaces the local matmul
    with a 1D ring-based distributed BMM over the cp axis.

    Parameters
    ----------
    direction : _Direction
        Whether this is outgoing or incoming multiplication.
    layer : SerialTriangleMultiplicationOutgoing | SerialTriangleMultiplicationIncoming
        The serial module whose weights we wrap.
    device_mesh : DeviceMesh
        The 2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        The CP process group for ring communication.
    incoming_rs_tiles : int | None
        Incoming-only reduce-scatter tile count (T). ``None`` (default) →
        cp_size, which holds peak at O(N^2/cp). See
        _tiled_reduce_scatter_incoming for the collective-count vs peak-memory
        tradeoff. Ignored for the outgoing direction (it uses the ring).
    """

    def __init__(
        self,
        direction: _Direction,
        layer: SerialTriangleMultiplicationOutgoing | SerialTriangleMultiplicationIncoming,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
        incoming_rs_tiles: int | None = None,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.cp_group = cp_group
        self._direction = direction
        self._incoming_rs_tiles = incoming_rs_tiles

        self.norm_in = LayerNormParamsReplicated(layer.norm_in, device_mesh)
        self.norm_out = LayerNormParamsReplicated(layer.norm_out, device_mesh)
        self.p_in = LinearParamsReplicated(layer.p_in, device_mesh)
        self.g_in = LinearParamsReplicated(layer.g_in, device_mesh)
        self.p_out = LinearParamsReplicated(layer.p_out, device_mesh)
        self.g_out = LinearParamsReplicated(layer.g_out, device_mesh)

    def forward(self, z: DTensor, mask: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        z : DTensor  [B, N, N, C_z], placements (Shard(0), Shard(1))
        mask : DTensor  [B, N, N], placements (Shard(0), Shard(1))
        """
        z_norm = self.norm_in(z)
        g_out = self.g_out(z_norm)

        # Projection (pre-gating)
        g = self.g_in(z_norm)
        x = self.p_in(z_norm)

        # Distributed triangle multiplication (mask and sigmoid gating applied inside)
        x = _TriangleMultiplication1DImpl.apply(x, mask, g, self.cp_group, self._direction, self._incoming_rs_tiles)

        # Output gating
        x = self.p_out(self.norm_out(x))
        x = sigmoid_gate(x, g_out)

        return x


class TriangleMultiplicationOutgoing1D(TriangleMultiplication1D):
    """Distributed outgoing triangle multiplication for 1D CP."""

    def __init__(
        self,
        layer: SerialTriangleMultiplicationOutgoing,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(_Direction.Outgoing, layer, device_mesh, cp_group)


class TriangleMultiplicationIncoming1D(TriangleMultiplication1D):
    """Distributed incoming triangle multiplication for 1D CP."""

    def __init__(
        self,
        layer: SerialTriangleMultiplicationIncoming,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__(_Direction.Incoming, layer, device_mesh, cp_group)
