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

"""Distributed outer product mean for 1D context parallelism (2D mesh).

Under 1D CP with mesh ``(dp, cp)``, MSA embedding ``m`` has shape
``[B, S, N, C_m]`` with placements ``(Shard(0), Shard(2))``: B on dp, N on cp,
S fully replicated.

OPM computes ``z[i,j] = mean_s(a[s,i] * b[s,j])`` where the mean is over S.
After projecting and transposing, each rank holds ``a[B, N/cp, S, c_hidden]``
and ``b[B, N/cp, S, c_hidden]``.  The output pair ``z[B, N, N, C_z]`` is a
row-slab ``[B, N/cp, N, C_z]`` with placements ``(Shard(0), Shard(1))``.

To produce ``z[i_local, j_all]``, we need ``b[s, j]`` for all j.  Rather than
materialising the full N-length ``b`` via an all-gather, we **ring-rotate**
``b`` and the mask across cp ranks for ``cp_size`` steps; each step
accumulates one N/cp j-block of ``z_local`` (and the mask norm) from the
chunk currently in the rotating buffer.  Peak buffer state stays at
``[B_local, N/cp, S, c_hidden]``.

Communication budget:
    Forward:  ``cp_size`` ring-rotations of ``(b_chunk, mask_chunk)``,
              each O(N/cp * S * c_hidden) per step.
    Backward: Stage A -- ``cp_size`` ring-rotations of ``b_local`` to
              accumulate ``grad_a`` per j-block.  Stage B -- reduce-scatter
              ring with ``cp_size - 1`` rotations of the partial accumulator,
              each O(N/cp * S * c_hidden) per step.  No O(N) tensor is
              materialised at any point.

Memory: peak ring buffer is O(N/cp * S * c_hidden) per side.  For S=512,
N=384, cp=2, c_hidden=32 this saves ~12.5 MB in fp32 per forward versus
the previous all-gather path.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.utils import _ring_p2p_send_recv
from boltz.distributed.utils import update_exhaustive_strides
from boltz.model.layers.outer_product_mean import OuterProductMean as SerialOuterProductMean

MSA_PLACEMENTS_1D = (Shard(0), Shard(2))
PAIR_PLACEMENTS_1D = (Shard(0), Shard(1))


class _OuterProductMean1DImpl(torch.autograd.Function):
    """Ring-based outer product sum for 1D CP.

    Computes ``z[i,j] = sum_s(a[s,i] * b[s,j])`` where ``i`` is local
    (``N/cp``) and ``j`` spans the full ``N`` dimension.  Rather than
    all-gathering ``b`` once, this implementation ring-rotates ``b`` and the
    masked padding tensor across cp ranks, accumulating one ``N/cp``
    j-block of ``z_local`` per step.

    Returns the unnormalized sum.  The mask norm is stored on the
    ``norm_container`` list (index 0) for the module wrapper to divide by
    after ``linear_out``.

    Input placements (2D mesh ``(dp, cp)``):
        a, b:   ``(Shard(0), Shard(2))``  -- shape ``[B, S, N, c_hidden]``
        mask:   ``(Shard(0), Shard(2))``  -- shape ``[B, S, N]``

    Output placements:
        z:      ``(Shard(0), Shard(1))``  -- shape ``[B, N, N, c_hidden^2]``

    Forward collectives:  ``cp_size`` ring-rotations of
        ``(b_chunk, mask_chunk)``; each step moves an O(N/cp * S * c_hidden)
        payload.
    Backward collectives: Stage A -- ``cp_size`` ring-rotations of
        ``b_local`` for ``grad_a`` accumulation.  Stage B -- reduce-scatter
        ring with ``cp_size - 1`` rotations of an O(N/cp * S * c_hidden)
        partial accumulator for ``grad_b``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        a: DTensor,
        b: DTensor,
        mask: DTensor,
        cp_group: dist.ProcessGroup,
        norm_container: list,
    ) -> DTensor:
        # --- Input validation ---
        for name, tensor in [("a", a), ("b", b), ("mask", mask)]:
            if not isinstance(tensor, DTensor):
                raise TypeError(f"Input '{name}' must be DTensor, got {type(tensor)}")

        device_mesh = a.device_mesh
        for name, tensor in [("b", b), ("mask", mask)]:
            if tensor.device_mesh != device_mesh:
                raise ValueError(f"Input '{name}' must share device_mesh with 'a'")

        for name, tensor in [("a", a), ("b", b), ("mask", mask)]:
            if tensor.placements != MSA_PLACEMENTS_1D:
                raise ValueError(f"Input '{name}' must have placements {MSA_PLACEMENTS_1D}, got {tensor.placements}")

        if a.shape != b.shape:
            raise ValueError(f"a and b must have the same shape, got {a.shape} and {b.shape}")
        if a.ndim != 4:
            raise ValueError(f"a and b must be 4D, got {a.ndim}D")

        cp_size = dist.get_world_size(cp_group)
        N_global = a.shape[2]
        if N_global % cp_size != 0:
            raise ValueError(f"N ({N_global}) must be evenly divisible by cp_size ({cp_size})")
        B_global = a.shape[0]
        if B_global % device_mesh.size(0) != 0:
            raise ValueError(f"B ({B_global}) must be evenly divisible by dp size ({device_mesh.size(0)})")

        ctx.mark_non_differentiable(mask)

        # to_local: a, b are [B_local, S, N_local, c_hidden], mask is [B_local, S, N_local]
        mask_local = mask.to_local().unsqueeze(-1)  # [B_local, S, N_local, 1]
        a_local = a.to_local() * mask_local
        b_local = b.to_local() * mask_local

        c_hidden = a_local.shape[-1]

        # Transpose: [B_local, S, N_local, c_hidden] -> [B_local, N_local, S, c_hidden]
        a_local = a_local.transpose(1, 2).contiguous()
        b_local = b_local.transpose(1, 2).contiguous()
        mask_t = mask_local.transpose(1, 2).contiguous()  # [B_local, N_local, S, 1]

        B_local, N_local = a_local.shape[0], a_local.shape[1]

        # Ring rotate b_chunk + mask_chunk across cp ranks.  Each step
        # accumulates one N/cp j-block of z_local from the chunk currently
        # in the rotating buffer; peak intermediate b-state stays at
        # [B_local, N/cp, S, c_hidden] instead of the previous full-N gather.
        cp_rank = dist.get_rank(cp_group)
        send_to = (cp_rank - 1) % cp_size
        recv_from = (cp_rank + 1) % cp_size
        parity = cp_rank % 2 == 0

        compute_dtype = torch.promote_types(a_local.dtype, torch.float32)
        norm_compute_dtype = torch.promote_types(mask_local.dtype, torch.float32)

        a_compute = a_local.to(compute_dtype)
        mask_compute = mask_t.to(norm_compute_dtype)

        z_local = torch.zeros(
            B_local,
            N_local,
            N_global,
            c_hidden,
            c_hidden,
            dtype=compute_dtype,
            device=a_local.device,
        )
        norm_local = torch.zeros(
            B_local,
            N_local,
            N_global,
            1,
            dtype=norm_compute_dtype,
            device=mask_t.device,
        )

        # NOTE: clone b_local / mask_t into buf_b[0] / buf_mask[0] so that the
        # cp>=3 ring recv (which targets buf_b[0] at step >=1 after the
        # i_ready/i_recv swap) does NOT overwrite the saved-for-backward
        # ``b_local`` / ``mask_t`` tensors -- ``.contiguous()`` returns the
        # same storage when the input is already contiguous, which would
        # otherwise corrupt the saved state at cp_size >= 3.
        buf_b = [b_local.clone(), torch.empty_like(b_local)]
        buf_mask = [mask_t.clone(), torch.empty_like(mask_t)]
        i_ready, i_recv = 0, 1

        for step in range(cp_size):
            if step < cp_size - 1:
                works = _ring_p2p_send_recv(
                    [buf_b[i_ready], buf_mask[i_ready]],
                    [buf_b[i_recv], buf_mask[i_recv]],
                    send_to,
                    recv_from,
                    cp_group,
                    parity,
                )

            # j-slice mapping: rank r owns columns [r*N_local, (r+1)*N_local).
            # At step ``step``, the buffer holds the b-shard of source_rank.
            source_rank = (cp_rank + step) % cp_size
            j_start = source_rank * N_local
            j_end = j_start + N_local
            b_chunk = buf_b[i_ready]  # [B_local, N_local, S, c_hidden]
            mask_chunk = buf_mask[i_ready]  # [B_local, N_local, S, 1]

            z_local[:, :, j_start:j_end, :, :] += torch.einsum(
                "bisc,bjsd->bijcd",
                a_compute,
                b_chunk.to(compute_dtype),
            )
            norm_local[:, :, j_start:j_end, :] += torch.einsum(
                "bise,bjse->bije",
                mask_compute,
                mask_chunk.to(norm_compute_dtype),
            )

            if step < cp_size - 1:
                for w in works:
                    w.wait()
                i_ready ^= 1
                i_recv ^= 1

        z_local = z_local.to(a_local.dtype)
        # Flatten: [B_local, N_local, N, c_hidden^2]
        z_local = z_local.flatten(start_dim=-2)
        norm_local = norm_local.clamp(min=1)

        # Store norm for module wrapper
        norm_container.clear()
        norm_container.append(norm_local)

        # Save for backward
        if a.requires_grad or b.requires_grad:
            ctx.save_for_backward(a_local.detach(), b_local.detach(), mask_t.detach())
            ctx.cp_group = cp_group
            ctx.cp_size = cp_size
            ctx.device_mesh = device_mesh
            ctx.input_shape_a = a.shape
            ctx.input_stride_a = a.stride()
            ctx.input_shape_b = b.shape
            ctx.input_stride_b = b.stride()
            ctx.c_hidden = c_hidden

        # Output: pair placements (Shard(0), Shard(1))
        shape_output = (B_global, N_global, N_global, z_local.shape[-1])
        # Reference shape only used for ndim and contiguity ordering (both are 4D C-contiguous).
        strides_output = update_exhaustive_strides(a.shape, a.stride(), shape_output)

        z = DTensor.from_local(
            z_local,
            device_mesh=device_mesh,
            placements=PAIR_PLACEMENTS_1D,
            shape=shape_output,
            stride=strides_output,
        )
        return z

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_z: DTensor):
        if not isinstance(grad_z, DTensor):
            raise TypeError(f"grad_z must be DTensor, got {type(grad_z)}")

        a_local, b_local, mask_t = ctx.saved_tensors
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        c_hidden = ctx.c_hidden

        # grad_z: pair placements (Shard(0), Shard(1))
        # local: [B_local, N_local, N, c_hidden^2]
        grad_z_local = grad_z.to_local()

        # Cast grad to match saved tensor dtype for mixed-precision safety
        go_dtype = a_local.dtype
        if grad_z_local.dtype != go_dtype:
            grad_z_local = grad_z_local.to(go_dtype)

        grad_z_local = grad_z_local.unflatten(-1, (c_hidden, c_hidden))

        N_local = a_local.shape[1]

        cp_rank = dist.get_rank(cp_group)
        send_to = (cp_rank - 1) % cp_size
        recv_from = (cp_rank + 1) % cp_size
        parity = cp_rank % 2 == 0

        compute_dtype = torch.promote_types(a_local.dtype, torch.float32)

        # --- Stage A: grad_a_local via ring on b_local ---
        # da[i_local, s, c] = sum_j sum_d grad_z[i_local, j, c, d] * b[j, s, d]
        # Each step contributes one N/cp j-block.  Memory stays at
        # [B_local, N_local, S, c_hidden].
        grad_a_local = torch.zeros_like(a_local, dtype=compute_dtype)

        # Clone b_local into buf_b[0] so the cp>=3 ring recv (into buf_b[0]
        # at step >= 1 after the i_ready/i_recv swap) does not overwrite
        # the saved b_local.
        buf_b = [b_local.clone(), torch.empty_like(b_local)]
        i_ready, i_recv = 0, 1

        for step in range(cp_size):
            if step < cp_size - 1:
                works = _ring_p2p_send_recv(
                    [buf_b[i_ready]],
                    [buf_b[i_recv]],
                    send_to,
                    recv_from,
                    cp_group,
                    parity,
                )

            source_rank = (cp_rank + step) % cp_size
            j_start = source_rank * N_local
            j_end = j_start + N_local
            grad_z_col = grad_z_local[:, :, j_start:j_end, :, :]  # [B_local, N_local_i, N_local_j, c, d]
            b_chunk = buf_b[i_ready]  # [B_local, N_local_j, S, c_hidden]

            grad_a_local += torch.einsum(
                "bijcd,bjsd->bisc",
                grad_z_col.to(compute_dtype),
                b_chunk.to(compute_dtype),
            )

            if step < cp_size - 1:
                for w in works:
                    w.wait()
                i_ready ^= 1
                i_recv ^= 1

        grad_a_local = grad_a_local.to(a_local.dtype)
        grad_a_local = grad_a_local * mask_t
        # Transpose back: [B_local, N_local, S, c_hidden] -> [B_local, S, N_local, c_hidden]
        grad_a_local = grad_a_local.transpose(1, 2).contiguous()

        grad_a = DTensor.from_local(
            grad_a_local,
            device_mesh=ctx.device_mesh,
            placements=MSA_PLACEMENTS_1D,
            shape=ctx.input_shape_a,
            stride=ctx.input_stride_a,
        )

        # --- Stage B: grad_b_local via reduce-scatter ring on the partial accumulator ---
        # db[j_local_r, s, d] = sum_{i_global, c} grad_z[i, j_local_r, c, d] * a[i, s, c]
        #                     = sum_{q} sum_{i_local_q, c} grad_z_local_q[i_local_q, j_local_r, c, d] * a_local_q[i_local_q, s, c]
        # Define partial[q->r] = einsum(grad_z_local_q[:, :, r*N_local:(r+1)*N_local, :, :], a_local_q).
        # The full grad_b at rank r is grad_b_local_r = sum_q partial[q->r].
        #
        # We accumulate this with a reduce-scatter ring: each rank maintains a
        # single [B_local, N_local, S, c_hidden] accumulator tracking the running
        # sum for one current j-chunk owner; the accumulator is rotated around
        # the ring so each chunk passes through every rank, picking up each
        # rank's local partial along the way.  Peak memory O(B_local * N_local * S * c_hidden)
        # mirrors Stage A and the forward o_acc.  Total communication = (cp_size - 1)
        # ring hops of one [B_local, N_local, S, c_hidden] tensor per rank.
        #
        # Ring direction matches forward / Stage A: send_to = (cp_rank - 1) % cp_size,
        # recv_from = (cp_rank + 1) % cp_size.  Initial chunk owner at rank q is
        # (q + 1) % cp_size; after each hop the owner pointer becomes whatever
        # came in from recv_from, which is one step further around the ring.
        # After cp_size - 1 hops, the owner pointer at rank q is q itself, so
        # the accumulator now holds sum_q partial[q->q] = grad_b_local_q.
        #
        # Why not a single all-reduce on a [B_local, N_local, S, c_hidden] buffer?
        # Each rank's local buffer would index DIFFERENT global-j positions (the
        # j-chunk it owns), so an elementwise sum across ranks would add partials
        # for disjoint global-j slots -- silently wrong while still passing
        # forward-only parity.  The ring routes each partial to the rank that
        # owns its j-chunk.
        #
        # Why not raw dist.isend / dist.irecv with per-step changing peers?
        # That pattern empirically deadlocks NCCL under Lightning bf16-mixed
        # autocast on this codepath (pinned by a 7-iter falsification chain on
        # sichu/bf16-deadlock-probe).  The fixed-direction ring with batched
        # P2P via _ring_p2p_send_recv mirrors the bf16-healthy pattern used in
        # PWA backward and OPM 1D Stage A / forward.
        buf_shape = (a_local.shape[0], N_local, a_local.shape[2], c_hidden)
        accum = torch.zeros(buf_shape, dtype=compute_dtype, device=a_local.device)
        if cp_size > 1:
            recv_buf = torch.empty(buf_shape, dtype=compute_dtype, device=a_local.device)

        a_compute_bwd = a_local.to(compute_dtype)

        # Initial chunk-owner at rank q is (q + 1) % cp_size so that after
        # cp_size - 1 hops the owner pointer lands on q.
        current_owner = (cp_rank + 1) % cp_size
        for step in range(cp_size):
            j_s = current_owner * N_local
            j_e = j_s + N_local
            accum.add_(
                torch.einsum(
                    "bijcd,bisc->bjsd",
                    grad_z_local[:, :, j_s:j_e, :, :].to(compute_dtype),
                    a_compute_bwd,
                )
            )

            if step < cp_size - 1:
                handles = _ring_p2p_send_recv(
                    [accum.contiguous()],
                    [recv_buf],
                    send_to,
                    recv_from,
                    cp_group,
                    parity,
                )
                for h in handles:
                    h.wait()
                accum, recv_buf = recv_buf, accum
                # The buffer just received was the accumulator that rank
                # recv_from = (cp_rank + 1) % cp_size had just finished adding
                # to.  Its current_owner at that moment was
                # ((cp_rank + 1) + (step + 1)) % cp_size = (cp_rank + 2 + step) % cp_size.
                # After the swap, that becomes the current_owner at this rank.
                current_owner = (cp_rank + 2 + step) % cp_size

        grad_b_local = accum.to(b_local.dtype)
        grad_b_local = grad_b_local * mask_t
        # Transpose back
        grad_b_local = grad_b_local.transpose(1, 2).contiguous()

        grad_b = DTensor.from_local(
            grad_b_local,
            device_mesh=ctx.device_mesh,
            placements=MSA_PLACEMENTS_1D,
            shape=ctx.input_shape_b,
            stride=ctx.input_stride_b,
        )

        return grad_a, grad_b, None, None, None


class OuterProductMean1D(nn.Module):
    """Distributed outer product mean for 1D CP (2D mesh ``(dp, cp)``).

    Wraps the serial ``OuterProductMean`` with ring-based communication on
    the cp axis.  Performs LayerNorm -> projections -> ring-rotated OPM ->
    mask normalize -> linear_out.

    Parameters
    ----------
    layer : SerialOuterProductMean
        The serial OuterProductMean module whose weights are replicated.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        The CP process group for ring P2P + all-reduce collectives.
    """

    def __init__(
        self,
        layer: SerialOuterProductMean,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.c_hidden = layer.c_hidden
        self.cp_group = cp_group

        self.norm = LayerNormParamsReplicated(layer.norm, device_mesh)
        self.proj_a = LinearParamsReplicated(layer.proj_a, device_mesh)
        self.proj_b = LinearParamsReplicated(layer.proj_b, device_mesh)
        self.proj_o = LinearParamsReplicated(layer.proj_o, device_mesh)

    def forward(self, m: DTensor, mask: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        m : DTensor
            MSA embedding ``[B, S, N, C_m]`` with placements ``(Shard(0), Shard(2))``.
        mask : DTensor
            MSA mask ``[B, S, N]`` with placements ``(Shard(0), Shard(2))``.

        Returns
        -------
        DTensor
            Pair update ``[B, N, N, C_z]`` with placements ``(Shard(0), Shard(1))``.
        """
        ln = self.norm(m)
        a = self.proj_a(ln)  # [B, S, N, c_hidden]
        b = self.proj_b(ln)  # [B, S, N, c_hidden]

        # Container to receive the mask norm from the autograd function
        norm_container: list[torch.Tensor] = []

        outer = _OuterProductMean1DImpl.apply(a, b, mask, self.cp_group, norm_container)

        # Normalize BEFORE proj_o to match serial order (serial divides by mask
        # norm before the output projection, which matters due to proj_o's bias).
        # norm_local is non-differentiable (detached mask counts), so this division
        # simply scales the gradient flowing into the autograd Function by 1/norm.
        norm_local = norm_container[0]
        outer_local = outer.to_local() / norm_local
        outer = DTensor.from_local(
            outer_local,
            device_mesh=outer.device_mesh,
            placements=outer.placements,
            shape=outer.shape,
            stride=outer.stride(),
        )

        # linear_out: [B, N, N, c_hidden^2] -> [B, N, N, c_z]
        outer = self.proj_o(outer)
        return outer
