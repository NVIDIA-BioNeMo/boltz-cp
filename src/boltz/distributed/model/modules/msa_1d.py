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

"""1D CP MSAModule and MSALayer for 2D mesh ``(dp, cp)``.

Tensor placements on 2D mesh ``(dp, cp)``:

- ``m [B, S, N, C_m]``: ``(Shard(0), Shard(2))`` -- S replicated, N on cp
- ``z [B, N, N, C_z]``: ``(Shard(0), Shard(1))`` -- row-slab: first N on cp
- ``msa_mask [B, S, N]``: ``(Shard(0), Shard(2))``
- ``token_mask [B, N, N]``: ``(Shard(0), Shard(1))``

Communication per MSALayer1D:

- PairWeightedAveraging: ring on cp (P steps) rotating v; bias is local
- MSA transition: local (elementwise)
- OPM: all-gather b on cp (forward), all-gather b + all-reduce db (backward)
- PairformerNoSeqLayer1D (tri-mul, tri-attn, pair transition): ring on cp

Composes existing 1D layer wrappers from ``layers/*_1d.py``.
"""

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.utils.checkpoint import checkpoint

from boltz.data import const
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.cat_and_chunk import shardwise_cat
from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, elementwise_op
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.outer_product_mean_1d import OuterProductMean1D
from boltz.distributed.model.layers.pairformer_1d import PairformerNoSeqLayer1D
from boltz.distributed.model.layers.shardwise_op import shardwise_one_hot
from boltz.distributed.model.layers.squeeze import shardwise_unsqueeze
from boltz.distributed.model.layers.transition_1d import Transition1D
from boltz.distributed.model.layers.utils import _ring_p2p_send_recv
from boltz.distributed.model.modules.utils import get_cpu_offload_context
from boltz.distributed.utils import tiled_softmax_attention_update, update_exhaustive_strides
from boltz.model.layers.pair_averaging import (
    PairWeightedAveraging as SerialPairWeightedAveraging,
)
from boltz.model.modules.trunkv2 import MSALayer as SerialMSALayer
from boltz.model.modules.trunkv2 import MSAModule as SerialMSAModule

MSA_PLACEMENTS_1D = (Shard(0), Shard(2))
PAIR_PLACEMENTS_1D = (Shard(0), Shard(1))


def _apply_msa_dropout_1d(
    x_dt: DTensor,
    dropout: float,
    training: bool,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Apply rowwise dropout to an MSA DTensor under 1D CP.

    MSA repr ``[B, S, N, C_m]`` with placements ``(Shard(0), Shard(2))`` has S
    replicated across CP ranks.  Rowwise dropout mask ``[B, S, 1, 1]`` must be
    identical on all ranks so that the same sequence positions are dropped for
    every N-shard.  Rank 0 generates the mask and broadcasts it.

    Parameters
    ----------
    x_dt : DTensor
        MSA representation with placements ``(Shard(0), Shard(2))``.
    dropout : float
        Dropout rate.
    training : bool
        Whether in training mode.
    cp_group : dist.ProcessGroup
        CP process group for broadcasting the mask.

    Returns
    -------
    DTensor
        Dropout-masked MSA representation.
    """
    if not training:
        return x_dt

    x_local = x_dt.to_local()
    # Match the 2D-CP dropout (`apply_dropout_mask_msa_or_pair`): when autocast
    # is active, keep the mask at fp32 so the resulting `x * mask` lifts to
    # fp32 and matches the serial path's `m + msa_dropout * pwa(...)`
    # behaviour (`get_dropout_mask` always returns fp32, so the addition's
    # output is fp32 under autocast).  Casting the mask down to x_local's
    # dtype (bf16) would produce a bf16 result and break activation dtype
    # parity with the serial trainer.  The mask is generated even at
    # dropout=0.0 (all-ones fp32) so the dtype-lifting behaviour is preserved
    # for the no-dropout configuration too.
    if torch.is_autocast_enabled("cuda"):
        mask_dtype = torch.promote_types(x_local.dtype, torch.float32)
    else:
        mask_dtype = x_local.dtype
    # Rowwise mask: [B, S, 1, 1] — same for all N-shards
    mask_shape = (x_local.shape[0], x_local.shape[1], 1, 1)
    samples = torch.rand(mask_shape, dtype=mask_dtype, device=x_local.device)
    if dist.get_world_size(cp_group) > 1:
        dist.broadcast(samples, src=dist.get_process_group_ranks(cp_group)[0], group=cp_group)
    mask = (samples >= dropout).to(dtype=mask_dtype) / (1.0 - dropout)

    out_local = (x_local * mask).contiguous()
    out_stride = update_exhaustive_strides(out_local.shape, out_local.stride(), x_dt.shape)
    return DTensor.from_local(
        out_local,
        x_dt.device_mesh,
        x_dt.placements,
        shape=x_dt.shape,
        stride=out_stride,
    )


class _PairWeightedAveraging1DTiledImpl(torch.autograd.Function):
    """Tiled-softmax 1D-CP PairWeightedAveraging.

    Ring-rotates v (projected MSA values) over cp while bias b (from pair z)
    is locally available.  Softmax is computed inside this Function per
    ring-step using the online-softmax merge pattern from
    :func:`tiled_softmax_attention_update`, mirroring the 2D-CP analog at
    :mod:`boltz.distributed.model.layers.pair_averaging`.

    The tiled-softmax pattern keeps forward accumulators in the input dtype
    (bf16 under autocast) rather than promoting the o/v ring buffers to fp32,
    which eliminates the three large fp32 buffers (~470 MiB each at
    B=1, S=3722, N_local=123, H=8, d=32, cp=2) that dominated the
    pre-tiled forward peak.  Per-chunk softmax with explicit amax tracking
    keeps the per-step numerics stable.  Backward still uses fp32
    ``w_local`` reconstructed from the saved ``(b_perm, lse_m, amax)``
    (out-of-scope follow-up to defuse the remaining fp32 backward buffers).

    Under 1D CP row-slab sharding:
    - v has shape ``[B, S, N, H*d]`` with placements ``(Shard(0), Shard(2))``
    - b has shape ``[B, N, N, H]`` with placements ``(Shard(0), Shard(1))``
      (full j range, local i range)
    - token_mask has shape ``[B, N, N]`` with placements ``(Shard(0), Shard(1))``

    Communication budget:
        Forward:  cp_size ring steps rotating v (1 tensor per step).
        Backward: cp_size-1 ring hops for dv (reduce-scatter ring on N_local
                  j-chunks); cp_size ring steps for db recompute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        v: DTensor,
        b: DTensor,
        g: DTensor,
        token_mask: DTensor,
        cp_group: dist.ProcessGroup,
        inf: float,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        v : DTensor
            ``[B, S, N, H*d]`` MSA values with placements ``(Shard(0), Shard(2))``.
        b : DTensor
            ``[B, N, N, H]`` pair bias with placements ``(Shard(0), Shard(1))``.
            Local shape is ``[B, N/cp, N, H]`` — full j range, local i range.
        g : DTensor
            ``[B, S, N, H*d]`` gating values with placements ``(Shard(0), Shard(2))``.
        token_mask : DTensor
            ``[B, N, N]`` with placements ``(Shard(0), Shard(1))``.
        cp_group : dist.ProcessGroup
            CP process group.
        inf : float
            Masking value.

        Returns
        -------
        DTensor
            ``[B, S, N, H*d]`` with placements ``(Shard(0), Shard(2))``.
        """
        for name, dt in [("v", v), ("b", b), ("g", g), ("token_mask", token_mask)]:
            if not isinstance(dt, DTensor):
                raise TypeError(f"Expected DTensor for {name}, got {type(dt)}")

        if v.placements != MSA_PLACEMENTS_1D:
            raise ValueError(f"v placements must be {MSA_PLACEMENTS_1D}, got {v.placements}")
        if g.placements != MSA_PLACEMENTS_1D:
            raise ValueError(f"g placements must be {MSA_PLACEMENTS_1D}, got {g.placements}")
        if b.placements != PAIR_PLACEMENTS_1D:
            raise ValueError(f"b placements must be {PAIR_PLACEMENTS_1D}, got {b.placements}")
        if token_mask.placements != PAIR_PLACEMENTS_1D:
            raise ValueError(f"token_mask placements must be {PAIR_PLACEMENTS_1D}, got {token_mask.placements}")

        device_mesh = v.device_mesh
        if b.device_mesh != device_mesh or g.device_mesh != device_mesh or token_mask.device_mesh != device_mesh:
            raise ValueError("v, b, g, token_mask must share the same device_mesh")

        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        parity = cp_rank % 2 == 0

        send_to = (cp_rank - 1) % cp_size
        recv_from = (cp_rank + 1) % cp_size

        v_local = v.to_local()  # [B, S, N_local, H*d]
        b_local = b.to_local()  # [B, N_local, N, H]
        g_local = g.to_local()  # [B, S, N_local, H*d]
        mask_local = token_mask.to_local()  # [B, N_local, N]

        B, S, N_local, Hd = v_local.shape
        num_heads = b_local.shape[-1]
        head_dim = Hd // num_heads
        N_full = b_local.shape[2]

        if N_full != N_local * cp_size:
            raise ValueError(f"Uneven sharding: N_full={N_full}, N_local={N_local}, cp_size={cp_size}")

        # b_perm = pre-softmax masked bias, [B, H, N_local, N_full], in v_local's
        # dtype (bf16 under autocast).  We keep b_perm in input dtype so the
        # saved-for-backward tensor stays bf16; per-chunk softmax inside the
        # ring is computed locally without fp32 promotion, with explicit amax
        # tracking giving full dynamic range coverage (including -inf masked
        # entries).
        b_perm = b_local.permute(0, 3, 1, 2).contiguous()  # [B, H, N_local, N_full]
        mask_bias = (1 - mask_local) * (-inf)  # [B, N_local, N_full]
        b_perm = b_perm + mask_bias[:, None, :, :].to(b_perm.dtype)

        # Reshape v for ring: [B, H, S, N_local, d]
        v_mh = v_local.view(B, S, N_local, num_heads, head_dim).permute(0, 3, 1, 2, 4).contiguous()

        # Save the local v (cp_rank-owned chunk) for backward BEFORE the ring
        # loop overwrites this buffer.  The ring uses v_mh as v_buffer[0]; on
        # subsequent steps the swap rotates incoming chunks INTO this buffer,
        # so by the time the loop ends v_mh has been overwritten with v from
        # rank (cp_rank − 1) mod cp_size, NOT the local rank's v.  The
        # baseline ``v_buffer = [v_mh, empty_like(v_mh)]`` pattern aliases
        # this saved tensor and corrupts it after ``cp_size − 1`` swaps;
        # invisible at cp=2 but fires at cp≥3.  Explicit detach+clone defuses
        # the alias.
        v_mh_local_save = v_mh.detach().clone()

        # Online accumulators, all in v_local.dtype (bf16-class under autocast).
        # tiled_softmax_attention_update expects o-shape (..., D) where the
        # last dim is the feature axis being accumulated.  We fold (S, d) into
        # a single feature axis of size S*d, giving o-shape (B, H, N_local, S*d).
        # lse_m, amax have shape (..., 1) = (B, H, N_local, 1).
        o: torch.Tensor | None = None
        lse_m: torch.Tensor | None = None
        amax: torch.Tensor | None = None

        v_buffer = [v_mh, torch.empty_like(v_mh)]
        i_ready = 0
        i_recv = 1

        for step in range(cp_size):
            if step < cp_size - 1:
                handles = _ring_p2p_send_recv(
                    [v_buffer[i_ready].contiguous()],
                    [v_buffer[i_recv]],
                    send_to,
                    recv_from,
                    cp_group,
                    parity,
                )

            source_rank = (cp_rank + step) % cp_size
            j_start = source_rank * N_local
            j_end = j_start + N_local

            # b_chunk: [B, H, N_local_i, N_local_j]  (a square slab of bias)
            b_chunk = b_perm[:, :, :, j_start:j_end].contiguous()

            # Per-chunk softmax state — input dtype throughout, with explicit
            # amax tracking to give the running merge full dynamic range.
            # amax_chunk, lse_m_chunk: [B, H, N_local_i, 1]
            amax_chunk = b_chunk.amax(dim=-1, keepdim=True)
            # logsumexp internally promotes to fp32 then we cast back to input dtype.
            lse_m_chunk = torch.logsumexp(b_chunk - amax_chunk, dim=-1, keepdim=True).to(b_chunk.dtype)

            # p_chunk: [B, H, N_local_i, N_local_j], input dtype, sums to 1 over last dim.
            p_chunk = torch.softmax(b_chunk, dim=-1)

            # o_block: [B, H, S, N_local_i, d]  ← einsum(p_chunk, v[j_chunk])
            # einsum dims: p[b,h,i,j] * v[b,h,s,j,d] -> o[b,h,s,i,d]
            o_block = torch.einsum("bhij,bhsjd->bhsid", p_chunk, v_buffer[i_ready])

            # Reshape o_block to match tiled_softmax_attention_update contract:
            # (B, H, N_local_i, S*d)  — move N_local_i ahead of (S, d) and flatten (S, d).
            o_block_folded = o_block.permute(0, 1, 3, 2, 4).contiguous().flatten(start_dim=-2)

            o, lse_m, amax = tiled_softmax_attention_update(
                o_block_folded,
                lse_m_chunk,
                amax_chunk,
                o,
                lse_m,
                amax,
            )

            if step < cp_size - 1:
                for h in handles:
                    h.wait()
                i_ready ^= 1
                i_recv ^= 1

        # o: (B, H, N_local_i, S*d).  Reshape back to (B, S, N_local, H*d).
        o_local = (
            o.unflatten(-1, (S, head_dim))  # (B, H, N_local, S, d)
            .permute(0, 3, 2, 1, 4)  # (B, S, N_local, H, d)
            .reshape(B, S, N_local, Hd)
        )

        # Apply gating
        sig_g = g_local.sigmoid()
        o_local = sig_g * o_local

        # Save for backward — keep everything in input dtype; recompute p in backward
        # from (b_perm, lse_m, amax) via p = exp(b_perm - amax - lse_m).
        if v.requires_grad or b.requires_grad or g.requires_grad:
            ctx.save_for_backward(
                v_mh_local_save,  # [B, H, S, N_local, d] — NOT v_mh (corrupted by ring)
                b_perm.detach(),  # [B, H, N_local, N_full]  (post-mask, pre-softmax)
                lse_m.detach(),  # [B, H, N_local, 1]
                amax.detach(),  # [B, H, N_local, 1]
                g_local.detach(),
                o_local.detach(),
            )
            ctx.cp_group = cp_group
            ctx.cp_size = cp_size
            ctx.cp_rank = cp_rank
            ctx.parity = parity
            ctx.send_to = send_to
            ctx.recv_from = recv_from
            ctx.v_requires_grad = v.requires_grad
            ctx.b_requires_grad = b.requires_grad
            ctx.g_requires_grad = g.requires_grad
            ctx.device_mesh = device_mesh
            ctx.msa_placements = v.placements
            ctx.pair_placements = b.placements
            ctx.v_shape = v.shape
            ctx.b_shape = b.shape
            ctx.g_shape = g.shape
            ctx.num_heads = num_heads
            ctx.head_dim = head_dim
            ctx.N_local = N_local
            ctx.S = S

        o_stride = update_exhaustive_strides(o_local.shape, o_local.stride(), v.shape)
        return DTensor.from_local(
            o_local.contiguous(),
            device_mesh,
            v.placements,
            shape=v.shape,
            stride=o_stride,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        v_mh, b_perm, lse_m, amax, g_local, o_local = ctx.saved_tensors
        go_local = grad_output.to_local()  # [B, S, N_local, H*d]

        B, S, N_local, Hd = go_local.shape
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        compute_dtype = torch.promote_types(go_local.dtype, torch.float32)

        # Gradient through gating: o_local = sigmoid(g) * o_ungated
        sig_g = g_local.sigmoid()
        dg_local = go_local * o_local * (1.0 - sig_g) if ctx.g_requires_grad else None
        go_ungated = go_local * sig_g

        # Reshape upstream gradient to multi-head form:
        # [B, S, N_local, H*d] -> [B, H, S, N_local, d]
        go_mh = go_ungated.view(B, S, N_local, num_heads, head_dim).permute(0, 3, 1, 2, 4).to(compute_dtype)

        # Reconstruct full softmax weights from saved (b_perm, lse_m, amax).
        # This mirrors 2D-CP backward at pair_averaging.py:495-507:
        #   p = exp(b - amax - lse_m)
        # b_perm is the pre-softmax masked bias [B, H, N_local, N_full]; lse_m
        # and amax are per-i scalars [B, H, N_local, 1].  Broadcasting over j
        # gives the post-softmax p of shape [B, H, N_local, N_full].  We work
        # in compute_dtype (fp32 or higher) to match the existing backward.
        # The memory budget here is O(B·H·N_local·N) which is identical to the
        # pre-tiled implementation's saved w_local, so we don't regress.
        w_local = (b_perm.to(compute_dtype) - amax.to(compute_dtype) - lse_m.to(compute_dtype)).exp()

        # dv computation — reduce-scatter ring over j-chunks of size N_local.
        #
        # Math: dv[s, j_global, d] = Σ_i w[i, j_global] · do[s, i, d].  Each rank q
        # holds w_local_q[i_local, j_full] (local i row-slab, full j) and
        # go_mh_q[s, i_local, d].  Define
        #
        #     partial[q→r] = einsum(w_local_q[..., r*N_local:(r+1)*N_local], go_mh_q)
        #
        # i.e. rank q's local-i contribution to rank r's j range.  The full dv at
        # rank r is dv_local_r = Σ_q partial[q→r].
        #
        # We accumulate this with a reduce-scatter ring: each rank maintains a
        # single [B, H, S, N_local, d] accumulator that tracks the running sum
        # for one current j-chunk owner; the accumulator is rotated around the
        # ring so each chunk passes through every rank, accumulating each rank's
        # local partial along the way.  Peak memory O(B·H·S·N_local·d) — cp×
        # smaller than a dv_full=[B, H, S, N_full, d] intermediate, matching
        # forward o_acc.  Total communication volume = (cp_size - 1) ring hops
        # of one [B, H, S, N_local, d] tensor (same as the prior all-reduce on
        # the smaller payload, with the bonus of avoiding the full-N partial
        # peak).
        #
        # Ring direction matches forward/dw: send_to = (cp_rank - 1) % cp_size,
        # recv_from = (cp_rank + 1) % cp_size.  Initial chunk owner at rank q is
        # (q + 1) % cp_size; after each hop the owner pointer becomes whatever
        # came in from recv_from, which is one step further "around" the ring.
        # After cp_size - 1 hops, the owner pointer at rank q is q itself, so
        # the accumulator now holds Σ_q partial[q→q] = dv_local_q.
        #
        # Why not a single all-reduce on a [B, H, S, N_local, d] buffer instead?
        # Each rank's local buffer would index DIFFERENT global-j positions (the
        # j-chunk it owns), so element-wise summation across ranks would add
        # partials for disjoint global-j slots — that is incorrect.  The ring
        # routes each partial to the rank that owns its j-chunk.
        grad_v = None
        if ctx.v_requires_grad:
            if ctx.cp_size == 1:
                dv_local = torch.einsum("bhij,bhsid->bhsjd", w_local, go_mh).contiguous()
            else:
                cp_size = ctx.cp_size
                cp_rank = ctx.cp_rank
                buf_shape = (B, ctx.num_heads, S, N_local, ctx.head_dim)
                accum = torch.zeros(buf_shape, dtype=compute_dtype, device=go_mh.device)
                recv_buf = torch.empty(buf_shape, dtype=compute_dtype, device=go_mh.device)

                # Initial chunk-owner at rank q is (q + 1) % cp_size so that
                # after cp_size - 1 hops the owner pointer lands on q.
                current_owner = (cp_rank + 1) % cp_size
                for step in range(cp_size):
                    j_s = current_owner * N_local
                    j_e = j_s + N_local
                    accum.add_(
                        torch.einsum(
                            "bhij,bhsid->bhsjd",
                            w_local[:, :, :, j_s:j_e],
                            go_mh,
                        )
                    )

                    if step < cp_size - 1:
                        handles = _ring_p2p_send_recv(
                            [accum.contiguous()],
                            [recv_buf],
                            ctx.send_to,
                            ctx.recv_from,
                            ctx.cp_group,
                            ctx.parity,
                        )
                        for h in handles:
                            h.wait()
                        accum, recv_buf = recv_buf, accum
                        # The buffer just received was the accumulator that rank
                        # recv_from = (cp_rank + 1) % cp_size had just finished
                        # adding to.  Its current_owner at that moment was
                        # ((cp_rank + 1) + (step + 1)) % cp_size = (cp_rank + 2 +
                        # step) % cp_size.  After the swap, that becomes the
                        # current_owner at this rank.
                        current_owner = (cp_rank + 2 + step) % cp_size

                dv_local = accum

            assert dv_local.shape[3] == N_local, f"dv_local dim-3 must be N_local={N_local}, got {dv_local.shape[3]}"

            dv_local = dv_local.permute(0, 2, 3, 1, 4).reshape(B, S, N_local, Hd).to(go_local.dtype)
            dv_stride = update_exhaustive_strides(dv_local.shape, dv_local.stride(), ctx.v_shape)
            grad_v = DTensor.from_local(
                dv_local.contiguous(),
                ctx.device_mesh,
                ctx.msa_placements,
                shape=ctx.v_shape,
                stride=dv_stride,
            )

        # db computation via ring rotation of v.
        grad_b = None
        if ctx.b_requires_grad:
            dw_local = torch.zeros_like(w_local, dtype=compute_dtype)

            v_ring = v_mh.to(compute_dtype)
            v_buffer = [v_ring, torch.empty_like(v_ring)]
            i_ready = 0
            i_recv = 1

            for step in range(ctx.cp_size):
                if step < ctx.cp_size - 1:
                    handles = _ring_p2p_send_recv(
                        [v_buffer[i_ready].contiguous()],
                        [v_buffer[i_recv]],
                        ctx.send_to,
                        ctx.recv_from,
                        ctx.cp_group,
                        ctx.parity,
                    )

                source_rank = (ctx.cp_rank + step) % ctx.cp_size
                j_start = source_rank * N_local
                j_end = j_start + N_local

                dw_block = torch.einsum("bhsid,bhsjd->bhij", go_mh, v_buffer[i_ready])
                dw_local[:, :, :, j_start:j_end] += dw_block

                if step < ctx.cp_size - 1:
                    for h in handles:
                        h.wait()
                    i_ready ^= 1
                    i_recv ^= 1

            # Softmax backward: db = w * (dw - sum_j(dw * w))
            dw_softmax = w_local * (dw_local - (dw_local * w_local).sum(dim=-1, keepdim=True))

            db_local = dw_softmax.permute(0, 2, 3, 1).to(go_local.dtype)
            db_stride = update_exhaustive_strides(db_local.shape, db_local.stride(), ctx.b_shape)
            grad_b = DTensor.from_local(
                db_local.contiguous(),
                ctx.device_mesh,
                ctx.pair_placements,
                shape=ctx.b_shape,
                stride=db_stride,
            )

        grad_g = None
        if ctx.g_requires_grad and dg_local is not None:
            dg_stride = update_exhaustive_strides(dg_local.shape, dg_local.stride(), ctx.g_shape)
            grad_g = DTensor.from_local(
                dg_local.contiguous(),
                ctx.device_mesh,
                ctx.msa_placements,
                shape=ctx.g_shape,
                stride=dg_stride,
            )

        return grad_v, grad_b, grad_g, None, None, None


class PairWeightedAveraging1DTiled(nn.Module):
    """Tiled-softmax distributed PairWeightedAveraging for 1D CP on a 2D mesh ``(dp, cp)``.

    Under 1D CP, MSA ``m [B, S, N, C_m]`` has placements ``(Shard(0), Shard(2))``
    (S replicated, N sharded on cp) and pair ``z [B, N, N, C_z]`` has placements
    ``(Shard(0), Shard(1))`` (row-slab: first N sharded, second N full).

    The attention weights ``w = softmax(proj_z(z))`` are accumulated locally via
    a tiled online-softmax merge (per-ring-step chunk softmax with amax
    tracking).  The value tensor ``v = proj_m(m)`` is ring-rotated over cp.

    Communication budget:
        Forward:  cp_size ring steps rotating v (1 tensor per step).
        Backward: cp_size-1 ring hops for dv (reduce-scatter ring on N_local
                  j-chunks); cp_size ring steps for db recompute.
    """

    def __init__(
        self,
        layer: SerialPairWeightedAveraging,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialPairWeightedAveraging):
            raise TypeError(f"Expected SerialPairWeightedAveraging, got {type(layer)}")

        self.device_mesh = device_mesh
        self.cp_group = cp_group
        self.c_m = layer.c_m
        self.c_z = layer.c_z
        self.c_h = layer.c_h
        self.num_heads = layer.num_heads
        self.inf = layer.inf

        self.norm_m = LayerNormParamsReplicated(layer.norm_m, device_mesh)
        self.norm_z = LayerNormParamsReplicated(layer.norm_z, device_mesh)
        self.proj_m = LinearParamsReplicated(layer.proj_m, device_mesh)
        self.proj_g = LinearParamsReplicated(layer.proj_g, device_mesh)
        self.proj_z = LinearParamsReplicated(layer.proj_z, device_mesh)
        self.proj_o = LinearParamsReplicated(layer.proj_o, device_mesh)

    def forward(self, m: DTensor, z: DTensor, token_mask: DTensor) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        m : DTensor
            ``[B, S, N, C_m]`` MSA representation with placements
            ``(Shard(0), Shard(2))``.
        z : DTensor
            ``[B, N, N, C_z]`` pair representation with placements
            ``(Shard(0), Shard(1))``.
        token_mask : DTensor
            ``[B, N, N]`` pair mask with placements ``(Shard(0), Shard(1))``.

        Returns
        -------
        DTensor
            ``[B, S, N, C_m]`` with placements ``(Shard(0), Shard(2))``.
        """
        m = self.norm_m(m)
        z = self.norm_z(z)
        v = self.proj_m(m)  # [B, S, N, H*d]
        g = self.proj_g(m)  # [B, S, N, H*d]
        b = self.proj_z(z)  # [B, N, N, H]
        o = _PairWeightedAveraging1DTiledImpl.apply(v, b, g, token_mask, self.cp_group, self.inf)
        return self.proj_o(o)


class MSALayer1D(nn.Module):
    """Distributed MSA layer for 1D CP on a 2D mesh ``(dp, cp)``.

    Input/Output Placements:
    - z: ``(Shard(0), Shard(1))`` - Pair representation ``[B, N, N, C_z]``
    - m: ``(Shard(0), Shard(2))`` - MSA representation ``[B, S, N, C_m]``
    - token_mask: ``(Shard(0), Shard(1))`` - Token pair mask ``[B, N, N]``
    - msa_mask: ``(Shard(0), Shard(2))`` - MSA mask ``[B, S, N]``

    Communication per layer:
    - PairWeightedAveraging: ring on cp (v rotation)
    - MSA transition: local (elementwise)
    - OPM: all-gather on cp
    - PairformerNoSeqLayer1D: ring on cp (tri-mul, tri-attn)
    """

    def __init__(
        self,
        layer: SerialMSALayer,
        dist_manager: DistributedManager,
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialMSALayer):
            raise TypeError(f"Expected SerialMSALayer, got {type(layer)}")

        self.device_mesh = dist_manager.device_mesh
        self.cp_group = dist_manager.group["cp"]
        self.msa_dropout = layer.msa_dropout
        device_mesh = self.device_mesh
        cp_group = self.cp_group

        self.pair_weighted_averaging = PairWeightedAveraging1DTiled(
            layer.pair_weighted_averaging,
            device_mesh,
            cp_group,
        )
        self.msa_transition = Transition1D(layer.msa_transition, device_mesh)
        self.outer_product_mean = OuterProductMean1D(
            layer.outer_product_mean,
            device_mesh,
            cp_group,
        )
        self.pairformer_layer = PairformerNoSeqLayer1D(
            layer.pairformer_layer,
            dist_manager,
        )

    def forward(
        self,
        z: DTensor,
        m: DTensor,
        token_mask: DTensor,
        msa_mask: DTensor,
    ) -> Tuple[DTensor, DTensor]:
        """Forward pass.

        Parameters
        ----------
        z : DTensor
            ``[B, N, N, C_z]`` with placements ``(Shard(0), Shard(1))``.
        m : DTensor
            ``[B, S, N, C_m]`` with placements ``(Shard(0), Shard(2))``.
        token_mask : DTensor
            ``[B, N, N]`` with placements ``(Shard(0), Shard(1))``.
        msa_mask : DTensor
            ``[B, S, N]`` with placements ``(Shard(0), Shard(2))``.

        Returns
        -------
        Tuple[DTensor, DTensor]
            Updated ``(z, m)``.
        """
        # Communication to MSA stack: pair-weighted averaging + MSA transition
        m = elementwise_op(
            m,
            _apply_msa_dropout_1d(
                self.pair_weighted_averaging(m, z, token_mask),
                self.msa_dropout,
                self.training,
                self.cp_group,
            ),
            ElementwiseOp.SUM,
        )
        m = elementwise_op(m, self.msa_transition(m), ElementwiseOp.SUM)

        # Communication to pairwise stack via outer product mean
        z = elementwise_op(z, self.outer_product_mean(m, msa_mask), ElementwiseOp.SUM)

        # Compute pairwise stack using PairformerNoSeqLayer1D
        z = self.pairformer_layer(z=z, pair_mask=token_mask)

        return z, m


class MSAModule1D(nn.Module):
    """Distributed MSA module for 1D CP on a 2D mesh ``(dp, cp)``.

    Wraps the serial ``MSAModule`` for 1D context parallelism.  MSA features
    ``[B, S, N, C]`` have S replicated and N sharded on cp.  Pair features
    ``[B, N, N, C]`` have row-slab sharding (first N on cp).

    Input/Output Placements:
    - z: ``(Shard(0), Shard(1))`` - Pair representation ``[B, N, N, C_z]``
    - emb: ``(Shard(0), Shard(1))`` - Single representation ``[B, N, C_s]``
    - MSA features: ``(Shard(0), Shard(2))`` - ``[B, S, N, ...]``
    - token_pair_pad_mask: ``(Shard(0), Shard(1))`` - ``[B, N, N]``

    Output:
    - z: ``(Shard(0), Shard(1))`` - Updated pair representation

    Communication:
    - PairWeightedAveraging: ring on cp for v rotation
    - OuterProductMean: all-gather on cp
    - PairformerNoSeqLayer1D: ring on cp for tri operations
    """

    def __init__(
        self,
        module: SerialMSAModule,
        dist_manager: DistributedManager,
        cpu_offloading: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(module, SerialMSAModule):
            raise TypeError(f"Expected SerialMSAModule, got {type(module)}")

        self.device_mesh = dist_manager.device_mesh
        self.cp_group = dist_manager.group["cp"]
        device_mesh = self.device_mesh

        self.msa_blocks = module.msa_blocks
        self.msa_dropout = module.msa_dropout
        self.z_dropout = module.z_dropout
        self.use_paired_feature = module.use_paired_feature
        self.subsample_msa = module.subsample_msa
        self.num_subsampled_msa = module.num_subsampled_msa

        if self.subsample_msa:
            raise NotImplementedError(
                "Subsampling MSA at module level is not supported with context parallelism. "
                "The serial MSAModule must be built with subsample_msa=False."
            )

        self.activation_checkpointing = getattr(module, "activation_checkpointing", False)
        self.cpu_offloading = cpu_offloading

        # Projection layers
        self.s_proj = LinearParamsReplicated(module.s_proj, device_mesh)
        self.msa_proj = LinearParamsReplicated(module.msa_proj, device_mesh)

        # MSA layers
        self.layers = nn.ModuleList()
        for serial_layer in module.layers:
            self.layers.append(MSALayer1D(serial_layer, dist_manager))

    def forward(
        self,
        z: DTensor,
        emb: DTensor,
        feats: Dict[str, DTensor],
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        z : DTensor
            ``[B, N, N, C_z]`` with placements ``(Shard(0), Shard(1))``.
        emb : DTensor
            ``[B, N, C_s]`` with placements ``(Shard(0), Shard(1))``.
        feats : Dict[str, DTensor]
            Input features as DTensors.

        Returns
        -------
        DTensor
            Updated pair representation ``z``.
        """
        expected_pair = PAIR_PLACEMENTS_1D
        expected_msa = MSA_PLACEMENTS_1D

        if tuple(z.placements) != expected_pair:
            raise ValueError(f"Expected z placement {expected_pair}, got {z.placements}")
        if tuple(emb.placements) != expected_pair:
            raise ValueError(f"Expected emb placement {expected_pair}, got {emb.placements}")

        # Load MSA features and apply one-hot encoding
        msa = feats["msa"]
        msa = shardwise_one_hot(msa, num_classes=const.num_tokens).to(dtype=z.dtype)
        has_deletion = shardwise_unsqueeze(feats["has_deletion"], -1)
        deletion_value = shardwise_unsqueeze(feats["deletion_value"], -1)
        msa_mask = feats["msa_mask"]
        token_mask = feats["token_pair_pad_mask"]

        # Concatenate MSA features
        feats_to_cat = [msa, has_deletion, deletion_value]
        if self.use_paired_feature:
            is_paired = shardwise_unsqueeze(feats["msa_paired"], -1)
            feats_to_cat.append(is_paired)

        m = shardwise_cat(feats_to_cat, dim=-1)

        # Compute input projections
        m = self.msa_proj(m)
        emb_proj = self.s_proj(emb)

        # Broadcast emb along MSA S dimension.
        # emb_proj has placement (Shard(0), Shard(1)) with shape [B, N, C_m].
        # m has placement (Shard(0), Shard(2)) with shape [B, S, N, C_m].
        # Under 1D CP, emb's N and m's N are both sharded on cp (dim 1 for emb,
        # dim 2 for m), so local shapes align: emb_proj_local [B, N/cp, C_m]
        # can be unsqueezed to [B, 1, N/cp, C_m] and added to
        # m_local [B, S, N/cp, C_m] directly.
        emb_proj_local = emb_proj.to_local().unsqueeze(1)  # [B, 1, N/cp, C_m]
        m_local = m.to_local() + emb_proj_local
        m_stride = update_exhaustive_strides(m_local.shape, m_local.stride(), m.shape)
        m = DTensor.from_local(
            m_local.contiguous(),
            self.device_mesh,
            expected_msa,
            shape=m.shape,
            stride=m_stride,
        )

        # Perform MSA blocks
        if self.activation_checkpointing and self.training:
            if self.cpu_offloading:
                with get_cpu_offload_context(optimized=True):
                    for i in range(self.msa_blocks):
                        z, m = checkpoint(
                            self.layers[i],
                            z,
                            m,
                            token_mask,
                            msa_mask,
                            use_reentrant=False,
                        )
            else:
                for i in range(self.msa_blocks):
                    z, m = checkpoint(
                        self.layers[i],
                        z,
                        m,
                        token_mask,
                        msa_mask,
                        use_reentrant=False,
                    )
        else:
            for i in range(self.msa_blocks):
                z, m = self.layers[i](z, m, token_mask, msa_mask)

        return z
