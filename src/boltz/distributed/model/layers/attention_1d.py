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

"""Distributed AttentionPairBias for 1D CP (2D mesh ``(dp, cp)``).

Self-attention over token dimension N with additive pair bias from the pair
representation z.  Under 1D CP row-slab sharding:

- Q comes from ``s [B, N/cp, C_s]`` — local query shard.
- K/V come from ``s`` or ``k_in`` — rotated on the cp ring.
- Pair bias comes from ``z [B, N/cp, N, C_z]`` (row-slab, full columns).
  At ring step t, K covers N-range ``n_t``, and ``bias[i_local, n_t, H]``
  is a column slice of the local row-slab.  **No bias communication needed.**

This is the biggest simplification vs 2D CP, which requires
``TransposeComm`` + ``AttentionPairBiasComm``.

Communication budget
--------------------
Forward:  cp_size ring steps rotating K, V, mask (3 tensors per step).
Backward: cp_size ring steps + 1 extra rotation (dK, dV return).
Bias:     **0 communication** in forward or backward.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.function import FunctionCtx
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from boltz.distributed.comm import One2OneComm
from boltz.distributed.model.layers.attention_impl import is_power_of_2
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.sigmoid_gate import sigmoid_gate
from boltz.distributed.model.modules.utils import Precision, SDPAWithBiasBackend, setup_tf32_env
from boltz.distributed.utils import tiled_softmax_attention_update, update_exhaustive_strides
from boltz.model.layers.attentionv2 import AttentionPairBias as SerialAttentionPairBias

try:
    from torch.nn.attention.flex_attention import flex_attention

    flex_attention_compiled = torch.compile(flex_attention)
    HAS_FLEX_ATTN = True
except ImportError:
    flex_attention_compiled = None
    HAS_FLEX_ATTN = False


class _RingComm1DAttn:
    """Simple 1D ring communicator for attention K/V/mask rotation.

    Left-shifts tensors on the cp group: rank r sends to (r-1) % cp_size
    and receives from (r+1) % cp_size.

    Parameters
    ----------
    cp_group : dist.ProcessGroup
        The context parallelism process group.
    """

    def __init__(self, cp_group: dist.ProcessGroup) -> None:
        self.group = cp_group
        self.cp_rank = dist.get_rank(cp_group)
        self.cp_size = dist.get_world_size(cp_group)

        send_to = (self.cp_rank - 1) % self.cp_size
        recv_from = (self.cp_rank + 1) % self.cp_size

        # Forward: left-shift K/V/mask
        self.comm_k = One2OneComm(cp_group, send_to, recv_from)
        self.comm_v = One2OneComm(cp_group, send_to, recv_from)
        self.comm_mask = One2OneComm(cp_group, send_to, recv_from)

        # Backward: right-shift K/V/mask to revisit forward steps in reverse
        self.comm_k_bwd = One2OneComm(cp_group, recv_from, send_to)
        self.comm_v_bwd = One2OneComm(cp_group, recv_from, send_to)
        self.comm_mask_bwd = One2OneComm(cp_group, recv_from, send_to)

        # Backward: right-shift dK/dV to return gradients to owners
        self.comm_dk = One2OneComm(cp_group, recv_from, send_to)
        self.comm_dv = One2OneComm(cp_group, recv_from, send_to)


def _get_bias_column_slice(
    pair_bias_local: torch.Tensor,
    step: int,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Extract the column slice of pair bias for the current ring step.

    The local pair bias has shape ``[B, H, N_q, N_full]`` where N_q = N/cp
    and N_full = N.  At ring step ``step``, K corresponds to the chunk
    originally owned by rank ``(cp_rank + step) % cp_size``.

    Parameters
    ----------
    pair_bias_local : Tensor
        ``[B, H, N_q, N_full]`` -- full column range for local rows.
    step : int
        Current ring step (0-based).
    cp_rank : int
        This rank's position in the cp group.
    cp_size : int
        Number of ranks in the cp group.

    Returns
    -------
    Tensor
        ``[B, H, N_q, N_k]`` column slice for the current K-chunk.
    """
    source_rank = (cp_rank + step) % cp_size
    n_k = pair_bias_local.shape[-1] // cp_size
    col_start = source_rank * n_k
    col_end = col_start + n_k
    return pair_bias_local[..., col_start:col_end]


def _ring_attention_forward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    mask_bias_local: torch.Tensor,
    pair_bias_local: Optional[torch.Tensor],
    ring_comm: Optional[_RingComm1DAttn],
    sdpa_with_bias_backend: SDPAWithBiasBackend = SDPAWithBiasBackend.REFERENCE,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Ring attention forward pass.

    Returns ``(o_acc, lse_m_acc, amax_acc, k_or_v_last)`` where the last
    element is the K buffer in its final rotation position (needed for backward).
    """
    cp_size = 1 if ring_comm is None else ring_comm.cp_size
    cp_rank = 0 if ring_comm is None else ring_comm.cp_rank
    compute_dtype = torch.promote_types(q_local.dtype, torch.float32)

    # flex_attention guard: same conditions as 2D CP ring_attention_simple_forward
    head_dim = q_local.shape[-1]
    use_flex_attn = (
        sdpa_with_bias_backend == SDPAWithBiasBackend.TORCH_FLEX_ATTN
        and HAS_FLEX_ATTN
        and q_local.is_cuda
        and q_local.dtype != torch.float64
        and is_power_of_2(head_dim)
        and head_dim >= 16
    )

    k_ring = k_local.contiguous()
    v_ring = v_local.contiguous()
    mask_ring = mask_bias_local.contiguous()

    o_acc = None
    lse_m_acc = None
    amax_acc = None

    i_ready = 0
    i_recv = 1
    k_buffer = [k_ring, torch.empty_like(k_ring)]
    v_buffer = [v_ring, torch.empty_like(v_ring)]
    mask_buffer = [mask_ring, torch.empty_like(mask_ring)]

    # [B, N_q, H, d] -> [B, H, N_q, d]
    q_c = q_local.to(compute_dtype).permute(0, 2, 1, 3)

    for step in range(cp_size):
        if ring_comm is not None and step < cp_size - 1:
            k_buffer[i_recv] = ring_comm.comm_k.enqueue_to_dispatch(k_buffer[i_ready], k_buffer[i_recv])
            v_buffer[i_recv] = ring_comm.comm_v.enqueue_to_dispatch(v_buffer[i_ready], v_buffer[i_recv])
            mask_buffer[i_recv] = ring_comm.comm_mask.enqueue_to_dispatch(mask_buffer[i_ready], mask_buffer[i_recv])

        k_c = k_buffer[i_ready].to(compute_dtype).permute(0, 2, 1, 3)
        v_c = v_buffer[i_ready].to(compute_dtype).permute(0, 2, 1, 3)
        m_c = mask_buffer[i_ready].to(compute_dtype)

        if use_flex_attn:
            # Build combined mask + pair bias tensor for score_mod.
            # m_c is (B, 1, 1, N_k); expand to (B, H, N_q, N_k) so that
            # score_mod can index with (b, h, q_idx, kv_idx) directly.
            B, H, N_q, D = q_c.shape
            N_k = k_c.shape[2]
            if pair_bias_local is not None:
                bias_slice = _get_bias_column_slice(pair_bias_local, step, cp_rank, cp_size)
                bias_and_mask = m_c.expand(B, H, N_q, N_k) + bias_slice.to(compute_dtype)
            else:
                bias_and_mask = m_c.expand(B, H, N_q, N_k).contiguous()

            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias_and_mask[b, h, q_idx, kv_idx]

            # scale=1.0 because queries are already scaled by 1/sqrt(d)
            # in the autograd forward before entering this function.
            # setup_tf32_env ensures full FP32 precision (no TF32 rounding).
            with setup_tf32_env(Precision.FP32), torch.amp.autocast("cuda", enabled=False):
                block_o, aux_data = flex_attention_compiled(
                    q_c, k_c, v_c, score_mod=score_mod, return_lse=True, scale=1.0
                )

            # aux_data is (B, H, N_q), reshape to (B, H, N_q, 1) for tiled_softmax_attention_update
            block_lse_m = aux_data.unsqueeze(-1)
            block_amax = None
        else:
            attn = torch.matmul(q_c, k_c.transpose(-1, -2))
            attn = attn + m_c

            if pair_bias_local is not None:
                bias_slice = _get_bias_column_slice(pair_bias_local, step, cp_rank, cp_size)
                attn = attn + bias_slice.to(compute_dtype)

            block_amax = attn.amax(dim=-1, keepdim=True)
            block_lse_m = torch.logsumexp(attn - block_amax, dim=-1, keepdim=True)
            block_o = torch.matmul(torch.softmax(attn, dim=-1), v_c)

        o_acc, lse_m_acc, amax_acc = tiled_softmax_attention_update(
            block_o, block_lse_m, block_amax, o_acc, lse_m_acc, amax_acc
        )

        if ring_comm is not None and step < cp_size - 1:
            ring_comm.comm_k.wait_until_finished()
            ring_comm.comm_v.wait_until_finished()
            ring_comm.comm_mask.wait_until_finished()
            i_ready ^= 1
            i_recv ^= 1

    # o_acc is [B, H, N_q, d], convert back to [B, N_q, H, d]
    o_out = o_acc.permute(0, 2, 1, 3).to(q_local.dtype)

    return o_out, o_acc, amax_acc, lse_m_acc, k_buffer[i_ready], v_buffer[i_ready], mask_buffer[i_ready]


def _ring_attention_backward(
    grad_output: torch.Tensor,
    q_local: torch.Tensor,
    k_ready: torch.Tensor,
    v_ready: torch.Tensor,
    mask_ready: torch.Tensor,
    pair_bias_local: Optional[torch.Tensor],
    o_acc: torch.Tensor,
    amax_acc: Optional[torch.Tensor],
    lse_m_acc: torch.Tensor,
    ring_comm: Optional[_RingComm1DAttn],
    input_dtype: torch.dtype,
    q_requires_grad: bool,
    k_requires_grad: bool,
    v_requires_grad: bool,
    bias_requires_grad: bool,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Ring attention backward pass."""
    cp_size = 1 if ring_comm is None else ring_comm.cp_size
    cp_rank = 0 if ring_comm is None else ring_comm.cp_rank
    compute_dtype = torch.promote_types(input_dtype, torch.float32)
    has_pair_bias = pair_bias_local is not None

    do = grad_output.to(compute_dtype).permute(0, 2, 1, 3)
    d = torch.linalg.vecdot(do, o_acc.to(compute_dtype), dim=-1).unsqueeze(-1)

    q_c = q_local.to(compute_dtype).permute(0, 2, 1, 3)
    dq = torch.zeros_like(q_c)

    if has_pair_bias and bias_requires_grad:
        d_pair_bias = torch.zeros_like(pair_bias_local, dtype=compute_dtype)
    else:
        d_pair_bias = None

    i_ready = 0
    i_recv = 1
    k_buffer = [k_ready, torch.empty_like(k_ready)]
    v_buffer = [v_ready, torch.empty_like(v_ready)]
    mask_buffer = [mask_ready, torch.empty_like(mask_ready)]
    dk_buffer = [
        torch.empty(k_ready.shape, dtype=compute_dtype, device=k_ready.device),
        torch.empty(k_ready.shape, dtype=compute_dtype, device=k_ready.device),
    ]
    dv_buffer = [
        torch.empty(v_ready.shape, dtype=compute_dtype, device=v_ready.device),
        torch.empty(v_ready.shape, dtype=compute_dtype, device=v_ready.device),
    ]

    for step in range(cp_size):
        is_last_step = step == cp_size - 1
        # Right-shift K/V/mask to revisit forward steps in reverse order.
        # Forward left-shifted: rank r saw K_r, K_(r+1), K_(r+2), ...
        # After forward, buffer holds K from the last step.  Right-shifting
        # walks backward through the forward sequence.
        if not is_last_step and ring_comm is not None:
            k_buffer[i_recv] = ring_comm.comm_k_bwd.enqueue_to_dispatch(k_buffer[i_ready], k_buffer[i_recv])
            v_buffer[i_recv] = ring_comm.comm_v_bwd.enqueue_to_dispatch(v_buffer[i_ready], v_buffer[i_recv])
            mask_buffer[i_recv] = ring_comm.comm_mask_bwd.enqueue_to_dispatch(mask_buffer[i_ready], mask_buffer[i_recv])

        fwd_step = (cp_size - 1 - step) % cp_size

        k_c = k_buffer[i_ready].to(compute_dtype).permute(0, 2, 1, 3)
        a = torch.matmul(q_c, k_c.transpose(-1, -2))
        a = a + mask_buffer[i_ready].to(compute_dtype)

        if has_pair_bias:
            bias_slice = _get_bias_column_slice(pair_bias_local, fwd_step, cp_rank, cp_size)
            a = a + bias_slice.to(compute_dtype)

        if amax_acc is not None:
            a = a - amax_acc.to(compute_dtype)
        a = a - lse_m_acc.to(compute_dtype)
        a = torch.exp(a)

        v_c = v_buffer[i_ready].to(compute_dtype).permute(0, 2, 1, 3)

        dv_block = torch.matmul(a.transpose(-1, -2), do)
        da = torch.matmul(do, v_c.transpose(-1, -2))
        ds = a * (da - d)
        dq = dq + torch.matmul(ds, k_c)
        dk_block = torch.matmul(ds.transpose(-1, -2), q_c)

        if d_pair_bias is not None:
            source_rank = (cp_rank + fwd_step) % cp_size
            n_k = pair_bias_local.shape[-1] // cp_size
            col_start = source_rank * n_k
            col_end = col_start + n_k
            d_pair_bias[..., col_start:col_end] += ds

        dk_block_t = dk_block.permute(0, 2, 1, 3).contiguous()
        dv_block_t = dv_block.permute(0, 2, 1, 3).contiguous()

        if step == 0:
            dk_buffer[i_ready] = dk_block_t
            dv_buffer[i_ready] = dv_block_t
        else:
            if ring_comm is not None:
                ring_comm.comm_dk.wait_until_finished()
                ring_comm.comm_dv.wait_until_finished()
            dk_buffer[i_ready] = dk_buffer[i_ready] + dk_block_t
            dv_buffer[i_ready] = dv_buffer[i_ready] + dv_block_t

        if not is_last_step and ring_comm is not None:
            dk_buffer[i_recv] = ring_comm.comm_dk.enqueue_to_dispatch(
                dk_buffer[i_ready].contiguous(), dk_buffer[i_recv]
            )
            dv_buffer[i_recv] = ring_comm.comm_dv.enqueue_to_dispatch(
                dv_buffer[i_ready].contiguous(), dv_buffer[i_recv]
            )

        if not is_last_step and ring_comm is not None:
            ring_comm.comm_k_bwd.wait_until_finished()
            ring_comm.comm_v_bwd.wait_until_finished()
            ring_comm.comm_mask_bwd.wait_until_finished()
            i_ready ^= 1
            i_recv ^= 1

    # dq is the gradient w.r.t. the SCALED query (q / sqrt(d)).
    # Since the scaling is applied inside the autograd function's forward,
    # we must chain-rule through it: dL/dq_original = dL/dq_scaled / sqrt(d).
    head_dim = q_local.shape[-1]
    dq = dq / (head_dim**0.5)

    dq_out = dq.permute(0, 2, 1, 3)
    dk_final = dk_buffer[i_ready]
    dv_final = dv_buffer[i_ready]

    grad_q = dq_out.to(input_dtype) if q_requires_grad else None
    grad_k = dk_final.to(input_dtype) if k_requires_grad else None
    grad_v = dv_final.to(input_dtype) if v_requires_grad else None
    grad_bias = d_pair_bias.to(input_dtype) if (has_pair_bias and bias_requires_grad) else None

    return grad_q, grad_k, grad_v, grad_bias


class _AttentionPairBias1DImpl(torch.autograd.Function):
    """Ring attention over N for AttentionPairBias under 1D CP.

    Accepts DTensors, converts to local tensors internally, runs ring
    attention, and returns a DTensor.  This ensures gradient flow is
    maintained through the DTensor autograd graph.

    Communication budget
    --------------------
    Forward:  cp_size ring steps (K, V, mask).
    Backward: cp_size ring steps (K, V, mask + dK, dV reverse).
    Bias:     0 communication.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        q: DTensor,
        k: DTensor,
        v: DTensor,
        z_bias: DTensor,
        mask: DTensor,
        ring_comm: Optional[_RingComm1DAttn],
        num_heads: int,
        head_dim: int,
        inf: float,
        compute_pair_bias: bool,
        sdpa_with_bias_backend: SDPAWithBiasBackend = SDPAWithBiasBackend.REFERENCE,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        q, k, v : DTensor
            ``[B, N, C_s]`` projected query/key/value with placements
            ``(Shard(0), Shard(1))``.
        z_bias : DTensor
            ``[B, N, N, H]`` projected pair bias with placements
            ``(Shard(0), Shard(1))``.
        mask : DTensor
            ``[B, N]`` token mask with placements ``(Shard(0), Shard(1))``.
        ring_comm : _RingComm1DAttn or None
            Ring communicator.
        num_heads, head_dim : int
            Attention config.
        inf : float
            Masking constant.
        compute_pair_bias : bool
            Whether z_bias is meaningful.
        sdpa_with_bias_backend : SDPAWithBiasBackend
            Attention backend. TORCH_FLEX_ATTN uses compiled flex_attention
            when available; REFERENCE uses manual matmul.
        """
        # Validate DTensor inputs
        for name, dt in [("q", q), ("k", k), ("v", v), ("z_bias", z_bias), ("mask", mask)]:
            if not isinstance(dt, DTensor):
                raise TypeError(f"Expected DTensor for {name}, got {type(dt)}")
        for name, dt in [("k", k), ("v", v), ("z_bias", z_bias), ("mask", mask)]:
            if dt.device_mesh != q.device_mesh:
                raise ValueError(f"{name} device_mesh differs from q")

        # Validate mask shape: 1D CP expects [B, N] aligned with q.shape[:2]
        if mask.ndim != 2:
            raise ValueError(f"mask must have 2 dimensions (B, N), but got mask.ndim={mask.ndim}")
        if mask.shape != q.shape[:2]:
            raise ValueError(
                f"mask.shape must equal q.shape[:2], but got mask.shape={mask.shape} " f"and q.shape[:2]={q.shape[:2]}"
            )

        # Assert expected placements
        expected_single = (Shard(0), Shard(1))
        for name, dt in [("q", q), ("k", k), ("v", v)]:
            if tuple(dt.placements) != expected_single:
                raise ValueError(f"{name} placements {dt.placements} != expected {expected_single}")
        expected_pair = (Shard(0), Shard(1))
        if tuple(z_bias.placements) != expected_pair:
            raise ValueError(f"z_bias placements {z_bias.placements} != expected {expected_pair}")

        # Assert even sharding on cp dimension
        cp_mesh_size = q.device_mesh.size(1)
        for name, dt, dim in [("q", q, 1), ("k", k, 1), ("v", v, 1), ("z_bias", z_bias, 1), ("mask", mask, 1)]:
            if dt.shape[dim] % cp_mesh_size != 0:
                raise ValueError(f"{name} shape[{dim}]={dt.shape[dim]} not divisible by cp_size={cp_mesh_size}")

        # Store metadata for backward
        ctx.device_mesh = q.device_mesh
        ctx.single_placements = q.placements
        ctx.pair_placements = z_bias.placements
        ctx.mask_placements = mask.placements
        ctx.q_shape = q.shape
        ctx.k_shape = k.shape
        ctx.v_shape = v.shape
        ctx.z_shape = z_bias.shape

        # Extract local tensors
        q_local = q.to_local()
        k_local = k.to_local()
        v_local = v.to_local()
        z_local = z_bias.to_local()
        mask_local = mask.to_local()

        B = q_local.shape[0]
        N_local = q_local.shape[1]
        c_s = q_local.shape[2]

        # Reshape for multi-head: [B, N_local, C_s] -> [B, N_local, H, d]
        q_mh = q_local.view(B, N_local, num_heads, head_dim)
        k_mh = k_local.view(B, N_local, num_heads, head_dim)
        v_mh = v_local.view(B, N_local, num_heads, head_dim)

        # Scale query
        q_mh = q_mh / (head_dim**0.5)

        # Prepare pair bias: [B, N/cp, N, H] -> [B, H, N/cp, N]
        pair_bias_local = z_local.permute(0, 3, 1, 2).contiguous()

        # Prepare mask bias: [B, N_local] -> [B, 1, 1, N_local]
        mask_bias_local = (inf * (mask_local.to(q_local.dtype) - 1))[:, None, None, :]

        # Run ring attention
        with torch.autocast("cuda", enabled=False):
            o_mh, o_acc, amax_acc, lse_m_acc, k_last, v_last, mask_last = _ring_attention_forward(
                q_mh,
                k_mh,
                v_mh,
                mask_bias_local,
                pair_bias_local,
                ring_comm,
                sdpa_with_bias_backend,
            )

        # Flatten heads: [B, N_local, H, d] -> [B, N_local, C_s]
        o_local = o_mh.reshape(B, N_local, c_s)

        # Save for backward -- save_for_backward does not accept None,
        # so use a scalar sentinel for absent tensors.
        _sentinel = torch.tensor(0.0, device=q_local.device)
        requires_grad = q.requires_grad or k.requires_grad or v.requires_grad or z_bias.requires_grad
        if requires_grad:
            ctx.save_for_backward(
                q_mh.detach(),
                k_last.detach(),
                v_last.detach(),
                mask_last.detach(),
                pair_bias_local.detach(),
                o_acc.detach(),
                amax_acc.detach() if amax_acc is not None else _sentinel,
                lse_m_acc.detach(),
            )
        ctx.ring_comm = ring_comm
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.c_s = c_s
        ctx.has_amax = amax_acc is not None
        ctx.q_requires_grad = q.requires_grad
        ctx.k_requires_grad = k.requires_grad
        ctx.v_requires_grad = v.requires_grad
        ctx.bias_requires_grad = z_bias.requires_grad
        ctx.input_dtype = q_local.dtype

        # Wrap output as DTensor with correct global strides
        o_stride = update_exhaustive_strides(o_local.shape, o_local.stride(), q.shape)
        o_dt = DTensor.from_local(
            o_local,
            ctx.device_mesh,
            ctx.single_placements,
            shape=q.shape,
            stride=o_stride,
        )
        return o_dt

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, grad_output: DTensor):
        """Backward pass with ring communication."""
        q_mh, k_last, v_last, mask_last, pair_bias_saved, o_acc, amax_saved, lse_m_acc = ctx.saved_tensors

        grad_output_local = grad_output.to_local()
        # [B, N_local, C_s] -> [B, N_local, H, d]
        go_mh = grad_output_local.view(
            grad_output_local.shape[0], grad_output_local.shape[1], ctx.num_heads, ctx.head_dim
        )

        grad_q, grad_k, grad_v, grad_bias = _ring_attention_backward(
            go_mh,
            q_mh,
            k_last,
            v_last,
            mask_last,
            pair_bias_saved,
            o_acc,
            amax_saved if ctx.has_amax else None,
            lse_m_acc,
            ctx.ring_comm,
            ctx.input_dtype,
            ctx.q_requires_grad,
            ctx.k_requires_grad,
            ctx.v_requires_grad,
            ctx.bias_requires_grad,
        )

        # Reshape gradients back to [B, N_local, C_s] and wrap as DTensors
        def wrap_single(g, global_shape):
            if g is None:
                return None
            g_flat = g.reshape(g.shape[0], g.shape[1], ctx.c_s)
            g_stride = update_exhaustive_strides(g_flat.shape, g_flat.stride(), global_shape)
            return DTensor.from_local(
                g_flat, ctx.device_mesh, ctx.single_placements, shape=global_shape, stride=g_stride
            )

        dq_dt = wrap_single(grad_q, ctx.q_shape)
        dk_dt = wrap_single(grad_k, ctx.k_shape)
        dv_dt = wrap_single(grad_v, ctx.v_shape)

        if grad_bias is not None:
            # grad_bias is [B, H, N/cp, N], permute back to [B, N/cp, N, H]
            grad_bias_perm = grad_bias.permute(0, 2, 3, 1)
            dz_stride = update_exhaustive_strides(grad_bias_perm.shape, grad_bias_perm.stride(), ctx.z_shape)
            dz_dt = DTensor.from_local(
                grad_bias_perm, ctx.device_mesh, ctx.pair_placements, shape=ctx.z_shape, stride=dz_stride
            )
        else:
            dz_dt = None

        # Return grads for: q, k, v, z_bias, mask, ring_comm, num_heads, head_dim, inf, compute_pair_bias, sdpa_with_bias_backend
        return dq_dt, dk_dt, dv_dt, dz_dt, None, None, None, None, None, None, None


class AttentionPairBias1D(nn.Module):
    """Distributed AttentionPairBias for 1D CP on a 2D mesh ``(dp, cp)``.

    Wraps the serial ``AttentionPairBias`` module.  Under 1D CP, ring attention
    rotates K/V/mask on the cp ring while pair bias is accessed via local column
    slices -- no bias communication needed.

    Parameters
    ----------
    layer : SerialAttentionPairBias
        Serial attention pair bias module (from ``boltz.model.layers.attentionv2``).
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        The CP process group for ring communication.
    """

    def __init__(
        self,
        layer: SerialAttentionPairBias,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
        sdpa_with_bias_backend: SDPAWithBiasBackend = SDPAWithBiasBackend.REFERENCE,
    ) -> None:
        super().__init__()
        if not isinstance(layer, SerialAttentionPairBias):
            raise TypeError(
                f"Expected SerialAttentionPairBias (from boltz.model.layers.attentionv2), " f"got {type(layer)}."
            )
        if not isinstance(device_mesh, DeviceMesh):
            raise TypeError(f"Expected DeviceMesh, got {type(device_mesh)}.")

        self.device_mesh = device_mesh
        self.c_s = layer.c_s
        self.num_heads = layer.num_heads
        self.head_dim = layer.head_dim
        self.inf = layer.inf
        self.compute_pair_bias = layer.compute_pair_bias

        # Mutable backend selection for scaled dot-product attention.
        # Default is REFERENCE. To switch backend for the entire model, use
        # ``model.apply(SetAttnPairBiasBackend(backend))``
        self.sdpa_with_bias_backend = (
            sdpa_with_bias_backend
            if isinstance(sdpa_with_bias_backend, SDPAWithBiasBackend)
            else SDPAWithBiasBackend(sdpa_with_bias_backend)
        )
        if self.sdpa_with_bias_backend not in [
            SDPAWithBiasBackend.TORCH_FLEX_ATTN,
            SDPAWithBiasBackend.REFERENCE,
        ]:
            raise ValueError(
                f"Unsupported sdpa_with_bias_backend for 1D CP: "
                f"{self.sdpa_with_bias_backend}. "
                f"Only TORCH_FLEX_ATTN and REFERENCE are supported."
            )

        self.ring_comm = _RingComm1DAttn(cp_group)

        self.proj_q = LinearParamsReplicated(layer.proj_q, device_mesh)
        self.proj_k = LinearParamsReplicated(layer.proj_k, device_mesh)
        self.proj_v = LinearParamsReplicated(layer.proj_v, device_mesh)
        self.proj_g = LinearParamsReplicated(layer.proj_g, device_mesh)
        self.proj_o = LinearParamsReplicated(layer.proj_o, device_mesh)

        # proj_z: strip the Rearrange; permute is done manually in the autograd function.
        if self.compute_pair_bias:
            self.proj_z = nn.Sequential(
                LayerNormParamsReplicated(layer.proj_z[0], device_mesh),
                LinearParamsReplicated(layer.proj_z[1], device_mesh),
            )

    def forward(
        self,
        s: DTensor,
        z: DTensor,
        mask: DTensor,
        k_in: Optional[DTensor] = None,
    ) -> DTensor:
        """Forward pass with ring attention over N (cp).

        Parameters
        ----------
        s : DTensor
            ``[B, N, C_s]`` with placements ``(Shard(0), Shard(1))``.
        z : DTensor
            ``[B, N, N, C_z]`` or ``[B, N, N, H]`` with placements
            ``(Shard(0), Shard(1))``.
        mask : DTensor
            ``[B, N]`` with placements ``(Shard(0), Shard(1))``.
        k_in : DTensor or None
            Key input. If None, k_in = s.

        Returns
        -------
        DTensor
            ``[B, N, C_s]`` same placements as ``s``.
        """
        if k_in is None:
            k_in = s

        # Project q, k, v, g
        q_dt: DTensor = self.proj_q(s)
        k_dt: DTensor = self.proj_k(k_in)
        v_dt: DTensor = self.proj_v(k_in)
        g_dt: DTensor = self.proj_g(s)

        # Project z to pair bias
        if self.compute_pair_bias:
            z_proj: DTensor = self.proj_z(z)
        else:
            z_proj = z

        # Ring attention (autograd function bridges DTensor <-> local)
        o_dt = _AttentionPairBias1DImpl.apply(
            q_dt,
            k_dt,
            v_dt,
            z_proj,
            mask,
            self.ring_comm,
            self.num_heads,
            self.head_dim,
            self.inf,
            self.compute_pair_bias,
            self.sdpa_with_bias_backend,
        )

        # Gate and output project
        gated_o: DTensor = sigmoid_gate(x=o_dt, g=g_dt)
        o: DTensor = self.proj_o(gated_o)

        return o
