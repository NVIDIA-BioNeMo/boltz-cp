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

"""Distributed triangular attention (starting + ending node) for 1D context parallelism.

Triangular attention operates on pair representations ``[B, I, J, C_in]``:

- **Starting node** (Algorithm 13): self-attention along J for each row I.
  Under 1D CP with placements ``(Shard(0), Shard(1))``, pair z is row-slab
  ``[B, I/cp, J, C_in]``.  Q/K/V are local (J is full).  The triangle bias
  has shape ``[B, 1, H, I, J]`` in the serial code, where ``I == J == N``.
  Under row-slab sharding the projected bias is ``[B, I/cp, J, H]``; its
  I/cp dimension maps to the attention's J_q dimension (= N).  We all-gather
  the bias slab once to assemble ``[B, 1, H, N, N]`` and run a single
  attention pass.  Both REFERENCE and CUEQ consume the full ``[B, 1, H, N, N]``
  bias directly (Q/K/V have the full J=N sequence dim, so S_qo = S_kv = N).

- **Ending node** (Algorithm 14): transposes I<->J, attends along I for each
  column J, then transposes back.  Both backends share the DAP path
  implemented by :class:`_DAPTriangleAttentionEndingNode1DImpl` (NCCL-only):

  * **REFERENCE backend.**  Re-shards q/k/v/mask once via all-to-all
    (row-slab ``[B, J=N, H, I/cp, D]`` → col-slab
    ``[B, J/cp, H, I=N, D]``), all-gathers the triangle bias's I_k axis to
    full ``[B, 1, H, J=N, I_k=N]`` (an a2a-flip is mathematically valid on
    the bias but produces a layout whose J/cp axis sits at dim 3, mismatching
    the attn's J/cp at dim 1; all-gather keeps J full at dim 3 and broadcasts
    via the existing singleton at dim 1 with no extra transpose), runs a
    *local* full ``N×N`` softmax attention (no online accumulation, no
    ring), then all-to-alls the output back to row-slab.

  * **CUEQ backend.**  Same re-shard + bias all-gather as REFERENCE.  The
    CUEQ kernel's "bias dim-3 == full N" constraint is already satisfied
    by the gathered ``[B, 1, H, J=N, I_k=N]`` bias (dim 3 = J = N), so the
    kernel can be called directly on the col-slab Q/K/V with no extra
    transpose.  Backward reduce-scatters the dbias on its I_k axis (dim -1)
    just like REFERENCE.  See Item #5 of the DAP migration plan.

Communication budget (forward):
    - Starting node: 1 all-gather of the bias slab along dim -2 (full N).
    - Ending node (REFERENCE and CUEQ, DAP): 5 a2a's (q, k, v, mask, output) +
      1 all-gather on the triangle bias's sharded I_k dim.  6 collectives total.

Backward:
    - Starting node: 1 all-gather to re-assemble the bias + 1 reduce-scatter
      of dbias along dim -2.
    - Ending node (REFERENCE and CUEQ, DAP): 1 all-gather (re-materialize
      bias_col transiently; the ctx only saves the row-slab to obey the
      O(N^2/cp) backward-memory budget) + 4 a2a's (grad_output + dq, dk, dv)
      + 1 reduce-scatter (dbias) on the I_k dim.  6 collectives total.  No
      ring, no recompute beyond the local softmax-attention backward.  The
      dbias reduce-scatter runs in fp32 (CLAUDE.md "Reduce gradients in
      fp32"); the result is cast to input dtype after the collective.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd.function import FunctionCtx
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from boltz.distributed.model.layers.layernorm import _LayerNormParamsReplicatedImpl
from boltz.distributed.model.layers.linear import _LinearParamsReplicatedImpl
from boltz.distributed.model.layers.triangular_attention import can_run_cueq_triattn_sm100f
from boltz.distributed.model.modules.utils import TriAttnBackend
from boltz.distributed.utils import update_exhaustive_strides
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode as SerialTriangleAttentionEndingNode,
)
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionStartingNode as SerialTriangleAttentionStartingNode,
)
from boltz.model.layers.triangular_attention.primitives import LayerNorm as SerialLayerNormNoAutoCastBF16
from boltz.model.layers.triangular_attention.primitives import Linear as SerialLinearNoAutoCastBF16
from boltz.model.layers.triangular_attention.utils import permute_final_dims

try:
    import cuequivariance_torch.primitives.triangle as cueq_triangle

    cueq_is_installed = True
except ImportError:
    cueq_is_installed = False


class LayerNormParamsReplicatedNoAutoCastBF16(nn.Module):
    """LayerNorm with replicated parameters and disabled autocast in BF16.

    Wraps the serial ``LayerNorm`` from the triangular attention primitives
    (which disables autocast for BF16) and distributes its parameters as
    replicated DTensors.

    Parameters
    ----------
    layer_local : SerialLayerNormNoAutoCastBF16
        An already-initialized LayerNorm instance.
    device_mesh : DeviceMesh
        The device mesh for distributed training.
    """

    def __init__(self, layer_local: SerialLayerNormNoAutoCastBF16, device_mesh: DeviceMesh) -> None:
        if not isinstance(layer_local, SerialLayerNormNoAutoCastBF16):
            raise ValueError(
                f"layer_local is not an instance of SerialLayerNormNoAutoCastBF16 but got {type(layer_local)}"
            )
        if layer_local.weight.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.weight and device_mesh are not on the same device type: "
                f"{layer_local.weight.device.type} != {device_mesh.device_type}"
            )
        if layer_local.bias is not None and layer_local.bias.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.bias and device_mesh are not on the same device type: "
                f"{layer_local.bias.device.type} != {device_mesh.device_type}"
            )

        super().__init__()
        self.c_in = layer_local.c_in
        self.normalized_shape = list(self.c_in)
        self.eps = layer_local.eps
        self.device_mesh = device_mesh

        all_replicate_placements = [Replicate()] * device_mesh.ndim

        if layer_local.weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(
                distribute_tensor(layer_local.weight.data, device_mesh, all_replicate_placements)
            )
        if layer_local.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(distribute_tensor(layer_local.bias.data, device_mesh, all_replicate_placements))

    def forward(self, x: DTensor) -> DTensor:
        d = x.dtype
        if d is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                out = _LayerNormParamsReplicatedImpl.apply(
                    x, self.normalized_shape, self.weight, self.bias, self.eps, True
                )
        else:
            out = _LayerNormParamsReplicatedImpl.apply(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return out


class LinearParamsReplicatedNoAutoCastBF16(nn.Module):
    """Linear layer with replicated parameters and disabled autocast in BF16.

    Wraps the serial ``Linear`` from the triangular attention primitives
    and distributes its parameters as replicated DTensors.

    Parameters
    ----------
    layer_local : SerialLinearNoAutoCastBF16
        An already-initialized Linear instance.
    device_mesh : DeviceMesh
        The device mesh for distributed training.
    """

    def __init__(self, layer_local: SerialLinearNoAutoCastBF16, device_mesh: DeviceMesh):
        if not isinstance(layer_local, SerialLinearNoAutoCastBF16):
            raise ValueError(
                f"layer_local is not an instance of SerialLinearNoAutoCastBF16 but got {type(layer_local)}"
            )
        if layer_local.weight.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.weight and device_mesh are not on the same device type: "
                f"{layer_local.weight.device.type} != {device_mesh.device_type}"
            )
        if layer_local.bias is not None and layer_local.bias.device.type != device_mesh.device_type:
            raise ValueError(
                f"layer_local.bias and device_mesh are not on the same device type: "
                f"{layer_local.bias.device.type} != {device_mesh.device_type}"
            )
        super().__init__()
        all_replicate_placements = [Replicate()] * device_mesh.ndim
        self.weight = nn.Parameter(distribute_tensor(layer_local.weight.data, device_mesh, all_replicate_placements))
        if layer_local.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(distribute_tensor(layer_local.bias.data, device_mesh, all_replicate_placements))

    def forward(self, input: DTensor) -> DTensor:
        d = input.dtype
        if d is torch.bfloat16:
            with torch.autocast("cuda", enabled=False):
                return _LinearParamsReplicatedImpl.apply(input, self.weight, self.bias, True)
        else:
            return _LinearParamsReplicatedImpl.apply(input, self.weight, self.bias)


def _promote_compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote dtype to at least float32 for compute precision."""
    return torch.promote_types(dtype, torch.float32)


def _all_gather_on_cp_async(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> tuple[Optional[list], Optional[object], torch.Tensor]:
    """Fire an async all-gather along the cp group; return (shards, work, local).

    The DAO (dynamic-axial-overlap) prologue: issue the all-gather with
    ``async_op=True`` and return immediately WITHOUT the ``torch.cat``, so the
    caller can run independent compute (e.g. the ``q @ k^T`` score matmul) on
    the compute stream while the gather's wire-time elapses, then call
    :func:`_all_gather_on_cp_wait` to ``work.wait()`` + assemble.

    Hides the bias all-gather's latency under the score matmul (FastFold §5a /
    ``triattn-start-comm-feasibility.md`` Part A(b)/§E).  No collective is added
    or removed -- only the issue-point moves earlier so the wire-time overlaps
    compute.  Byte volume + peak are identical to the synchronous gather.

    Returns
    -------
    (gathered, work, tensor) : the pre-cat shard list + the async work handle +
        the (contiguous) local tensor.  At ``cp_size == 1`` returns
        ``(None, None, tensor)`` (the wait helper passes the tensor through).
    """
    if cp_size == 1:
        return None, None, tensor
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(cp_size)]
    work = dist.all_gather(gathered, tensor, group=cp_group, async_op=True)
    return gathered, work, tensor


def _all_gather_on_cp_wait(
    gathered: Optional[list],
    work: Optional[object],
    local: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """DAO epilogue: wait on the async gather handle and assemble along ``dim``.

    Pairs with :func:`_all_gather_on_cp_async`.  At ``cp_size == 1`` (``work is
    None``) returns ``local`` unchanged.
    """
    if work is None:
        return local
    work.wait()
    return torch.cat(gathered, dim=dim)


class _AllGatherTriangleAttentionStartingNode1DImpl(torch.autograd.Function):
    """All-gather attention for triangular attention starting node (1D CP).

    Replaces the cp-step ring rotation of the triangle bias with a single
    ``dist.all_gather`` of the full bias ``[B, 1, H, N, N]``.  A single
    forward attention pass over the full ``J_q = N`` is then performed with
    the gathered bias.  Backward reduce-scatters ``dbias_full`` to per-rank
    ``dbias_local``.

    Supports both REFERENCE and CUEQ backends.  Both consume the full
    ``bias_full`` ``[B, 1, H, N, N]`` directly (Q/K/V have the full J=N
    sequence dim, so S_qo = S_kv = N for the CUEQ kernel contract).

    Input/output placements (at the autograd boundary):
        - ``q_local, k_local, v_local``: ``[B, I/cp, H, J, c_hidden]`` (J full).
        - ``mask_bias_local``: ``[B, I/cp, 1, 1, J]`` additive mask bias
          (REFERENCE) or boolean mask (CUEQ).
        - ``triangle_bias_local``: ``[B, 1, H, I/cp, J]`` slab on dim -2.
        - ``o_out``: ``[B, I/cp, H, J, c_hidden]``.

    Communication (forward):
        - 1 ``dist.all_gather`` on the bias along dim -2 (full N).

    Communication (backward):
        - 1 reduce-scatter on ``dbias_full`` along dim -2.
          NCCL path: ``dist.reduce_scatter`` over chunks.
          Gloo path: ``dist.all_reduce`` + slice (Gloo lacks reduce_scatter).
        - 1 ``dist.all_gather`` to re-gather ``bias_full`` from the saved
          slab at the start of backward (avoids violating the O(N²/cp) memory
          budget that would result from saving ``bias_full``).

    Memory:
        - Forward peak transient: ``bias_full = [B, 1, H, N, N]`` (O(N²)).
        - ``ctx.save_for_backward``: ``triangle_bias_local`` slab only
          (O(N²/cp)).  NOT the gathered ``bias_full``.
        - Backward peak transient: ``bias_full`` (re-gathered) and
          ``dbias_full`` (before scatter), both O(N²).  Neither is saved.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        mask_bias_local: torch.Tensor,
        triangle_bias_local: torch.Tensor,
        cp_group: dist.ProcessGroup,
        triattn_backend: TriAttnBackend = TriAttnBackend.REFERENCE,
        apply_scale: bool = True,
        q_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward: single attention pass over the gathered bias.

        Parameters
        ----------
        q_local, k_local, v_local : Tensor
            ``[B, I/cp, H, J, c_hidden]``.  Q is pre-scaled when
            ``apply_scale`` is True.
        mask_bias_local : Tensor
            ``[B, I/cp, 1, 1, J]`` additive mask bias (REFERENCE) or boolean
            mask (CUEQ).
        triangle_bias_local : Tensor
            ``[B, 1, H, I/cp, J]`` triangle bias slab (this rank's I-chunk).
        cp_group : dist.ProcessGroup
            CP process group.
        triattn_backend : TriAttnBackend
            REFERENCE or CUEQ.
        apply_scale : bool
            If True, q is already scaled (scale=1.0 to math).
            If False, q_scale is applied here.
        q_scale : float
            Query scale factor ``1/sqrt(c_hidden)``.
        """
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        # Catch miswired q/k/v early (mixed-precision callers, device mismatch).
        assert (
            q_local.dtype == k_local.dtype == v_local.dtype
        ), f"q/k/v dtype mismatch: q={q_local.dtype}, k={k_local.dtype}, v={v_local.dtype} (rank={cp_rank})"
        assert (
            q_local.device == k_local.device == v_local.device
        ), f"q/k/v device mismatch: q={q_local.device}, k={k_local.device}, v={v_local.device} (rank={cp_rank})"

        jq_chunk = triangle_bias_local.shape[-2]  # I/cp = J/cp = N/cp
        n_global = jq_chunk * cp_size
        if q_local.shape[-2] != n_global:
            raise ValueError(
                f"q_local J dim {q_local.shape[-2]} must equal cp_size * jq_chunk = "
                f"{cp_size} * {jq_chunk} = {n_global} (rank={cp_rank})"
            )

        compute_dtype = _promote_compute_dtype(q_local.dtype)

        # DAO (Opt-0): fire the bias all-gather ASYNC, then run the
        # bias-independent compute (the q @ k^T score matmul + mask add) on the
        # compute stream so the gather's wire-time overlaps it; wait on the
        # handle only just before the bias is consumed.  No collective is added
        # or removed -- only the issue-point moves earlier (see
        # _all_gather_on_cp_async / triattn-start-comm-feasibility.md §E).
        triangle_bias_local_c = triangle_bias_local.contiguous()
        bias_shards, bias_work, bias_local_c = _all_gather_on_cp_async(
            triangle_bias_local_c, cp_group=cp_group, cp_size=cp_size
        )

        if triattn_backend == TriAttnBackend.CUEQ:
            # CUEQ kernel expects bias [B, 1, H, S_qo, S_kv].  In the 1D
            # start-node, Q/K/V have the full J=N sequence dim, so
            # S_qo = S_kv = N.  The kernel consumes the bias as its first
            # dependent op (no q@k^T to slide under inside this fn), so the
            # overlap window here is just the kernel-launch prologue; the larger
            # DAO win for CUEQ is firing in the module forward under the QKV
            # projections (future refinement).  Wait + assemble bias_full now.
            bias_full = _all_gather_on_cp_wait(bias_shards, bias_work, bias_local_c, dim=-2)
            o_out, _, _ = cueq_triangle.triangle_attention(
                q_local,
                k_local,
                v_local,
                bias_full,
                mask_bias_local,
                scale=1.0 if apply_scale else q_scale,
                return_aux=True,
            )
        else:
            # REFERENCE backend: single attention pass over the full N x N bias.
            q_c = q_local.to(compute_dtype)
            k_c = k_local.to(compute_dtype)
            v_c = v_local.to(compute_dtype)
            if not apply_scale:
                q_c = q_c * q_scale

            # Bias-independent prologue overlaps the async gather: attn = q@k^T
            # (the layer's dominant O(N^3/cp) FLOP) + the mask add.
            # attn: [B, I/cp, H, J, J]; bias_full: [B, 1, H, N, N] broadcasts on dim 1.
            attn = torch.matmul(q_c, k_c.transpose(-1, -2))
            attn = attn + mask_bias_local.to(compute_dtype)
            # DAO epilogue: wait + assemble bias_full just before it is consumed.
            bias_full = _all_gather_on_cp_wait(bias_shards, bias_work, bias_local_c, dim=-2)
            attn = attn + bias_full.to(compute_dtype)
            attn = torch.softmax(attn, dim=-1)

            o_out = torch.matmul(attn, v_c).to(q_local.dtype)

        requires_grad = (
            q_local.requires_grad or k_local.requires_grad or v_local.requires_grad or triangle_bias_local.requires_grad
        )
        if requires_grad:
            # CRITICAL: save only the local slab, NOT the gathered bias_full.
            # Saving bias_full would commit O(N^2) memory per rank,
            # violating the O(N^2/cp) backward budget.
            ctx.save_for_backward(
                q_local.detach(),
                k_local.detach(),
                v_local.detach(),
                mask_bias_local.detach(),
                triangle_bias_local_c.detach(),
            )

        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.jq_chunk = jq_chunk
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = q_local.dtype
        ctx.triattn_backend = triattn_backend
        ctx.apply_scale = apply_scale
        ctx.q_scale = q_scale
        ctx.q_requires_grad = q_local.requires_grad
        ctx.k_requires_grad = k_local.requires_grad
        ctx.v_requires_grad = v_local.requires_grad
        ctx.bias_requires_grad = triangle_bias_local.requires_grad

        return o_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        """Backward: single recompute pass + reduce-scatter on dbias.

        Re-gathers ``bias_full`` from the saved slab (avoids saving the
        full bias in ctx).  Computes dq, dk, dv from the recomputed attention
        and reduce-scatters ``dbias_full`` to per-rank ``dbias_local``.
        """
        q, k, v, mask_bias, triangle_bias_local = ctx.saved_tensors
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        cp_rank = ctx.cp_rank
        jq_chunk = ctx.jq_chunk
        compute_dtype = ctx.compute_dtype
        triattn_backend = ctx.triattn_backend
        apply_scale = ctx.apply_scale
        q_scale = ctx.q_scale

        # DAO (Opt-0): fire the bias re-gather ASYNC at backward entry so its
        # wire-time overlaps the bias-independent recompute prefix (the
        # a = q@k^T recompute on the REFERENCE path); wait just before the bias
        # is consumed.  Re-gathered transiently; not committed via ctx.
        bias_shards, bias_work, bias_local_c = _all_gather_on_cp_async(
            triangle_bias_local.contiguous(), cp_group=cp_group, cp_size=cp_size
        )

        if triattn_backend == TriAttnBackend.CUEQ:
            # CUEQ backward uses the full bias [B, 1, H, N, N] (same as forward).
            # The kernel consumes the bias immediately (recompute o/lse), so wait
            # + assemble now (no q@k^T recompute to slide under on this path).
            bias_full = _all_gather_on_cp_wait(bias_shards, bias_work, bias_local_c, dim=-2)
            cueq_scale = 1.0 if apply_scale else q_scale
            # Recompute o and lse via the forward kernel (CUEQ backward needs both).
            o_recomp, lse_recomp, _ = cueq_triangle.triangle_attention(
                q.to(q.dtype),
                k.to(q.dtype),
                v.to(q.dtype),
                bias_full,
                mask_bias,
                scale=cueq_scale,
                return_aux=True,
            )
            lse_recomp = lse_recomp.to(dtype=torch.float32)

            # SM100f kernel accepts bias in native dtype; non-SM100f requires fp32.
            can_run_sm100f = can_run_cueq_triattn_sm100f(q.device, q.dtype, k.shape[3], q.shape[-1], False)
            bias_dtype = bias_full.dtype if can_run_sm100f else torch.float32

            do_q = grad_output.to(q.dtype)
            dq_block, dk_block, dv_block, dbias_full_fp32 = torch.ops.cuequivariance.triangle_attention_bwd(
                do_q,
                o_recomp,
                q.to(q.dtype),
                k.to(q.dtype),
                v.to(q.dtype),
                bias_full.to(dtype=bias_dtype),
                mask_bias,
                lse_recomp,
                cueq_scale,
            )
            dq = dq_block.to(compute_dtype)
            dk = dk_block.to(compute_dtype)
            dv = dv_block.to(compute_dtype)
            # CUEQ returns dbias for the full [B, 1, H, N, N] bias.
            dbias_full = dbias_full_fp32.to(compute_dtype)
        else:
            # REFERENCE backend: single recompute pass over the full N x N bias.
            do = grad_output.to(compute_dtype)
            q_c = q.to(compute_dtype)
            k_c = k.to(compute_dtype)
            v_c = v.to(compute_dtype)
            if not apply_scale:
                q_c = q_c * q_scale

            # Bias-independent recompute prefix overlaps the async re-gather:
            # a = q@k^T + mask.  Recompute attention: a [B, I/cp, H, J, J]
            a = torch.matmul(q_c, k_c.transpose(-1, -2))
            a = a + mask_bias.to(compute_dtype)
            # DAO epilogue: wait + assemble bias_full just before it is consumed.
            bias_full = _all_gather_on_cp_wait(bias_shards, bias_work, bias_local_c, dim=-2)
            a = a + bias_full.to(compute_dtype)
            p = torch.softmax(a, dim=-1)

            # dv: [B, I/cp, H, J, c_hidden]
            dv = torch.matmul(p.transpose(-1, -2), do)

            # da = do @ v^T: [B, I/cp, H, J, J]
            da = torch.matmul(do, v_c.transpose(-1, -2))

            # ds = p * (da - sum_k(do * o))
            o_c = torch.matmul(p, v_c)
            d = torch.linalg.vecdot(do, o_c, dim=-1).unsqueeze(-1)
            ds = p * (da - d)

            # dq: [B, I/cp, H, J, c_hidden]
            dq = torch.matmul(ds, k_c)
            if not apply_scale:
                dq = dq * q_scale

            # dk: [B, I/cp, H, J, c_hidden]
            dk = torch.matmul(ds.transpose(-1, -2), q_c)

            # dbias_full: ds is [B, I/cp, H, J, J]; bias is [B, 1, H, N, N].
            # Sum over dim 1 (I/cp rows) for this rank's contribution to bias.
            # dbias_full holds this rank's contribution to bias gradient over
            # all N rows in dim -2 (because every rank's queries attended to
            # the full gathered bias).
            dbias_full = ds.sum(dim=1, keepdim=True, dtype=compute_dtype)

        # Reduce-scatter dbias_full along dim -2 back to per-rank slab.
        if cp_size == 1:
            dbias_local = dbias_full
        else:
            backend = dist.get_backend(cp_group)
            if backend == "gloo":
                # Gloo lacks reduce_scatter; fall back to all_reduce + slice.
                # Bind .contiguous() so we both reduce into and slice from the
                # same buffer — if dbias_full is non-contiguous, .contiguous()
                # returns a copy and reducing into the copy while slicing from
                # the original would read pre-reduce data.
                dbias_contig = dbias_full.contiguous()
                dist.all_reduce(dbias_contig, op=dist.ReduceOp.SUM, group=cp_group)
                dbias_local = dbias_contig[..., cp_rank * jq_chunk : (cp_rank + 1) * jq_chunk, :]
            else:
                dbias_chunks = list(dbias_full.chunk(cp_size, dim=-2))
                dbias_local = torch.empty_like(dbias_chunks[0])
                dist.reduce_scatter(
                    dbias_local,
                    [c.contiguous() for c in dbias_chunks],
                    op=dist.ReduceOp.SUM,
                    group=cp_group,
                )

        grad_q = dq.to(ctx.input_dtype) if ctx.q_requires_grad else None
        grad_k = dk.to(ctx.input_dtype) if ctx.k_requires_grad else None
        grad_v = dv.to(ctx.input_dtype) if ctx.v_requires_grad else None
        grad_bias = dbias_local.to(ctx.input_dtype) if ctx.bias_requires_grad else None

        # Returns: grad_q, grad_k, grad_v, None(mask), grad_bias,
        #          None(cp_group), None(triattn_backend), None(apply_scale), None(q_scale)
        return grad_q, grad_k, grad_v, None, grad_bias, None, None, None, None


def _build_bias_full_from_proj(
    bias_col_proj: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """Assemble the full ending-node bias operand from its col-slab projection.

    Input:  bias_col_proj ``[B, I=row=N, J/cp=col, H]`` -- this rank's col shard
            of the bias projection ``linear(z_col)`` (O(N^2/cp)).  The row axis
            (dim 1) is full == attention I_k; the col axis (dim 2) is the
            cp-sharded axis == attention I_q.
    Output: bias_full ``[B, 1, H, I_q=col=N, I_k=row=N]`` -- full N x N operand
            (O(N^2) transient), broadcast over the J/cp attention batch via the
            dim-1 singleton.  Consumed identically by REFERENCE and CUEQ.

    All-gathers the col (I_q) axis (dim 2) to full N, then permutes to the
    attention bias layout.  Short-circuits at cp_size == 1.  NCCL-only when
    cp_size > 1.

    NOTE: no divisibility guard on ``bias_col_proj.shape[2]`` -- ``all_gather``
    does NOT chunk the input; each rank contributes its full local col shard
    (size J/cp) and they are concatenated to the global J.  The local shard
    size need NOT be divisible by cp (e.g. J=192, cp=3 -> J/cp=64, and 64 % 3
    != 0 is fine -- 3 shards of 64 concat to 192).  The global-J % cp == 0
    invariant is enforced upstream by :class:`_PairReshardRowColImpl` (the
    reshard that produced the col-slab z this bias was projected from).
    """
    if cp_size == 1:
        gathered = bias_col_proj
    else:
        bias_c = bias_col_proj.contiguous()
        shards = [torch.empty_like(bias_c) for _ in range(cp_size)]
        dist.all_gather(shards, bias_c, group=cp_group)
        gathered = torch.cat(shards, dim=2)  # [B, row=N, col=N, H]
    # [B, row=N, col=N, H] -> [B, H, col=I_q=N, row=I_k=N] -> [B, 1, H, I_q, I_k]
    return permute_final_dims(gathered, (2, 1, 0)).unsqueeze(-4)


def _reduce_scatter_sum_dim(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    dim: int,
) -> torch.Tensor:
    """Reduce-scatter (sum) a tensor along an arbitrary ``dim``.

    Each rank holds a partial (full-size) tensor; the result sums across ranks
    and scatters so each rank keeps a ``1/cp`` shard along ``dim``.  Used by the
    z-first ending-node backward to reduce-scatter ``dbias`` on the col(=I_q)
    axis back to the bias projection's col/cp shard.  Even-sharding guard on
    ``dim``; short-circuits at cp_size == 1.  NCCL-only when cp_size > 1.
    """
    if cp_size == 1:
        return tensor
    if tensor.shape[dim] % cp_size != 0:
        raise ValueError(
            f"_reduce_scatter_sum_dim: dim {dim} size {tensor.shape[dim]} not divisible by cp_size {cp_size}"
        )
    input_list = [c.contiguous() for c in tensor.chunk(cp_size, dim=dim)]
    output = torch.empty_like(input_list[0])
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=cp_group)
    return output


def _all_to_all_pair_reshard(
    tensor: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
    full_dim: int,
    shard_dim: int,
) -> torch.Tensor:
    """Re-shard a 4-D pair tensor by swapping which sequence axis is full vs sharded.

    Splits ``tensor`` along ``full_dim`` into ``cp_size`` chunks, exchanges them
    among CP ranks via ``dist.all_to_all``, and concatenates received chunks
    along ``shard_dim``.  After the call, ``full_dim`` becomes sharded
    (size ``full_dim_size / cp_size``) and ``shard_dim`` becomes full
    (size ``shard_dim_size * cp_size``).

    This 4-D pair-tensor (``[B, I, J, C]``) re-shard moves the *pre-projection*
    pair representation (channel width ``c_z``) once, rather than the three
    projected ``H*c_hidden``-wide Q/K/V tensors -- the z-first comm-volume win
    (see ``.claude/scratch/dap-isf/triattn-end-project-order-efficiency.md``).

    NCCL-only.  Short-circuits at ``cp_size == 1``.

    Parameters
    ----------
    tensor : Tensor
        Input pair tensor (any dtype, rank >= ``max(full_dim, shard_dim) + 1``).
    cp_group : dist.ProcessGroup
        CP process group (NCCL).
    cp_size : int
        Number of ranks in ``cp_group``.
    full_dim : int
        Dimension currently full (size ``N``), becomes sharded (``N/cp``).
    shard_dim : int
        Dimension currently sharded (``M/cp``), becomes full (``M``).

    Raises
    ------
    ValueError
        If ``tensor.shape[full_dim]`` is not divisible by ``cp_size`` (uneven
        sharding -- CLAUDE.md "No uneven sharding").
    """
    if cp_size == 1:
        return tensor
    n_full = tensor.shape[full_dim]
    if n_full % cp_size != 0:
        raise ValueError(
            f"_all_to_all_pair_reshard: full_dim={full_dim} size {n_full} not divisible by cp_size {cp_size}"
        )
    send_list = [c.contiguous() for c in tensor.chunk(cp_size, dim=full_dim)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(cp_size)]
    dist.all_to_all(recv_list, send_list, group=cp_group)
    return torch.cat(recv_list, dim=shard_dim)


class _PairReshardRowColImpl(torch.autograd.Function):
    """Re-shard a pair-representation DTensor between row-slab and col-slab (1D CP).

    The ending node's contraction axis ``I`` is the cp-sharded axis under the
    fixed row-slab layout ``(Shard(0), Shard(1))`` (batch on dp dim 0, ``I`` on
    cp dim 1).  To attend along ``I`` locally, the layer needs ``I`` full and
    ``J`` sharded -- the col-slab layout ``(Shard(0), Shard(2))``.  This
    autograd.Function performs that re-shard on the *pre-projection* pair tensor
    ``z [B, I, J, C]`` via a single all-to-all (forward) and its dual (backward).

    By re-sharding ``z`` ONCE (channel width ``c_z``) instead of the three
    projected Q/K/V tensors (each ``H*c_hidden`` wide), the entry comm volume
    drops by ``3*H*c_hidden / c_z`` (~3x at the default Boltz-2 config), and the
    projections run on the proven :class:`_LinearParamsReplicatedImpl` DTensor
    path with their fp32 ``Partial("avg")`` parameter-gradient reduction intact
    (no hand-rolled grad path).  The bias all-gather is **NOT** eliminated: the
    bias is projected col-slab from ``z_col`` but its ``I_q``=column axis is the
    cp-sharded one, so it keeps a col-axis all-gather inside the attention fn
    (see :class:`_DAPTriangleAttentionEndingNode1DImpl` /
    :func:`_build_bias_full_from_proj`).  See
    ``.claude/scratch/dap-isf/trifuse-proposal.md`` for the design rationale and
    the backward dz-assembly derivation.

    This function carries NO parameter gradients -- it is a pure activation
    re-shard.  All projection gradients flow through the unchanged DTensor
    linears.

    Input/output placements (forward):
        in  : ``[B, I=N, J=N, C]`` row-slab ``(Shard(0), Shard(1))``  (I on cp)
        out : ``[B, I=N, J=N, C]`` col-slab ``(Shard(0), Shard(2))``  (J on cp)

    Collectives:
        forward : 1 all-to-all (full=1 I, shard=2 J)
        backward: 1 all-to-all (the dual: full=2 J, shard=1 I)

    The forward and backward a2a's fire UNCONDITIONALLY (no data-dependent
    branch) so all ranks execute collectives in identical order (CLAUDE.md
    "All ranks must execute collectives in identical order").  Per-rank
    transient is O(N^2/cp) in both directions (a genuine re-shard, never
    O(N^2)).  Transport: NCCL only.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        z_local: torch.Tensor,
        cp_group: dist.ProcessGroup,
        cp_size: int,
        in_full_dim: int,
        in_shard_dim: int,
        out_shape: tuple,
    ) -> torch.Tensor:
        """Re-shard local row-slab ``z`` -> col-slab via one all-to-all.

        Parameters
        ----------
        z_local : Tensor
            ``[B, I=N, J/cp, C]`` row-slab local shard (the ``to_local()`` of a
            ``(Shard(0), Shard(1))`` DTensor: ``I`` is full at dim 1, the cp
            shard sits on dim 1 globally but locally this rank holds ``I/cp``).
            NOTE the caller passes the local shard; the global axis bookkeeping
            is in ``in_full_dim`` / ``in_shard_dim`` / ``out_shape``.
        cp_group, cp_size : process group + world size.
        in_full_dim : int
            Local dim that is full (``J``) and becomes sharded after the a2a.
        in_shard_dim : int
            Local dim that is sharded (``I/cp``) and becomes full after the a2a.
        out_shape : tuple
            GLOBAL output shape (unused for the local compute; carried for the
            caller to build the output DTensor with explicit metadata).
        """
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.in_full_dim = in_full_dim
        ctx.in_shard_dim = in_shard_dim
        # Even-sharding guard on the axis being made sharded (full_dim).
        if cp_size > 1 and z_local.shape[in_full_dim] % cp_size != 0:
            raise ValueError(
                f"_PairReshardRowColImpl.forward: full_dim={in_full_dim} size "
                f"{z_local.shape[in_full_dim]} not divisible by cp_size {cp_size}"
            )
        z_col = _all_to_all_pair_reshard(
            z_local.contiguous(), cp_group, cp_size, full_dim=in_full_dim, shard_dim=in_shard_dim
        )
        return z_col

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, grad_z_col: torch.Tensor):
        """Backward: the dual a2a (col-slab grad -> row-slab grad).

        The forward made ``in_full_dim`` (J) sharded and ``in_shard_dim`` (I)
        full; the dual swaps them back.  Fires unconditionally (identical
        collective order across ranks).  Per-rank transient O(N^2/cp).
        """
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        # Dual: the axis the forward made full (in_shard_dim, I) is now full and
        # becomes sharded again; the axis the forward made sharded (in_full_dim,
        # J) is now sharded and becomes full again.
        if cp_size > 1 and grad_z_col.shape[ctx.in_shard_dim] % cp_size != 0:
            raise ValueError(
                f"_PairReshardRowColImpl.backward: full_dim={ctx.in_shard_dim} size "
                f"{grad_z_col.shape[ctx.in_shard_dim]} not divisible by cp_size {cp_size}"
            )
        grad_z_row = _all_to_all_pair_reshard(
            grad_z_col.contiguous(), cp_group, cp_size, full_dim=ctx.in_shard_dim, shard_dim=ctx.in_full_dim
        )
        return grad_z_row, None, None, None, None, None


class _DAPTriangleAttentionEndingNode1DImpl(torch.autograd.Function):
    """z-first local attention core for the triangular attention ending node (1D CP).

    Inputs arrive ALREADY col-slab (the module re-sharded the pre-projection ``z``
    once via :class:`_PairReshardRowColImpl` and projected Q/K/V/gate/bias locally
    on the col-slab).  This function runs the local full ``N x N`` softmax
    attention on the col-slab and returns the col-slab output; the module applies
    ``linear_o`` and re-shards back to row-slab.  The triangle bias is **NOT**
    born col-correct -- it keeps a col-axis all-gather (see below).

    Backends:
        - **REFERENCE**: local softmax-attention fwd/bwd on the col-slab tensors.
        - **CUEQ**: calls ``cueq_triangle.triangle_attention`` on the same
          col-slab inputs; the full bias operand assembled by
          :func:`_build_bias_full_from_proj` satisfies the kernel's
          ``[B, 1, H, S_qo=N, S_kv=N]`` contract.  Backward calls
          ``torch.ops.cuequivariance.triangle_attention_bwd`` and recomputes
          ``o``/``lse`` via the forward kernel.

    **z-first fusion (the per-layer collective budget moves but is NOT zero).**
    Under the z-first design the q/k/v ENTRY re-shard is fused into a single
    pre-projection ``z`` all-to-all (via :class:`_PairReshardRowColImpl` in the
    module forward), and Q/K/V/gate are then projected *locally on the col-slab*
    ``z_col [B, I=N, J/cp, C]``.  That fuses the 3 shipped q/k/v entry a2a's into
    1 c_z-wide z-a2a (the ~3x q/k/v entry-volume win).

    **The bias all-gather is NOT eliminated.**  The ending-node triangle bias is
    ``b[i_q, i_k]`` where the attention QUERY axis ``i_q`` is the original COLUMN
    axis ``J`` (post-transpose), and ``i_k`` is the original ROW axis ``I`` (see
    project-memory ``triattn-endnode-bias-zfirst`` -- proven by perturbation
    trace).  z-first SHARDS the column axis (``z_col`` has ``J/cp``), which is
    exactly the axis the attention needs FULL at the ``i_q`` position.  So the
    bias projection ``b_col = linear(z_col) = [B, I=N, J/cp, H]`` has its
    ``i_q``(=J) axis sharded and MUST be all-gathered on ``J`` to assemble the
    full ``[B, 1, H, I_q=N, I_k=N]`` the local softmax (and the CUEQ kernel
    contract) require.  This mirrors the shipped DAP path's bias gather, just on
    the ``J`` axis instead of the ``I_k`` axis.

    By the time this function runs the inputs are col-slab:
        q/k/v_col  ``[B, J/cp, H, I=N, D]``   (I full -- contraction complete)
        mask_col   ``[B, J/cp, 1, 1, I=N]``
        b_col      ``[B, I=N, J/cp, H]``      (col-slab bias projection, O(N^2/cp);
                                               this fn all-gathers its J axis to
                                               form bias_full transiently)

    Forward collectives: **1** (bias all-gather on the J axis; the entry z-a2a
        and exit o-a2a live in the module's :class:`_PairReshardRowColImpl`
        calls, not here).  End-node per-layer fwd total = 3 (z-a2a + bias-AG +
        o-a2a), vs the shipped 6.
    Backward collectives: **2** (bias re-gather on J to re-materialize bias_full
        transiently + dbias reduce-scatter on J back to the J/cp shard, fp32;
        the dz/grad-out re-shards live in the module's reshard-fn backward).

    Saved tensors (all O(N^2/cp) per-rank per CLAUDE.md pair-repr budget):
        q_col, k_col, v_col   ``[B, J/cp, H, I=N, D]``
        mask_col              ``[B, J/cp, 1, 1, I=N]``
        b_col                 ``[B, I=N, J/cp, H]``  (col-slab bias PROJECTION,
                                O(N^2/cp); NOT the gathered O(N^2) bias_full --
                                backward re-gathers transiently, mirroring the
                                shipped row-slab-save + re-gather pattern, just on
                                the J axis).
        o_col                 ``[B, J/cp, H, I=N, D]`` on REFERENCE; ``None`` on
                                CUEQ (CUEQ backward recomputes o via the kernel).

    Transport: NCCL only (the bias all-gather / reduce-scatter fire when
    cp_size > 1; short-circuit at cp_size == 1).
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        q_col: torch.Tensor,
        k_col: torch.Tensor,
        v_col: torch.Tensor,
        mask_col: torch.Tensor,
        bias_col_proj: torch.Tensor,
        cp_group: dist.ProcessGroup,
        no_heads: int,
        c_hidden: int,
        triattn_backend: TriAttnBackend = TriAttnBackend.REFERENCE,
        apply_scale: bool = True,
        q_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward: local N×N softmax attention on col-slab inputs (z-first).

        Inputs ALREADY arrive col-slab and col-correct (the module re-sharded the
        pre-projection ``z`` once and projected Q/K/V/gate/bias LOCALLY on the
        col-slab ``z_col``).  This function: all-gathers the bias projection on
        its col (=I_q) axis to build the full ``[B,1,H,I_q=N,I_k=N]`` operand,
        runs the local attention, and returns the col-slab output.  The module
        applies ``linear_o`` and re-shards the output back to row-slab.

        Parameters
        ----------
        q_col, k_col, v_col : Tensor
            ``[B, J/cp, H, I=N, D]`` col-slab query/key/value (Q pre-scaled when
            ``apply_scale``).  I (the attention seq axis) is full; J/cp is the
            attention batch.
        mask_col : Tensor
            ``[B, J/cp, 1, 1, I_k=N]`` additive mask bias (REFERENCE) or boolean
            mask (CUEQ).
        bias_col_proj : Tensor
            ``[B, I=row=N, J/cp=col, H]`` col-slab triangle-bias PROJECTION
            (= ``linear(z_col)``, O(N^2/cp); the col(=I_q) axis is sharded).
            This fn all-gathers the col axis to assemble the full bias operand
            ``[B, 1, H, I_q=col=N, I_k=row=N]`` (O(N^2) transient, not saved).
        cp_group : dist.ProcessGroup
            CP process group (NCCL).
        no_heads, c_hidden : int
            Attention shape parameters (saved on ctx).
        triattn_backend : TriAttnBackend
            REFERENCE or CUEQ; both consume the IDENTICAL full bias operand.
        apply_scale : bool
            If True, q is already pre-scaled (kernel uses scale=1.0).
        q_scale : float
            Query scale factor ``1/sqrt(c_hidden)``.  Used when
            ``apply_scale`` is False.
        """
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        # NCCL is only required when the bias all-gather actually fires
        # (cp_size > 1).  At cp_size == 1 _build_bias_full_from_proj short-circuits,
        # so the degenerate single-rank path runs on any backend.
        if cp_size > 1 and dist.get_backend(cp_group) != "nccl":
            raise NotImplementedError(
                "End-node z-first attention requires NCCL backend with cp_size>1 "
                f"(got {dist.get_backend(cp_group)} with cp_size={cp_size})."
            )

        # Catch miswired q/k/v early (mixed-precision callers, device mismatch).
        assert (
            q_col.dtype == k_col.dtype == v_col.dtype
        ), f"q/k/v dtype mismatch: q={q_col.dtype}, k={k_col.dtype}, v={v_col.dtype} (rank={cp_rank})"
        assert (
            q_col.device == k_col.device == v_col.device
        ), f"q/k/v device mismatch: q={q_col.device}, k={k_col.device}, v={v_col.device} (rank={cp_rank})"

        # Validate col-slab shapes.  q/k/v: [B, J/cp, H, I=N, D] (I=dim 3, full).
        if k_col.shape != q_col.shape:
            raise ValueError(f"k_col.shape {k_col.shape} != q_col.shape {q_col.shape}")
        if v_col.shape != q_col.shape:
            raise ValueError(f"v_col.shape {v_col.shape} != q_col.shape {q_col.shape}")
        # mask_col [B, J/cp, 1, 1, I_k=N]: J/cp matches q dim 1, I_k matches q dim 3.
        if mask_col.shape[1] != q_col.shape[1]:
            raise ValueError(f"mask_col.shape[1]={mask_col.shape[1]} != q_col.shape[1] (J/cp) {q_col.shape[1]}")
        if mask_col.shape[-1] != q_col.shape[3]:
            raise ValueError(f"mask_col.shape[-1]={mask_col.shape[-1]} != q_col.shape[3] (I_k=N) {q_col.shape[3]}")
        # bias_col_proj [B, I=row=N, J/cp=col, H]: row (dim 1) full == I_k == q dim 3;
        # col (dim 2) sharded, gathered to N == I_q == q dim 3.
        if bias_col_proj.shape[1] != q_col.shape[3]:
            raise ValueError(
                f"bias_col_proj.shape[1]={bias_col_proj.shape[1]} (I=row=N) != q_col.shape[3] (I=N) {q_col.shape[3]}"
            )
        if bias_col_proj.shape[2] != q_col.shape[1]:
            raise ValueError(
                f"bias_col_proj.shape[2]={bias_col_proj.shape[2]} (J/cp=col) != q_col.shape[1] (J/cp) {q_col.shape[1]}"
            )

        requires_grad = q_col.requires_grad or k_col.requires_grad or v_col.requires_grad or bias_col_proj.requires_grad
        compute_dtype = _promote_compute_dtype(q_col.dtype)
        input_dtype = q_col.dtype

        # Build the full bias operand by all-gathering the col(=I_q) axis of the
        # bias PROJECTION.  bias_col_proj [B, I=row=N, col/cp, H] -> AG col ->
        # [B, row=N, col=N, H] -> permute -> [B, 1, H, I_q=col=N, I_k=row=N].
        # The bias is b[i_q=col, i_k=row], broadcast over the J/cp attention batch
        # via the dim-1 singleton (see test_ending_node_bias_index_semantics_*).
        # O(N^2) transient, NOT saved (ctx keeps the O(N^2/cp) projection; backward
        # re-gathers).  Both REFERENCE and CUEQ consume this identical operand.
        bias_full = _build_bias_full_from_proj(bias_col_proj, cp_group, cp_size)

        if triattn_backend == TriAttnBackend.CUEQ:
            # CUEQ kernel contract: bias [B, 1, H, S_qo=N, S_kv=N] -- satisfied by
            # bias_full (I_q=col=N at dim 3, I_k=row=N at dim 4).
            cueq_scale = 1.0 if apply_scale else q_scale
            o_col, _, _ = cueq_triangle.triangle_attention(
                q_col,
                k_col,
                v_col,
                bias_full,
                mask_col,
                scale=cueq_scale,
                return_aux=True,
            )
        else:
            # REFERENCE: local full N x N softmax attention (no online accumulation).
            q_c = q_col.to(compute_dtype)
            k_c = k_col.to(compute_dtype)
            v_c = v_col.to(compute_dtype)

            # attn: [B, J/cp, H, I_q=N, I_k=N]
            # mask_col broadcasts: [B, J/cp, 1, 1, I_k=N] -> over (H, I_q).
            # bias_full broadcasts: [B, 1, H, I_q=N, I_k=N] -> over J/cp via dim 1.
            attn = torch.matmul(q_c, k_c.transpose(-1, -2))
            attn = attn + mask_col.to(compute_dtype)
            attn = attn + bias_full.to(compute_dtype)
            p = torch.softmax(attn, dim=-1)
            o_col_compute = torch.matmul(p, v_c)
            o_col = o_col_compute.to(input_dtype)

        # NO output a2a: the output stays col-slab [B, J/cp, H, I=N, D].  The
        # module applies linear_o on the col-slab then re-shards z back to
        # row-slab via _PairReshardRowColImpl (the single exit reshard).

        if requires_grad:
            # Save the O(N^2/cp) col-slab bias PROJECTION bias_col_proj
            # [B, I=row=N, J/cp=col, H] -- NOT the gathered O(N^2) bias_full.
            # Backward re-gathers the col axis transiently (mirrors the shipped
            # save-projection + re-gather pattern, on the col axis under z-first).
            # o_col is only consumed by the REFERENCE backward (d = vecdot(do, o));
            # CUEQ backward recomputes o via the kernel, so gate the save.
            save_o_col = o_col.contiguous().detach() if triattn_backend == TriAttnBackend.REFERENCE else None
            ctx.save_for_backward(
                q_col.contiguous().detach(),
                k_col.contiguous().detach(),
                v_col.contiguous().detach(),
                mask_col.contiguous().detach(),
                bias_col_proj.contiguous().detach(),
                save_o_col,
            )

        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = input_dtype
        ctx.no_heads = no_heads
        ctx.c_hidden = c_hidden
        ctx.q_requires_grad = q_col.requires_grad
        ctx.k_requires_grad = k_col.requires_grad
        ctx.v_requires_grad = v_col.requires_grad
        ctx.bias_requires_grad = bias_col_proj.requires_grad
        ctx.triattn_backend = triattn_backend
        ctx.apply_scale = apply_scale
        ctx.q_scale = q_scale

        return o_col

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, grad_o_col: torch.Tensor):
        """Backward: re-gather bias (col), local softmax bwd, reduce-scatter dbias (col).

        z-first: the grad arrives ALREADY col-slab (``grad_o_col`` -- the module's
        linear_o backward + reshard-fn produced it); dq/dk/dv stay col-slab
        (returned to the module, whose projection-linears' backward + reshard-fn
        carry the dz re-shard).  So this function fires only the bias collectives.
        Collective ordering is identical across all ranks (no rank-divergent
        branch).  The 2 collectives:
          1. all-gather the saved O(N^2/cp) bias PROJECTION on its col(=I_q) axis
             to re-materialize the O(N^2) bias_full transiently (dual of fwd).
          2. reduce-scatter dbias on the col axis back to the projection's
             col/cp shard (run in compute_dtype >= fp32, cast to input_dtype
             after).  dbias is first LOCALLY summed over the J/cp attention batch
             (dim 1) -- the bias was broadcast over that batch in fwd -- THEN
             reduce-scattered across ranks (two distinct reductions).
        Inputs that do not require grad have their returned tensor replaced with
        ``None``.
        """
        q_col, k_col, v_col, mask_col, bias_col_proj, o_col = ctx.saved_tensors
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        compute_dtype = ctx.compute_dtype
        input_dtype = ctx.input_dtype
        triattn_backend = ctx.triattn_backend
        apply_scale = ctx.apply_scale
        q_scale = ctx.q_scale

        # Re-gather the bias transiently: bias_col_proj [B,row=N,col/cp,H] -> AG
        # col -> bias_full [B,1,H,I_q=col=N,I_k=row=N] (O(N^2), NOT in ctx; ctx
        # holds only the O(N^2/cp) projection, preserving the pair-repr budget).
        bias_full = _build_bias_full_from_proj(bias_col_proj, cp_group, cp_size)

        # grad arrives col-slab already (no a2a here -- the module's reshard-fn
        # handled the row<->col on the output z).  No NCCL-buffer dtype contract
        # to honour at this boundary (the bias a2a below carries its own cast),
        # so consume grad_o_col directly; each branch recasts to its own compute
        # dtype below.
        if triattn_backend == TriAttnBackend.CUEQ:
            # CUEQ backward: recompute o/lse via the forward kernel (mirrors
            # _AllGatherTriangleAttentionStartingNode1DImpl) then call
            # triangle_attention_bwd.  Inputs/outputs stay in q.dtype (kernel
            # contract); we cast grads to compute_dtype (>=fp32) afterward
            # so the cross-rank reduce-scatter on dbias runs in fp32 per
            # CLAUDE.md "Reduce gradients in fp32".
            cueq_scale = 1.0 if apply_scale else q_scale
            q_kernel = q_col.to(q_col.dtype)
            k_kernel = k_col.to(q_col.dtype)
            v_kernel = v_col.to(q_col.dtype)
            o_recomp, lse_recomp, _ = cueq_triangle.triangle_attention(
                q_kernel,
                k_kernel,
                v_kernel,
                bias_full,
                mask_col,
                scale=cueq_scale,
                return_aux=True,
            )
            lse_recomp = lse_recomp.to(dtype=torch.float32)

            # SM100f kernel accepts bias in native dtype; non-SM100f requires fp32.
            can_run_sm100f = can_run_cueq_triattn_sm100f(
                q_col.device, q_col.dtype, k_col.shape[3], q_col.shape[-1], False
            )
            bias_dtype = bias_full.dtype if can_run_sm100f else torch.float32

            do_kernel = grad_o_col.to(q_col.dtype)
            dq_block, dk_block, dv_block, dbias_block_fp32 = torch.ops.cuequivariance.triangle_attention_bwd(
                do_kernel,
                o_recomp,
                q_kernel,
                k_kernel,
                v_kernel,
                bias_full.to(dtype=bias_dtype),
                mask_col,
                lse_recomp,
                cueq_scale,
            )
            dq_col = dq_block.to(compute_dtype)
            dk_col = dk_block.to(compute_dtype)
            dv_col = dv_block.to(compute_dtype)
            # CUEQ returns dbias matching the full bias operand
            # [B, 1, H, I_q=col=N, I_k=row=N] (already summed over the kernel's
            # J/cp batch).  Same layout as the REFERENCE branch's dbias_full.
            dbias_full = dbias_block_fp32.to(compute_dtype)
        else:
            # REFERENCE softmax-attention backward on col-slab (local).
            do = grad_o_col.to(compute_dtype)
            q_c = q_col.to(compute_dtype)
            k_c = k_col.to(compute_dtype)
            v_c = v_col.to(compute_dtype)

            # Recompute softmax weights p locally (cheaper than saving p; same O(N^2/cp) budget).
            attn = torch.matmul(q_c, k_c.transpose(-1, -2))
            attn = attn + mask_col.to(compute_dtype)
            attn = attn + bias_full.to(compute_dtype)
            p = torch.softmax(attn, dim=-1)

            # dv = p^T @ do  ->  [B, J/cp, H, I_k=N, D]
            dv_col = torch.matmul(p.transpose(-1, -2), do)

            # da = do @ v^T  ->  [B, J/cp, H, I_q=N, I_k=N]
            da = torch.matmul(do, v_c.transpose(-1, -2))

            # ds = p * (da - sum_k(do * o))
            o_c = o_col.to(compute_dtype)
            d = torch.linalg.vecdot(do, o_c, dim=-1).unsqueeze(-1)
            ds = p * (da - d)

            # dq = ds @ k  ->  [B, J/cp, H, I_q=N, D]
            dq_col = torch.matmul(ds, k_c)

            # dk = ds^T @ q  ->  [B, J/cp, H, I_k=N, D]
            dk_col = torch.matmul(ds.transpose(-1, -2), q_c)

            # dbias_full [B, 1, H, I_q=N, I_k=N] = local SUM of ds over the J/cp
            # attention batch (dim 1) -- the bias_full was broadcast over that
            # batch via its dim-1 singleton in fwd (C1 step (a): local sum
            # BEFORE the cross-rank reduce-scatter).
            dbias_full = ds.sum(dim=1, keepdim=True, dtype=compute_dtype)

        # dq/dk/dv stay COL-SLAB: returned to the module, whose projection-linears'
        # backward + reshard-fn carry the dz re-shard.  No a2a here.
        grad_q = dq_col.to(input_dtype) if ctx.q_requires_grad else None
        grad_k = dk_col.to(input_dtype) if ctx.k_requires_grad else None
        grad_v = dv_col.to(input_dtype) if ctx.v_requires_grad else None

        # dbias: reduce-scatter on the COL(=I_q) axis back to the projection's
        # col/cp shard (C1 step (b) + C2).  dbias_full is [B,1,H,I_q=col=N,
        # I_k=row=N]; permute back to the projection layout [B,row=N,col=N,H]
        # (inverse of the fwd _build_bias_full_from_proj permute), then RS the
        # col axis -> [B,row=N,col/cp,H] matching bias_col_proj.  fp32, cast
        # after (CLAUDE.md "Reduce gradients in fp32").
        if ctx.bias_requires_grad:
            # [B,1,H,I_q=col,I_k=row] -> squeeze dim1 -> [B,H,col,row]
            #   -> permute_final_dims((2,1,0)) -> [B,row,col,H]  (inverse of fwd)
            dbias_proj_full = permute_final_dims(dbias_full.squeeze(-4), (2, 1, 0))  # [B,row=N,col=N,H]
            dbias_proj = _reduce_scatter_sum_dim(dbias_proj_full, cp_group, cp_size, dim=2)  # [B,row=N,col/cp,H]
            grad_bias = dbias_proj.to(input_dtype)
        else:
            grad_bias = None

        # Returns: grad_q, grad_k, grad_v, None(mask), grad_bias,
        #          None(cp_group), None(no_heads), None(c_hidden),
        #          None(triattn_backend), None(apply_scale), None(q_scale)
        return grad_q, grad_k, grad_v, None, grad_bias, None, None, None, None, None, None


class TriangleAttentionStartingNode1D(nn.Module):
    """Distributed triangle attention starting node (Algorithm 13) for 1D CP.

    Under 1D CP with ``(dp, cp)`` mesh, pair z is ``[B, I/cp, J, C_in]``.
    The starting node attends along J (the full, unsharded dimension).
    Q/K/V are local (J is full).  The triangle bias is ``[B, 1, H, I/cp, J]``
    locally; its I/cp dimension maps to the attention's J_q dimension (= N).
    We all-gather the bias slab to assemble ``[B, 1, H, N, N]`` once and run
    a single attention pass; REFERENCE consumes the full bias directly while
    CUEQ slices the I axis to ``[B, 1, H, I/cp, J]`` to match its kernel
    contract.

    Communication budget:
        - Forward: 1 all-gather of the bias slab along dim -2.
        - Backward: 1 all-gather to re-assemble the bias + 1 reduce-scatter
          of dbias along dim -2.

    Memory: ``bias_full = [B, 1, H, N, N]`` is held transiently in forward
    and backward (O(N^2)); ``ctx.save_for_backward`` keeps only the local
    slab (O(N^2/cp)).

    Parameters
    ----------
    layer : SerialTriangleAttentionStartingNode
        The serial starting node module.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        The CP process group.
    """

    def __init__(
        self,
        layer: SerialTriangleAttentionStartingNode,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        if not layer.starting:
            raise ValueError("Serial layer must have starting=True for TriangleAttentionStartingNode1D")

        self.device_mesh = device_mesh
        self.c_in = layer.c_in
        self.c_hidden = layer.c_hidden
        self.no_heads = layer.no_heads
        self.inf = layer.inf
        self.cp_group = cp_group

        self.layer_norm = LayerNormParamsReplicatedNoAutoCastBF16(layer.layer_norm, device_mesh)
        self.linear = LinearParamsReplicatedNoAutoCastBF16(layer.linear, device_mesh)

        mha = layer.mha
        self.mha = nn.Module()
        self.mha.linear_q = LinearParamsReplicatedNoAutoCastBF16(mha.linear_q, device_mesh)
        self.mha.linear_k = LinearParamsReplicatedNoAutoCastBF16(mha.linear_k, device_mesh)
        self.mha.linear_v = LinearParamsReplicatedNoAutoCastBF16(mha.linear_v, device_mesh)
        self.mha.linear_o = LinearParamsReplicatedNoAutoCastBF16(mha.linear_o, device_mesh)
        self.mha.linear_g = (
            LinearParamsReplicatedNoAutoCastBF16(mha.linear_g, device_mesh) if mha.linear_g is not None else None
        )
        self.mha.no_heads = mha.no_heads
        self.mha.c_hidden = mha.c_hidden
        self.mha.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: DTensor,
        mask: Optional[DTensor] = None,
        triattn_backend: TriAttnBackend = TriAttnBackend.REFERENCE,
    ) -> DTensor:
        """Forward pass with ring communication for triangle bias.

        Parameters
        ----------
        x : DTensor
            ``[B, I, J, C_in]`` pair representation with placements
            ``(Shard(0), Shard(1))``.
        mask : DTensor, optional
            ``[B, I, J]`` pair mask with placements ``(Shard(0), Shard(1))``.
        triattn_backend : TriAttnBackend
            Backend to use for local attention computation.

        Returns
        -------
        DTensor
            ``[B, I, J, C_in]`` attention output, same placements as x.
        """
        if triattn_backend in (TriAttnBackend.TRIFAST, TriAttnBackend.CUEQ_FWD_TRIFAST_BWD):
            raise NotImplementedError(
                f"TriAttnBackend.{triattn_backend.name} is not supported for 1D CP triangle attention. "
                "Use REFERENCE or CUEQ."
            )
        if triattn_backend == TriAttnBackend.CUEQ and not cueq_is_installed:
            raise ValueError(
                "cuequivariance_torch is not installed. For Triangle Attention support, "
                "install using: pip install cuequivariance_ops_torch_cu13==<version> cuequivariance_torch==<version> "
                "where the 'version' tag can be found in the pyproject.toml file"
            )

        x_placements = x.placements

        # LayerNorm + triangle bias projection (all via DTensor)
        x_normed = self.layer_norm(x)
        triangle_bias_dt = self.linear(x_normed)  # DTensor [B, I, J, H]

        # Go local for the triangle bias -- [B_local, I/cp, J, H]
        triangle_bias_local = triangle_bias_dt.to_local()
        # [B_local, I/cp, J, H] -> [B_local, H, I/cp, J] -> [B_local, 1, H, I/cp, J]
        triangle_bias_local = permute_final_dims(triangle_bias_local, (2, 0, 1)).unsqueeze(-4)

        # Go local for attention
        x_local = x_normed.to_local()  # [B_local, I/cp, J, C_in]

        # Mask: convert to backend-specific format
        if mask is not None:
            mask_local = mask.to_local()
        else:
            mask_local = x_local.new_ones(x_local.shape[:-1])

        if triattn_backend == TriAttnBackend.CUEQ:
            # CUEQ expects boolean mask: [B, I/cp, 1, 1, J]
            mask_bias_local = mask_local[..., :, None, None, :].to(
                dtype=torch.bool, memory_format=torch.contiguous_format
            )
        else:
            # REFERENCE expects additive mask bias: [B, I/cp, 1, 1, J]
            mask_bias_local = (self.inf * (mask_local - 1))[..., :, None, None, :]

        # Q, K, V projections (via DTensor for proper gradient flow)
        q_dt = self.mha.linear_q(x_normed)
        k_dt = self.mha.linear_k(x_normed)
        v_dt = self.mha.linear_v(x_normed)

        # Use Q DTensor for global shape info (has H*c_hidden as last dim)
        q_shape = q_dt.shape  # [B, I, J, H*c_hidden]

        q_local = q_dt.to_local()
        k_local = k_dt.to_local()
        v_local = v_dt.to_local()

        # [B, I/cp, J, H*c_hidden] -> [B, I/cp, H, J, c_hidden]
        q_local = q_local.view(q_local.shape[:-1] + (self.no_heads, -1)).transpose(-2, -3)
        k_local = k_local.view(k_local.shape[:-1] + (self.no_heads, -1)).transpose(-2, -3)
        v_local = v_local.view(v_local.shape[:-1] + (self.no_heads, -1)).transpose(-2, -3)

        q_scale = self.c_hidden**-0.5
        if triattn_backend == TriAttnBackend.REFERENCE:
            # REFERENCE: pre-scale q, kernel uses scale=1.0
            q_local = q_local * q_scale
            apply_scale = True
        else:
            # CUEQ: let the kernel handle scaling internally
            apply_scale = False

        # All-gather the bias once and run a single attention pass.  Both
        # REFERENCE and CUEQ consume the full [B, 1, H, N, N] bias.
        o_local = _AllGatherTriangleAttentionStartingNode1DImpl.apply(
            q_local,
            k_local,
            v_local,
            mask_bias_local,
            triangle_bias_local,
            self.cp_group,
            triattn_backend,
            apply_scale,
            q_scale,
        )

        # [B, I/cp, H, J, c_hidden] -> [B, I/cp, J, H, c_hidden]
        o_local = o_local.transpose(-2, -3)

        # Gating
        if self.mha.linear_g is not None:
            g = self.mha.linear_g(x_normed)
            g_local = self.mha.sigmoid(g.to_local())
            g_local = g_local.view(g_local.shape[:-1] + (self.no_heads, -1))
            o_local = o_local * g_local

        # Flatten heads: [B, I/cp, J, H*c_hidden]
        o_local = o_local.reshape(o_local.shape[:-2] + (-1,))

        # Output projection (via DTensor) -- shape is [B, I, J, H*c_hidden]
        o_dt = DTensor.from_local(
            o_local,
            self.device_mesh,
            x_placements,
            shape=q_shape,
            stride=update_exhaustive_strides(o_local.shape, o_local.stride(), q_shape),
        )
        o_dt = self.mha.linear_o(o_dt)

        return o_dt


class TriangleAttentionEndingNode1D(nn.Module):
    """Distributed triangle attention ending node (Algorithm 14) for 1D CP.

    Under 1D CP with ``(dp, cp)`` mesh, pair z is ``[B, I/cp, J, C_in]``.
    The ending node transposes I<->J, attends along I, then transposes back.
    Since I is sharded on cp, K/V/mask/bias rotate around the ring.

    Parameters
    ----------
    layer : SerialTriangleAttentionEndingNode
        The serial ending node module.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        The CP process group.
    """

    def __init__(
        self,
        layer: SerialTriangleAttentionEndingNode,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        if layer.starting:
            raise ValueError("Serial layer must have starting=False for TriangleAttentionEndingNode1D")

        self.device_mesh = device_mesh
        self.c_in = layer.c_in
        self.c_hidden = layer.c_hidden
        self.no_heads = layer.no_heads
        self.inf = layer.inf
        self.cp_group = cp_group

        self.layer_norm = LayerNormParamsReplicatedNoAutoCastBF16(layer.layer_norm, device_mesh)
        self.linear = LinearParamsReplicatedNoAutoCastBF16(layer.linear, device_mesh)

        mha = layer.mha
        self.mha = nn.Module()
        self.mha.linear_q = LinearParamsReplicatedNoAutoCastBF16(mha.linear_q, device_mesh)
        self.mha.linear_k = LinearParamsReplicatedNoAutoCastBF16(mha.linear_k, device_mesh)
        self.mha.linear_v = LinearParamsReplicatedNoAutoCastBF16(mha.linear_v, device_mesh)
        self.mha.linear_o = LinearParamsReplicatedNoAutoCastBF16(mha.linear_o, device_mesh)
        self.mha.linear_g = (
            LinearParamsReplicatedNoAutoCastBF16(mha.linear_g, device_mesh) if mha.linear_g is not None else None
        )
        self.mha.no_heads = mha.no_heads
        self.mha.c_hidden = mha.c_hidden
        self.mha.sigmoid = nn.Sigmoid()

    def _reshard_row_to_col(self, z_row: DTensor, cp_size: int) -> DTensor:
        """Re-shard a pair DTensor row-slab (Shard(0),Shard(1)) -> col-slab (Shard(0),Shard(2)).

        Wraps :class:`_PairReshardRowColImpl` with the DTensor boundary: to_local
        (row-slab local ``[B, I/cp, J, C]``), a2a (full=2 J, shard=1 I -> ``[B, I=N,
        J/cp, C]``), from_local with explicit col-slab placements + global shape/
        stride.  Reviewer-required: explicit shape+stride+placements on from_local.
        """
        global_shape = z_row.shape  # [B, I=N, J=N, C]
        z_row_local = z_row.to_local()  # [B, I/cp, J=N, C]
        z_col_local = _PairReshardRowColImpl.apply(
            z_row_local, self.cp_group, cp_size, 2, 1, global_shape
        )  # [B, I=N, J/cp, C]
        col_placements = (Shard(0), Shard(2))
        return DTensor.from_local(
            z_col_local,
            self.device_mesh,
            col_placements,
            shape=global_shape,
            stride=update_exhaustive_strides(z_col_local.shape, z_col_local.stride(), global_shape),
        )

    def _reshard_col_to_row(self, z_col: DTensor, cp_size: int) -> DTensor:
        """Re-shard a pair DTensor col-slab (Shard(0),Shard(2)) -> row-slab (Shard(0),Shard(1)).

        Inverse of :func:`_reshard_row_to_col`: a2a (full=1 I, shard=2 J -> ``[B,
        I/cp, J=N, C]``), from_local with explicit row-slab placements + global
        shape/stride.  Restores the pipeline's (Shard(0), Shard(1)) invariant.
        """
        global_shape = z_col.shape  # [B, I=N, J=N, C]
        z_col_local = z_col.to_local()  # [B, I=N, J/cp, C]
        z_row_local = _PairReshardRowColImpl.apply(
            z_col_local, self.cp_group, cp_size, 1, 2, global_shape
        )  # [B, I/cp, J=N, C]
        row_placements = (Shard(0), Shard(1))
        return DTensor.from_local(
            z_row_local,
            self.device_mesh,
            row_placements,
            shape=global_shape,
            stride=update_exhaustive_strides(z_row_local.shape, z_row_local.stride(), global_shape),
        )

    def forward(
        self,
        x: DTensor,
        mask: Optional[DTensor] = None,
        triattn_backend: TriAttnBackend = TriAttnBackend.REFERENCE,
    ) -> DTensor:
        """Forward pass with ring communication over I (cp).

        Parameters
        ----------
        x : DTensor
            ``[B, I, J, C_in]`` pair representation with placements
            ``(Shard(0), Shard(1))``.
        mask : DTensor, optional
            ``[B, I, J]`` pair mask with placements ``(Shard(0), Shard(1))``.
        triattn_backend : TriAttnBackend
            Backend to use for local attention computation.

        Returns
        -------
        DTensor
            ``[B, I, J, C_in]`` attention output, same placements as x.
        """
        if triattn_backend in (TriAttnBackend.TRIFAST, TriAttnBackend.CUEQ_FWD_TRIFAST_BWD):
            raise NotImplementedError(
                f"TriAttnBackend.{triattn_backend.name} is not supported for 1D CP triangle attention. "
                "Use REFERENCE or CUEQ."
            )
        if triattn_backend == TriAttnBackend.CUEQ and not cueq_is_installed:
            raise ValueError(
                "cuequivariance_torch is not installed. For Triangle Attention support, "
                "install using: pip install cuequivariance_ops_torch_cu13==<version> cuequivariance_torch==<version> "
                "where the 'version' tag can be found in the pyproject.toml file"
            )

        cp_size = dist.get_world_size(self.cp_group)

        # ---- z-first: re-shard the LN'd pair tensor ONCE to col-slab ----
        # x_normed: row-slab DTensor [B, I, J, C] (Shard(0), Shard(1)) -- I on cp.
        # z_col:    col-slab DTensor [B, I, J, C] (Shard(0), Shard(2)) -- J on cp,
        #           I full.  The single c_z-wide entry a2a (replaces the 3 q/k/v
        #           entry a2a's).  Projections then run LOCALLY on z_col.
        x_normed = self.layer_norm(x)  # row-slab DTensor
        z_col = self._reshard_row_to_col(x_normed, cp_size)  # col-slab DTensor

        # Projections on the col-slab z (proven _LinearParamsReplicatedImpl path;
        # grads ride that path's fp32 Partial(avg) reduction).  All col-slab
        # DTensors (Shard(0), Shard(2)).
        q_dt = self.mha.linear_q(z_col)  # [B, I=N, J/cp, H*c_hidden]
        k_dt = self.mha.linear_k(z_col)
        v_dt = self.mha.linear_v(z_col)
        bias_dt = self.linear(z_col)  # [B, I=N, J/cp, H]
        q_shape = q_dt.shape  # global [B, I=N, J=N, H*c_hidden]

        # Reviewer Option-(a) boundary: assert col-slab placements + reject
        # Partial on every projected DTensor BEFORE to_local (a mis-sharded
        # input is silent otherwise).
        col_placements = (Shard(0), Shard(2))
        for name, dt in (("q", q_dt), ("k", k_dt), ("v", v_dt), ("bias", bias_dt)):
            if dt.placements != col_placements:
                raise ValueError(
                    f"end-node z-first: {name} projection placements {dt.placements} "
                    f"!= expected col-slab {col_placements}"
                )

        q_local = q_dt.to_local()  # [B, I=N, J/cp, H*c_hidden]
        k_local = k_dt.to_local()
        v_local = v_dt.to_local()
        bias_col_proj = bias_dt.to_local()  # [B, I=row=N, J/cp=col, H]

        # Head-split + lay out each of q/k/v for the attn fn:
        # [B, I=N, J/cp, H*c_hidden] -> [B, I=N, J/cp, H, D] (head-split) ->
        # permute to [B, J/cp(batch), H, I=N(seq), D].
        q_local = q_local.view(q_local.shape[:-1] + (self.no_heads, -1)).permute(0, 2, 3, 1, 4).contiguous()
        k_local = k_local.view(k_local.shape[:-1] + (self.no_heads, -1)).permute(0, 2, 3, 1, 4).contiguous()
        v_local = v_local.view(v_local.shape[:-1] + (self.no_heads, -1)).permute(0, 2, 3, 1, 4).contiguous()

        # Mask: re-shard to col-slab like z, then to attn layout [B, J/cp, 1, 1, I_k=N].
        if mask is not None:
            # SF2: assert the input mask is row-slab (Shard(0),Shard(1)) before the
            # reshard, mirroring the q/k/v/bias/gate placement guards (a mis-sharded
            # mask is silent otherwise).  reject Partial via the equality.
            row_placements = (Shard(0), Shard(1))
            if mask.placements != row_placements:
                raise ValueError(
                    f"end-node z-first: mask placements {mask.placements} != expected row-slab {row_placements}"
                )
            mask_col_dt = self._reshard_row_to_col(mask.unsqueeze(-1), cp_size)  # [B, I=N, J/cp, 1]
            mask_local = mask_col_dt.to_local().squeeze(-1)  # [B, I=N, J/cp]
        else:
            mask_local = q_local.new_ones((q_local.shape[0], q_local.shape[3], q_local.shape[1]))  # [B, I=N, J/cp]
        # [B, I=row=N, J/cp=col] -> attn mask over (I_k=row): [B, J/cp, 1, 1, I_k=N]
        mask_local = mask_local.transpose(-1, -2)  # [B, J/cp, I=N]
        if triattn_backend == TriAttnBackend.CUEQ:
            mask_bias_local = mask_local[..., :, None, None, :].to(
                dtype=torch.bool, memory_format=torch.contiguous_format
            )
        else:
            mask_bias_local = (self.inf * (mask_local - 1))[..., :, None, None, :]

        q_scale = self.c_hidden**-0.5
        if triattn_backend == TriAttnBackend.REFERENCE:
            q_local = q_local * q_scale
            apply_scale = True
        else:
            apply_scale = False

        # Local attention (NO entry a2a; bias all-gathered on col inside the fn).
        o_col = _DAPTriangleAttentionEndingNode1DImpl.apply(
            q_local,
            k_local,
            v_local,
            mask_bias_local,
            bias_col_proj,
            self.cp_group,
            self.no_heads,
            self.c_hidden,
            triattn_backend,
            apply_scale,
            q_scale,
        )  # [B, J/cp, H, I=N, D]  col-slab

        # o_col -> [B, I=N, J/cp, H, D] (move I_seq back to a pair axis)
        o_local = o_col.permute(0, 3, 1, 2, 4)  # [B, I=N, J/cp, H, D]

        # Gating (projected on z_col, col-correct natively).
        if self.mha.linear_g is not None:
            g_dt = self.mha.linear_g(z_col)
            if g_dt.placements != col_placements:
                raise ValueError(f"end-node z-first: gate placements {g_dt.placements} != {col_placements}")
            g_local = self.mha.sigmoid(g_dt.to_local())  # [B, I=N, J/cp, H*c_hidden]
            g_local = g_local.view(g_local.shape[:-1] + (self.no_heads, -1))  # [B, I=N, J/cp, H, D]
            o_local = o_local * g_local

        # Flatten heads: [B, I=N, J/cp, H*c_hidden]  (col-slab local)
        o_local = o_local.reshape(o_local.shape[:-2] + (-1,))

        # linear_o on the col-slab DTensor (PRE-exit-a2a), then re-shard z back
        # to row-slab.  Wrap o_local as a col-slab DTensor with global shape
        # q_shape (the projected width); linear_o maps H*c_hidden -> c_z.
        o_col_dt = DTensor.from_local(
            o_local,
            self.device_mesh,
            col_placements,
            shape=q_shape,
            stride=update_exhaustive_strides(o_local.shape, o_local.stride(), q_shape),
        )
        o_col_dt = self.mha.linear_o(o_col_dt)  # col-slab [B, I=N, J/cp, c_z]

        # Exit reshard: col-slab -> row-slab, restoring the (Shard(0), Shard(1))
        # pipeline invariant.
        o_dt = self._reshard_col_to_row(o_col_dt, cp_size)
        return o_dt
