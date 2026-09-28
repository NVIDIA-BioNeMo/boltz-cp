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

"""DTensor-based context-parallel distogram loss for 1D CP (2D mesh ``(dp, cp)``).

Adapts the 2D CP distogram loss (distogram.py) for 1D CP where:
- Mesh is 2D ``(dp, cp)`` instead of 3D ``(dp, cp_axis_0, cp_axis_1)``
- Pair tensors ``[B, N, N, ...]`` use placement ``(Shard(0), Shard(1))`` —
  row-slab partitioning where each rank holds ``[B_local, N_row, N_full, ...]``
- Token mask ``[B, N]`` uses placement ``(Shard(0), Shard(1))``

Since each rank holds a row-slab (full columns), the pairwise mask requires
an all-gather of the token mask across cp to obtain the full column mask.
No TransposeComm is needed.

Communication budget:
  Forward (3 collective calls):
    1. all_gather over cp group for column mask (1 call)
    2. all_reduce over cp group for total (1 call)
    3. all_reduce over dp group for batch mean (1 call)
  Backward (0 collective calls):
    The backward of all_reduce(SUM) is identity — each rank computes
    gradients for its own local spatial chunk with no communication.

Equivalence to serial code (src/boltz/model/loss/distogramv2.py):
  Same as the 2D CP version — see distogram.py module docstring.
"""

import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh


def _build_pairwise_mask_local_1d(
    mask_local: torch.Tensor,
    cp_group: ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """Build pairwise [B, N_row, N_full] boolean mask from token mask [B, N_local].

    All-gathers the token mask across cp to get the full column mask, then
    computes the outer BITAND with the local row mask. Zeros the diagonal
    on the block where row and column ranges overlap.

    Parameters
    ----------
    mask_local : Tensor
        Local token mask [B_local, N_local].
    cp_group : ProcessGroup
        Process group for the cp mesh dimension.
    cp_size : int
        Number of ranks in the cp group.

    Returns
    -------
    Tensor
        Pairwise mask [B_local, N_row, N_full].
    """
    if cp_size == 1:
        # No communication needed — local mask covers the full sequence
        mask_full = mask_local
    else:
        # All-gather token mask across cp to get full column mask
        gathered = [torch.empty_like(mask_local) for _ in range(cp_size)]
        dist.all_gather(gathered, mask_local.contiguous(), group=cp_group)
        mask_full = torch.cat(gathered, dim=1)  # [B_local, N_full]

    # Outer BITAND: [B_local, N_row, 1] & [B_local, 1, N_full]
    mask_2d = mask_local.unsqueeze(2) & mask_full.unsqueeze(1)

    # Zero diagonal on the self-block (where row and column ranges overlap)
    cp_rank = dist.get_rank(cp_group)
    n_local = mask_local.shape[1]
    diag_offset = cp_rank * n_local
    # Create indices for the diagonal within the local block
    row_idx = torch.arange(n_local, device=mask_local.device)
    col_idx = row_idx + diag_offset
    mask_2d[:, row_idx, col_idx] = False

    return mask_2d


class _DistogramLoss1DCP(torch.autograd.Function):
    """Single autograd.Function for the full distogram loss under 1D CP.

    Forward: to_local() -> local math with explicit all_reduces -> from_local()
    Backward: local math only (no communication)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred: DTensor,
        target: DTensor,
        mask_token: DTensor,
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
        aggregate_distogram: bool,
    ) -> tuple[DTensor, DTensor]:
        """Forward pass.

        Parameters
        ----------
        pred : DTensor
            Prediction logits [B, N, N, D, bins], placements (Shard(0), Shard(1)).
        target : DTensor
            Target distributions [B, N, N, K, bins], placements (Shard(0), Shard(1)).
        mask_token : DTensor
            Token validity mask [B, N], placements (Shard(0), Shard(1)).
        device_mesh : DeviceMesh
            2D device mesh (dp, cp).
        dp_group : ProcessGroup
            Process group for dp mesh dimension.
        cp_group : ProcessGroup
            Process group for cp mesh dimension.
        aggregate_distogram : bool
            Whether to aggregate target over K conformers.

        Returns
        -------
        global_loss : DTensor
            Scalar loss, placements (Replicate(), Replicate()).
        batch_loss : DTensor
            Per-example loss [B], placements (Shard(0), Replicate()).
        """
        # --- Validate inputs ---
        if not isinstance(pred, DTensor):
            raise TypeError(f"pred must be DTensor, got {type(pred)}")
        if not isinstance(target, DTensor):
            raise TypeError(f"target must be DTensor, got {type(target)}")
        if not isinstance(mask_token, DTensor):
            raise TypeError(f"mask_token must be DTensor, got {type(mask_token)}")

        for name, dtensor in [("target", target), ("mask_token", mask_token)]:
            if dtensor.device_mesh != device_mesh:
                raise ValueError(f"{name} has different device_mesh than pred")
        if pred.device_mesh != device_mesh:
            raise ValueError("pred has different device_mesh than expected")

        expected_pair = (Shard(0), Shard(1))
        if pred.placements != expected_pair:
            raise ValueError(f"pred placements {pred.placements} must be {expected_pair}")
        if target.placements != expected_pair:
            raise ValueError(f"target placements {target.placements} must be {expected_pair}")
        expected_mask = (Shard(0), Shard(1))
        if mask_token.placements != expected_mask:
            raise ValueError(f"mask_token placements {mask_token.placements} must be {expected_mask}")

        for name, dtensor in [("pred", pred), ("target", target), ("mask_token", mask_token)]:
            for i_dim, placement in enumerate(dtensor.placements):
                if isinstance(placement, Partial):
                    raise ValueError(f"Partial placement on {name} mesh dim {i_dim} is not supported")
                elif isinstance(placement, Shard):
                    if dtensor.shape[placement.dim] % device_mesh.shape[i_dim] != 0:
                        raise ValueError(
                            f"Uneven sharding {name} tensor dimension {placement.dim} of size "
                            f"{dtensor.shape[placement.dim]} along device mesh dimension {i_dim} "
                            f"of size {device_mesh.shape[i_dim]} is not supported"
                        )

        if pred.ndim != 5:  # noqa: PLR2004
            raise ValueError(f"pred must be 5D [B, N, N, D, bins], got {pred.ndim}D")
        if target.ndim != 5:  # noqa: PLR2004
            raise ValueError(f"target must be 5D [B, N, N, K, bins], got {target.ndim}D")
        if mask_token.ndim != 2:  # noqa: PLR2004
            raise ValueError(f"mask_token must be 2D [B, N], got {mask_token.ndim}D")
        if pred.shape[0] != target.shape[0] or pred.shape[1] != target.shape[1] or pred.shape[2] != target.shape[2]:
            raise ValueError(f"pred shape {pred.shape} and target shape {target.shape} must match on dims 0,1,2")
        if pred.shape[4] != target.shape[4]:
            raise ValueError(f"pred bins {pred.shape[4]} != target bins {target.shape[4]}")
        if mask_token.shape[0] != pred.shape[0] or mask_token.shape[1] != pred.shape[1]:
            raise ValueError(f"mask_token shape {mask_token.shape} inconsistent with pred shape {pred.shape}")

        # --- Extract local tensors ---
        compute_dtype = torch.promote_types(torch.promote_types(pred.dtype, target.dtype), torch.float32)
        pred_local = pred.to_local().to(compute_dtype)  # [B_local, N_row, N_full, D, bins]
        target_local = target.to_local().to(compute_dtype)  # [B_local, N_row, N_full, K, bins]
        mask_token_local = mask_token.to_local().to(torch.bool)  # [B_local, N_local]

        D = pred_local.shape[3]  # noqa: N806
        K = target_local.shape[3]  # noqa: N806

        # --- Build pairwise mask ---
        cp_size = dist.get_world_size(cp_group)
        mask_local = _build_pairwise_mask_local_1d(mask_token_local, cp_group, cp_size).to(compute_dtype)
        # mask_local: [B_local, N_row, N_full]

        # --- Denom: launch async all_reduce so latency overlaps with compute ---
        denom_local = mask_local.sum(dim=(-1, -2))  # [B_local]
        denom_work = dist.all_reduce(denom_local, op=dist.ReduceOp.SUM, group=cp_group, async_op=True)

        # --- Target preparation ---
        if aggregate_distogram:
            P_local = target_local.sum(dim=3)  # [B, N_r, N_full, bins]
            P_denom = P_local.sum(dim=-1, keepdim=True).clamp(min=1)
            P_local = P_local / P_denom
            P_local = P_local.unsqueeze(3)  # [B, N_r, N_full, 1, bins]
            K_eff = 1  # noqa: N806
        else:
            P_local = target_local
            K_eff = K  # noqa: N806

        # --- Vectorized cross-entropy for all (k, d) pairs ---
        log_Q_local = torch.nn.functional.log_softmax(pred_local, dim=-1)
        softmax_local = log_Q_local.exp()

        P_expanded = P_local.unsqueeze(4).expand(-1, -1, -1, -1, D, -1)
        log_Q_expanded = log_Q_local.unsqueeze(3).expand(-1, -1, -1, K_eff, -1, -1)

        errors_local = -(P_expanded * log_Q_expanded).sum(dim=-1)  # [B, N_r, N_full, K_eff, D]

        # --- Flatten K_eff*D, apply mask, spatial reduction ---
        errors_flat = errors_local.reshape(
            errors_local.shape[0], errors_local.shape[1], errors_local.shape[2], K_eff * D
        )
        mask_exp = mask_local.unsqueeze(-1).expand(-1, -1, -1, K_eff * D)
        masked = errors_flat * mask_exp

        # --- Reduce over spatial dims, all_reduce over cp for row-slab partial sums ---
        total_local = masked.sum(dim=(1, 2))  # [B_local, K_eff*D]
        dist.all_reduce(total_local, op=dist.ReduceOp.SUM, group=cp_group)
        denom_work.wait()

        denom_local = denom_local + 1e-5
        total_local = total_local / denom_local.unsqueeze(-1)

        # --- Min over D, mean over K_eff ---
        batch_loss_kd = total_local.reshape(total_local.shape[0], K_eff, D)
        min_result = torch.min(batch_loss_kd, dim=-1)
        batch_loss_k = min_result.values
        min_indices = min_result.indices

        batch_loss_local = batch_loss_k.sum(dim=-1) / K_eff  # [B_local]

        # --- Global loss: mean over batch (all_reduce over DP dim) ---
        B_global = pred.shape[0]  # noqa: N806
        global_loss_local = batch_loss_local.sum(dim=0, keepdim=False)
        dist.all_reduce(global_loss_local, op=dist.ReduceOp.SUM, group=dp_group)
        global_loss_local = global_loss_local / B_global

        # --- Save for backward ---
        if pred.requires_grad:
            ctx.save_for_backward(
                softmax_local,
                P_local,
                mask_local,
                denom_local,
                min_indices,
            )
            ctx.D = D
            ctx.K_eff = K_eff
            ctx.B_global = B_global
            ctx.device_mesh = device_mesh
            ctx.pred_placements = pred.placements
            ctx.pred_shape = pred.shape
            ctx.pred_stride = pred.stride()

        # --- Wrap results as DTensors ---
        batch_loss_placements = (Shard(0), Replicate())
        bl_shape = (B_global,)
        bl_stride = (1,)
        batch_loss_dt = DTensor.from_local(
            batch_loss_local, device_mesh, batch_loss_placements, shape=bl_shape, stride=bl_stride
        )

        global_loss_placements = (Replicate(), Replicate())
        global_loss_dt = DTensor.from_local(global_loss_local, device_mesh, global_loss_placements, shape=(), stride=())

        return global_loss_dt, batch_loss_dt

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx: FunctionCtx, d_global_loss: DTensor, d_batch_loss: DTensor
    ) -> tuple[DTensor | None, None, None, None, None, None, None]:
        """Backward pass — entirely local, no collective communication."""
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None, None

        softmax_local, P_local, mask_local, denom_local, min_indices = ctx.saved_tensors
        D = ctx.D  # noqa: N806
        K_eff = ctx.K_eff  # noqa: N806
        B_global = ctx.B_global  # noqa: N806
        device_mesh = ctx.device_mesh

        B_local = softmax_local.shape[0]  # noqa: N806
        N_row = softmax_local.shape[1]  # noqa: N806
        N_full = softmax_local.shape[2]  # noqa: N806

        compute_dtype = softmax_local.dtype
        d_gl = (d_global_loss.to_local() if isinstance(d_global_loss, DTensor) else d_global_loss).to(compute_dtype)
        if d_batch_loss is None:
            d_bl = torch.zeros(B_local, device=softmax_local.device, dtype=compute_dtype)
        elif isinstance(d_batch_loss, DTensor):
            d_bl = d_batch_loss.to_local().to(compute_dtype)
        else:
            d_bl = d_batch_loss.to(compute_dtype)

        # Chain rule from global_loss = sum(batch_loss) / B_global
        d_batch_local = d_bl + d_gl / B_global

        # Backward through mean over K_eff
        d_batch_loss_k = (d_batch_local / K_eff).unsqueeze(-1).expand(-1, K_eff)

        # Backward through min over D
        d_batch_loss_kd = torch.zeros(B_local, K_eff, D, device=d_batch_local.device, dtype=d_batch_local.dtype)
        d_batch_loss_kd.scatter_(-1, min_indices.unsqueeze(-1), d_batch_loss_k.unsqueeze(-1))

        # Backward through reshape and division by denom
        d_total = d_batch_loss_kd.reshape(B_local, K_eff * D)
        d_total = d_total / denom_local.unsqueeze(-1)

        # Backward through spatial sum + all_reduce (identity for SUM backward)
        d_masked = d_total.unsqueeze(1).unsqueeze(2).expand(-1, N_row, N_full, -1)

        # Backward through mask multiply
        mask_exp = mask_local.unsqueeze(-1).expand(-1, -1, -1, K_eff * D)
        d_errors_flat = d_masked * mask_exp

        # Reshape to [B, N_r, N_full, K_eff, D]
        d_errors = d_errors_flat.reshape(B_local, N_row, N_full, K_eff, D)

        # Backward through cross-entropy
        P_expanded = P_local.unsqueeze(4).expand(-1, -1, -1, -1, D, -1)
        d_log_Q_expanded = -P_expanded * d_errors.unsqueeze(-1)
        d_log_Q = d_log_Q_expanded.sum(dim=3)

        # Backward through log_softmax
        d_pred_local = d_log_Q - softmax_local * d_log_Q.sum(dim=-1, keepdim=True)

        d_pred = DTensor.from_local(
            d_pred_local,
            device_mesh=device_mesh,
            placements=ctx.pred_placements,
            shape=ctx.pred_shape,
            stride=ctx.pred_stride,
        )

        return d_pred, None, None, None, None, None, None


def distogram_loss_1d(
    output: dict[str, DTensor],
    feats: dict[str, DTensor],
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    aggregate_distogram: bool = True,
) -> tuple[DTensor, DTensor]:
    """Compute the distogram loss for 1D CP using a single fused autograd.Function.

    Parameters
    ----------
    output : dict[str, DTensor]
        Output of the model containing:
        - "pdistogram": [B, N, N, D, bins] prediction logits (DTensor).
    feats : dict[str, DTensor]
        Input features containing:
        - "disto_target": [B, N, N, K, bins] target distributions (DTensor).
        - "token_disto_mask": [B, N] token validity mask (DTensor).
    device_mesh : DeviceMesh
        2D device mesh (dp, cp).
    dp_group : ProcessGroup
        Process group for the dp mesh dimension.
    cp_group : ProcessGroup
        Process group for the cp mesh dimension.
    aggregate_distogram : bool
        If True, aggregates target over K conformers.

    Returns
    -------
    DTensor
        The globally averaged loss (scalar DTensor).
    DTensor
        Per-example loss [B] (DTensor).
    """
    with torch.autocast("cuda", enabled=False):
        return _DistogramLoss1DCP.apply(
            output["pdistogram"],
            feats["disto_target"],
            feats["token_disto_mask"],
            device_mesh,
            dp_group,
            cp_group,
            aggregate_distogram,
        )
