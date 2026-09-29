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

"""DTensor-based context-parallel B-factor loss for 1D CP (2D mesh ``(dp, cp)``).

Adapts the 2D CP B-factor loss (bfactor.py) for 1D CP where:
- Mesh is 2D ``(dp, cp)`` instead of 3D ``(dp, cp_axis_0, cp_axis_1)``
- pred ``[B, N, bins]`` uses placement ``(Shard(0), Shard(1))``
- ``token_to_rep_atom`` ``[B, N_tokens, N_atoms]`` uses placement
  ``(Shard(0), Shard(1))`` — token-sharded with full atom axis
- ``bfactor`` ``[B, N_atoms]`` is redistributed to ``(Shard(0), Replicate())``
  at the wrapper boundary so its atom axis aligns with the full atom axis
  of ``token_to_rep_atom`` for the bmm projection (mirrors the canonical
  pattern in ``confidence_1d.resolved_loss_1d``).
- There is no cp1 (Replicate) dimension — all cp ranks hold unique data

The B-factor loss is per-token (no pairwise interaction).  The serial code
computes a single global fraction::

    loss = sum_{b,n}(errors * mask) / (sum_{b,n}(mask) + eps)

The distributed version reduces both numerator and denominator globally
across dp and cp before dividing.

Communication budget:
  Forward (1 all_gather + 2 all_reduce calls):
    1. all_gather of bfactor atom axis via redistribute to Replicate (1 call)
    2. all_reduce(SUM) over dp group for packed [loss_sum, mask_sum] (1 call)
    3. all_reduce(SUM) over cp group for the same packed tensor (1 call)
  Backward (0 collective calls):
    The backward of all_reduce(SUM) is identity.
"""

import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh


class _BFactorLoss1DCP(torch.autograd.Function):
    """Single autograd.Function for the full B-factor loss under 1D CP.

    Forward: to_local() -> local math with explicit all_reduces -> from_local()
    Backward: local math only (no communication)
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred: DTensor,
        token_to_rep_atom: DTensor | torch.Tensor,
        bfactor: DTensor | torch.Tensor,
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        pred : DTensor
            Predicted B-factor logits [B, N, bins], placements (Shard(0), Shard(1)).
        token_to_rep_atom : DTensor | Tensor
            Token-to-representative-atom mapping [B, N_tokens, max_atoms_per_shard].
        bfactor : DTensor | Tensor
            Per-atom B-factors [B, A].
        device_mesh : DeviceMesh
            2D device mesh (dp, cp).
        dp_group : ProcessGroup
            Process group for the dp mesh dimension.
        cp_group : ProcessGroup
            Process group for the cp mesh dimension.

        Returns
        -------
        global_loss : DTensor
            Scalar loss, placements (Replicate(), Replicate()).
        """
        # --- Validate differentiable input ---
        if not isinstance(pred, DTensor):
            raise TypeError(f"pred must be DTensor, got {type(pred)}")

        expected_single = (Shard(0), Shard(1))
        if pred.placements != expected_single:
            raise ValueError(f"pred placements {pred.placements} must be {expected_single}")
        for i_dim, placement in enumerate(pred.placements):
            if isinstance(placement, Partial):
                raise ValueError(f"Partial placement on pred mesh dim {i_dim} is not supported")
            if isinstance(placement, Shard) and pred.shape[placement.dim] % device_mesh.shape[i_dim] != 0:
                raise ValueError(
                    f"Uneven sharding pred tensor dim {placement.dim} of size "
                    f"{pred.shape[placement.dim]} along mesh dim {i_dim} of size "
                    f"{device_mesh.shape[i_dim]}"
                )

        for name, tensor in [("token_to_rep_atom", token_to_rep_atom), ("bfactor", bfactor)]:
            if not isinstance(tensor, (DTensor, torch.Tensor)):
                raise TypeError(f"{name} must be DTensor or Tensor, got {type(tensor)}")

        # token_to_rep_atom must keep full atom axis (last dim) so the bmm
        # contracts over the full global atom axis (matching serial semantics).
        if isinstance(token_to_rep_atom, DTensor) and token_to_rep_atom.placements != expected_single:
            raise ValueError(f"token_to_rep_atom placements {token_to_rep_atom.placements} must be {expected_single}")

        # bfactor must arrive atom-Replicate so its [B_local, N_atoms_global]
        # slice aligns with token_to_rep_atom's [B_local, N_local, N_atoms_global].
        expected_atom = (Shard(0), Replicate())
        if isinstance(bfactor, DTensor) and bfactor.placements != expected_atom:
            raise ValueError(f"bfactor placements {bfactor.placements} must be {expected_atom}")

        # --- Extract local tensors ---
        compute_dtype = torch.promote_types(pred.dtype, torch.float32)
        pred_local = pred.to_local().to(compute_dtype)  # [B_local, N_local, bins]
        t2ra_local = (token_to_rep_atom.to_local() if isinstance(token_to_rep_atom, DTensor) else token_to_rep_atom).to(
            compute_dtype
        )
        bf_local = (bfactor.to_local() if isinstance(bfactor, DTensor) else bfactor).to(compute_dtype)

        bins = pred_local.shape[2]

        # --- Construct target (non-differentiable) ---
        bfactor_token = torch.bmm(t2ra_local, bf_local.unsqueeze(-1))  # [B_local, N_local, 1]

        boundaries = torch.linspace(0, 100, bins - 1, device=pred_local.device, dtype=compute_dtype)
        bfactor_token_bin = (bfactor_token > boundaries).sum(dim=-1).long()  # [B_local, N_local]
        bfactor_target = torch.nn.functional.one_hot(bfactor_token_bin, num_classes=bins).to(compute_dtype)

        token_mask = (bfactor_token > 1e-5).squeeze(-1).to(compute_dtype)  # [B_local, N_local]

        # --- Cross-entropy loss ---
        log_softmax_local = torch.nn.functional.log_softmax(pred_local, dim=-1)
        softmax_local = log_softmax_local.exp()

        errors = -(bfactor_target * log_softmax_local).sum(dim=-1)  # [B_local, N_local]
        masked_errors = errors * token_mask

        # --- Global reduction ---
        # Serial: loss = sum_{b,n}(errors * mask) / (sum_{b,n}(mask) + eps)
        # Reduce over dp and cp in two sequential calls.
        packed = torch.stack([masked_errors.sum(), token_mask.sum()])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=dp_group)
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=cp_group)

        global_denom = packed[1] + 1e-5
        global_loss_local = packed[0] / global_denom

        # --- Save for backward ---
        if pred.requires_grad:
            ctx.save_for_backward(
                softmax_local,
                bfactor_target,
                token_mask,
                global_denom.unsqueeze(0),
            )
            ctx.device_mesh = device_mesh
            ctx.pred_placements = pred.placements
            ctx.pred_shape = pred.shape
            ctx.pred_stride = pred.stride()

        # --- Wrap result as DTensor ---
        global_loss_placements = (Replicate(), Replicate())
        return DTensor.from_local(
            global_loss_local,
            device_mesh,
            global_loss_placements,
            shape=(),
            stride=(),
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, d_global_loss: DTensor) -> tuple[DTensor | None, None, None, None, None, None]:
        """Backward pass — entirely local, no collective communication."""
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None

        softmax_local, bfactor_target, token_mask, (global_denom,) = ctx.saved_tensors
        device_mesh = ctx.device_mesh

        d_gl = (d_global_loss.to_local() if isinstance(d_global_loss, DTensor) else d_global_loss).to(
            softmax_local.dtype
        )

        scale = d_gl / global_denom
        d_pred_local = (softmax_local - bfactor_target) * token_mask.unsqueeze(-1) * scale

        d_pred = DTensor.from_local(
            d_pred_local,
            device_mesh=device_mesh,
            placements=ctx.pred_placements,
            shape=ctx.pred_shape,
            stride=ctx.pred_stride,
        )

        return d_pred, None, None, None, None, None


def bfactor_loss_1d(
    output: dict[str, DTensor],
    feats: dict[str, DTensor],
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
) -> DTensor:
    """Compute the B-factor loss for 1D CP using a single fused autograd.Function.

    Parameters
    ----------
    output : dict[str, DTensor]
        Model outputs containing:
        - "pbfactor": [B, N, bins] predicted B-factor logits (DTensor).
    feats : dict[str, DTensor]
        Input features containing:
        - "token_to_rep_atom": [B, N_tokens, max_atoms_per_shard]
          token-to-atom mapping (DTensor).
        - "bfactor": [B, A] per-atom B-factors (DTensor).
    device_mesh : DeviceMesh
        2D device mesh (dp, cp).
    dp_group : ProcessGroup
        Process group for the dp mesh dimension.
    cp_group : ProcessGroup
        Process group for the cp mesh dimension.

    Returns
    -------
    DTensor
        The globally averaged B-factor loss (scalar DTensor).
    """
    # Redistribute bfactor's atom axis to Replicate so the bmm against
    # token_to_rep_atom (which holds the full atom axis in its last dim)
    # is valid. Atoms may arrive as either (Shard(0), Replicate()) or
    # (Shard(0), Shard(1)) depending on the feature pipeline; this normalises
    # both. Mirrors the canonical pattern in confidence_1d.resolved_loss_1d.
    bfactor = feats["bfactor"]
    if isinstance(bfactor, DTensor):
        bfactor = bfactor.redistribute(device_mesh, (Shard(0), Replicate()))

    with torch.autocast("cuda", enabled=False):
        return _BFactorLoss1DCP.apply(
            output["pbfactor"],
            feats["token_to_rep_atom"],
            bfactor,
            device_mesh,
            dp_group,
            cp_group,
        )
