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

"""DTensor-based context-parallel confidence loss for 1D CP (2D mesh ``(dp, cp)``).

Adapts the 2D CP confidence loss (confidencev2.py) for 1D CP where:
- Mesh is 2D ``(dp, cp)`` instead of 3D ``(dp, cp_axis_0, cp_axis_1)``
- Single tensors ``[B, N, ...]`` use placement ``(Shard(0), Shard(1))``
- Pair tensors ``[B, N, N, ...]`` use placement ``(Shard(0), Shard(1))`` (row-slab)
- Atom tensors ``[B, N_atoms, ...]`` use placement ``(Shard(0), Replicate())``
- Scalar / replicated: ``(Shard(0), Replicate())`` or ``(Replicate(), Replicate())``

No TransposeComm is needed for 1D CP — row-slab sharding gives each rank
full columns in pair tensors.

Sub-losses:
- **resolved_loss**: Per-token NLL, shardwise (no cross-CP comm for NLL).
  Reduction: all_reduce(SUM) over cp, then dp.
- **plddt_loss**: Per-token lDDT score via cdist between local row tokens
  and all-gathered R-set columns. The per-token R-set summation is already
  complete locally (R-set is gathered, tokens are sharded), so no cp
  reduction is needed for the target lDDT. The cross-entropy is reduced
  over cp (per-sample numerator/denominator over local tokens) and dp.
- **pde_loss**: Row-slab pair cdist, all-gather column token coords, compute
  target PDE, cross-entropy with row-slab pred_pde logits.  All_reduce
  over cp for partial sums, then dp.
- **pae_loss**: Frame computation via gather-to-local + serial
  ``_compute_frame_pred``, then distributed PAE target via all-gather of
  column coordinates. Cross-entropy with row-slab pred_pae logits.

Communication budget:
  resolved_loss: 2 all_reduce (cp + dp)
  plddt_loss:    1 all_gather (cp, R-set coords) + 1 all_reduce (cp) on
                 cross-entropy num/den + 1 all_reduce (dp) on batch-summed loss
  pde_loss:      1 all_gather (cp, token coords) + 2 all_reduce (cp + dp)
  pae_loss:      all_gathers for frame computation + 2 all_reduce (cp + dp)
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd.function import FunctionCtx
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.device_mesh import DeviceMesh

from boltz.data import const
from boltz.distributed.model.layers.atom_to_token import single_repr_token_to_atom
from boltz.distributed.model.loss.confidencev2 import _compute_frame_pred
from boltz.distributed.utils import LayoutRightMap, all_gather_on_cp

_REPLICATED = (Replicate(), Replicate())

# Atom-DTensor inputs to the loss wrappers are redistributed to this placement
# before validation. The inner autograd.Functions require (Shard(0), Replicate())
# because their einsums and frame indexing operate on the full N_atom axis;
# callers may pass atoms as either (Shard(0), Replicate()) (already-replicated)
# or (Shard(0), Shard(1)) (the post-694573d2 sharded convention).
_ATOM_REPLICATED = (Shard(0), Replicate())

# Numerical stability constants
_EPS_DENOM = 1e-7  # denominator clamping for loss normalization
_EPS_LDDT = 1e-10  # lddt normalization epsilon
_EPS_PAE_DIST = 1e-8  # prevents sqrt(0) in PAE target distance
_EPS_FRAME_NORM = 1e-5  # prevents division by zero in frame basis normalization


def _validate_dtensor(
    tensor: DTensor,
    name: str,
    device_mesh: DeviceMesh,
    expected_placements: tuple,
    expected_ndim: int | None = None,
) -> None:
    """Validate a DTensor input for autograd.Function conformance.

    Checks: DTensor type, device_mesh match, expected placements, no Partial,
    even sharding.
    """
    if not isinstance(tensor, DTensor):
        raise TypeError(f"{name} must be DTensor, got {type(tensor)}")
    if tensor.device_mesh != device_mesh:
        raise ValueError(f"{name} device_mesh mismatch: expected {device_mesh}, got {tensor.device_mesh}")
    if tensor.placements != expected_placements:
        raise ValueError(f"{name} placements {tensor.placements} != expected {expected_placements}")
    for i_dim, placement in enumerate(tensor.placements):
        if isinstance(placement, Partial):
            raise ValueError(f"Partial placement on {name} mesh dim {i_dim} is not supported")
        if isinstance(placement, Shard) and tensor.shape[placement.dim] % device_mesh.shape[i_dim] != 0:
            raise ValueError(
                f"Uneven sharding: {name} tensor dim {placement.dim} of size "
                f"{tensor.shape[placement.dim]} along mesh dim {i_dim} of size "
                f"{device_mesh.shape[i_dim]}"
            )
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ValueError(f"{name} must be {expected_ndim}D, got {tensor.ndim}D")


# ---------------------------------------------------------------------------
# resolved_loss_1d
# ---------------------------------------------------------------------------


class _ResolvedLoss1D(torch.autograd.Function):
    """Fused resolved loss for 1D CP.

    Forward: to_local() -> local NLL + mask -> all_reduce(cp+dp) -> scalar loss
    Backward: local autograd subgraph (all_reduce(SUM) has identity gradient)

    Communication budget:
      Forward:  2 all_reduce(SUM) — 1 cp, 1 dp
      Backward: 0 collectives
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred_resolved: DTensor,
        token_to_rep_atom: DTensor,
        true_coords_resolved_mask: DTensor,
        token_pad_mask: DTensor,
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
        multiplicity: int,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        pred_resolved : DTensor
            [B*mult, N, 2], placements (Shard(0), Shard(1)).
        token_to_rep_atom : DTensor
            [B, N, N_atom], placements (Shard(0), Shard(1)).
        true_coords_resolved_mask : DTensor
            [B*mult, N_atom], placements (Shard(0), Replicate()).
        token_pad_mask : DTensor
            [B, N], placements (Shard(0), Shard(1)).
        """
        _SINGLE = (Shard(0), Shard(1))
        _ATOM = (Shard(0), Replicate())
        _validate_dtensor(pred_resolved, "pred_resolved", device_mesh, _SINGLE, expected_ndim=3)
        _validate_dtensor(token_to_rep_atom, "token_to_rep_atom", device_mesh, _SINGLE, expected_ndim=3)
        _validate_dtensor(true_coords_resolved_mask, "true_coords_resolved_mask", device_mesh, _ATOM, expected_ndim=2)
        _validate_dtensor(token_pad_mask, "token_pad_mask", device_mesh, _SINGLE, expected_ndim=2)

        compute_dtype = torch.promote_types(pred_resolved.dtype, torch.float32)
        pred_local = pred_resolved.to_local().to(compute_dtype).detach().requires_grad_(pred_resolved.requires_grad)
        t2ra_local = token_to_rep_atom.to_local().to(compute_dtype)  # [B_l, N_l, N_atom]
        mask_local = true_coords_resolved_mask.to_local().to(compute_dtype)  # [B_l*mult, N_atom]
        pad_local = token_pad_mask.to_local().to(compute_dtype)  # [B_l, N_l]

        B_local = t2ra_local.shape[0]

        with torch.enable_grad():
            # ref_mask via einsum to avoid repeat_interleave
            mask_reshaped = mask_local.view(B_local, multiplicity, -1)
            ref_mask = torch.einsum("btn,bmn->bmt", t2ra_local, mask_reshaped).flatten(0, 1)

            log_probs = F.log_softmax(pred_local, dim=-1)
            errors = -ref_mask * log_probs[:, :, 0] - (1 - ref_mask) * log_probs[:, :, 1]

            # Expand pad_mask with multiplicity
            pad_expanded = pad_local.unsqueeze(1).expand(-1, multiplicity, -1).reshape(-1, pad_local.shape[1])

            numerator = (errors * pad_expanded).sum(dim=-1)  # [B_l*mult]
            denominator = pad_expanded.sum(dim=-1)  # [B_l*mult]

            # All-reduce over cp (token dimension sharded)
            num = numerator.clone()
            den = denominator.clone()
            with torch.no_grad():
                dist.all_reduce(num, op=dist.ReduceOp.SUM, group=cp_group)
                dist.all_reduce(den, op=dist.ReduceOp.SUM, group=cp_group)

            per_sample = num / den.clamp(min=_EPS_DENOM)
            loss_sum = per_sample.sum()

            # All-reduce over dp (batch dimension sharded)
            loss_val = loss_sum.clone()
            with torch.no_grad():
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM, group=dp_group)

            B_global_mult = pred_resolved.shape[0]
            loss_local = loss_val / B_global_mult

        ctx.save_for_backward(pred_local, loss_local)
        ctx.device_mesh = device_mesh
        ctx.pred_placements = pred_resolved.placements
        ctx.pred_shape = pred_resolved.shape
        ctx.pred_stride = pred_resolved.stride()

        return DTensor.from_local(loss_local.detach(), device_mesh, _REPLICATED, shape=torch.Size(()), stride=())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, grad_loss: DTensor):
        pred_local, loss_local = ctx.saved_tensors
        if not pred_local.requires_grad:
            return None, None, None, None, None, None, None, None

        grad_local = grad_loss.to_local()
        (d_pred,) = torch.autograd.grad(
            outputs=[loss_local], inputs=[pred_local], grad_outputs=[grad_local], retain_graph=False
        )
        d_pred_dt = DTensor.from_local(
            d_pred, ctx.device_mesh, ctx.pred_placements, shape=ctx.pred_shape, stride=ctx.pred_stride
        )
        return d_pred_dt, None, None, None, None, None, None, None


def resolved_loss_1d(
    pred_resolved: DTensor,
    feats: dict[str, DTensor],
    true_coords_resolved_mask: DTensor,
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    multiplicity: int = 1,
) -> DTensor:
    """Compute resolved loss for 1D CP.

    Parameters
    ----------
    pred_resolved : DTensor
        [B*mult, N, 2], placements (Shard(0), Shard(1)).
    feats : dict
        Must contain "token_to_rep_atom" and "token_pad_mask".
    true_coords_resolved_mask : DTensor
        [B*mult, N_atom]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    device_mesh : DeviceMesh
    dp_group, cp_group : ProcessGroup
    multiplicity : int

    Returns
    -------
    DTensor
        Scalar loss, placements (Replicate(), Replicate()).
    """
    # Redistribute atom-DTensor inputs to the canonical (Shard(0), Replicate())
    # before the autograd.Function boundary; the inner forward asserts this
    # placement and the einsums project atoms via the full N_atom axis.
    true_coords_resolved_mask = true_coords_resolved_mask.redistribute(device_mesh, _ATOM_REPLICATED)
    return _ResolvedLoss1D.apply(
        pred_resolved,
        feats["token_to_rep_atom"],
        true_coords_resolved_mask,
        feats["token_pad_mask"],
        device_mesh,
        dp_group,
        cp_group,
        multiplicity,
    )


# ---------------------------------------------------------------------------
# plddt_loss_1d
# ---------------------------------------------------------------------------


class _PLDDTLoss1D(torch.autograd.Function):
    """Fused pLDDT loss for 1D CP.

    Computes token-level lDDT targets (no gradient), then cross-entropy
    loss against predicted lDDT bins (gradient flows through log_softmax).

    The lDDT computation requires pairwise distances between token
    coordinates (row) and R-set coordinates (column). Under 1D CP,
    tokens are sharded but atoms are replicated — so token coords and
    R-set coords are both local after the einsum projection. However,
    the cdist requires the full column dimension, so we all-gather
    R-set coords across cp.

    Communication budget:
      Forward:  1 all_gather (R-set coords, masks, cutoffs) +
                2 all_reduce(SUM) (cp + dp) on the per-sample cross-entropy
                numerator/denominator and 1 all_reduce(SUM) (dp) on the
                batch-summed scalar loss.
      Backward: 0 collectives
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred_lddt: DTensor,
        pred_atom_coords: DTensor,
        true_atom_coords: DTensor,
        true_coords_resolved_mask: DTensor,
        token_to_rep_atom: DTensor,
        r_set_to_rep_atom: DTensor,
        atom_to_token: DTensor,
        mol_type: DTensor,
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
        multiplicity: int,
    ) -> DTensor:
        _SINGLE = (Shard(0), Shard(1))
        _ATOM = (Shard(0), Replicate())
        _validate_dtensor(pred_lddt, "pred_lddt", device_mesh, _SINGLE, expected_ndim=3)
        _validate_dtensor(pred_atom_coords, "pred_atom_coords", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(true_atom_coords, "true_atom_coords", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(true_coords_resolved_mask, "true_coords_resolved_mask", device_mesh, _ATOM, expected_ndim=2)
        _validate_dtensor(token_to_rep_atom, "token_to_rep_atom", device_mesh, _SINGLE, expected_ndim=3)
        _validate_dtensor(r_set_to_rep_atom, "r_set_to_rep_atom", device_mesh, _SINGLE, expected_ndim=3)
        _validate_dtensor(atom_to_token, "atom_to_token", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(mol_type, "mol_type", device_mesh, _SINGLE, expected_ndim=2)

        cp_size = device_mesh.shape[1]
        compute_dtype = torch.promote_types(pred_lddt.dtype, torch.float32)

        # Extract locals
        pred_lddt_local = pred_lddt.to_local().detach().requires_grad_(pred_lddt.requires_grad)

        t2ra_local = token_to_rep_atom.to_local().to(compute_dtype)  # [B_l, N_l, N_atom]
        r2ra_local = r_set_to_rep_atom.to_local().to(compute_dtype)  # [B_l, N_R_l, N_atom]
        a2t_local = atom_to_token.to_local().to(compute_dtype)  # [B_l, N_atom, N_tok_max]
        mol_local = mol_type.to_local()  # [B_l, N_l]
        pred_coords_local = pred_atom_coords.to_local().to(compute_dtype)  # [B_l*m, N_atom, 3]
        true_coords_local = true_atom_coords.to_local().to(compute_dtype)
        resolved_local = true_coords_resolved_mask.to_local().to(compute_dtype)

        B_local = t2ra_local.shape[0]
        N_local = t2ra_local.shape[1]

        # Compute target lDDT (no gradient)
        with torch.no_grad():
            # Project atom coords to token and R-set space
            pred_reshaped = pred_coords_local.view(B_local, multiplicity, -1, 3)
            true_reshaped = true_coords_local.view(B_local, multiplicity, -1, 3)

            pred_token_local = torch.einsum("btn,bmnc->bmtc", t2ra_local, pred_reshaped).reshape(-1, N_local, 3)
            true_token_local = torch.einsum("btn,bmnc->bmtc", t2ra_local, true_reshaped).reshape(-1, N_local, 3)

            N_R_local = r2ra_local.shape[1]
            pred_R_local = torch.einsum("brn,bmnc->bmrc", r2ra_local, pred_reshaped).reshape(-1, N_R_local, 3)
            true_R_local = torch.einsum("brn,bmnc->bmrc", r2ra_local, true_reshaped).reshape(-1, N_R_local, 3)

            # All-gather R-set coords across cp for full column dimension
            pred_R_full = all_gather_on_cp(pred_R_local, dim=1, cp_group=cp_group, cp_size=cp_size)
            true_R_full = all_gather_on_cp(true_R_local, dim=1, cp_group=cp_group, cp_size=cp_size)

            # Compute resolved masks
            resolved_reshaped = resolved_local.view(B_local, multiplicity, -1)
            mask_row = torch.einsum("btn,bmn->bmt", t2ra_local, resolved_reshaped).reshape(-1, N_local)
            mask_col_local = torch.einsum("brn,bmn->bmr", r2ra_local, resolved_reshaped).reshape(-1, N_R_local)
            mask_col_full = all_gather_on_cp(mask_col_local, dim=1, cp_group=cp_group, cp_size=cp_size)

            # Nucleotide cutoff
            # mol_local is [B_l, N_tok_local] (token-sharded) but a2t_local is
            # [B_l, N_atom, N_tok_global] (atom_to_token is Replicate on token dim).
            # All-gather mol_type across cp so the inner dimensions match for bmm.
            mol_full = all_gather_on_cp(mol_local, dim=1, cp_group=cp_group, cp_size=cp_size)
            is_nuc = (mol_full == const.chain_type_ids["DNA"]).to(compute_dtype) + (
                mol_full == const.chain_type_ids["RNA"]
            ).to(compute_dtype)
            is_nuc_atom = torch.bmm(a2t_local, is_nuc.unsqueeze(-1)).squeeze(-1)  # [B_l, N_atom]
            is_nuc_R_local = torch.bmm(r2ra_local, is_nuc_atom.unsqueeze(-1)).squeeze(-1)  # [B_l, N_R_l]
            is_nuc_R_full = all_gather_on_cp(is_nuc_R_local, dim=1, cp_group=cp_group, cp_size=cp_size)  # [B_l, N_R]
            cutoff = 15.0 + 15.0 * is_nuc_R_full  # [B_l, N_R_full]

            # Build pair mask
            # pair_mask[i,j] = mask_row[i] * mask_col[j] * (not_same_atom)
            pair_mask = mask_row.unsqueeze(-1) * mask_col_full.unsqueeze(1)  # [B_l*m, N_l, N_R_full]

            # Self-pair masking: zero pair_mask[i, k] where local token i's
            # representative atom equals R-set entry k's representative atom.
            #
            # ``t2ra_local`` and ``r2ra_local`` have placement ``(Shard(0),
            # Shard(1))`` on a 3D tensor of shape ``[B, N, N_atom]``; dim 2
            # (the atom axis) is implicitly Replicated, so each rank holds the
            # full atom width. ``argmax(dim=-1)`` therefore returns GLOBAL atom
            # indices on both sides directly — no per-rank atom-offset
            # translation is needed.
            #
            # The all_gather concatenates per-rank R-set chunks into the full
            # ``[B_l, N_R_full]`` axis. We compare every local token's global
            # rep_atom against every R-set entry's global rep_atom (across all
            # ranks), so cross-shard self-pairs are zeroed as well.
            rep_atom_token = t2ra_local.argmax(dim=-1)  # [B_l, N_l] — global atom idx
            rep_atom_R_local = r2ra_local.argmax(dim=-1)  # [B_l, N_R_l] — global atom idx
            rep_atom_R_full = all_gather_on_cp(
                rep_atom_R_local, dim=1, cp_group=cp_group, cp_size=cp_size
            )  # [B_l, N_R_full] — global atom idx

            same_atom = rep_atom_token.unsqueeze(-1) == rep_atom_R_full.unsqueeze(1)  # [B_l, N_l, N_R_full]
            if multiplicity > 1:
                same_atom = (
                    same_atom.unsqueeze(1).expand(-1, multiplicity, -1, -1).reshape(B_local * multiplicity, N_local, -1)
                )
            pair_mask = pair_mask.masked_fill(same_atom, 0)

            # Compute cdist
            true_d = torch.cdist(true_token_local, true_R_full)  # [B_l*m, N_l, N_R_full]
            pred_d = torch.cdist(pred_token_local, pred_R_full)

            # Cutoff mask: expand cutoff to [B_l*m, N_l, N_R_full]
            cutoff_expanded = (
                cutoff.unsqueeze(1).expand(B_local, multiplicity, -1).reshape(B_local * multiplicity, 1, -1)
            )
            cutoff_expanded = cutoff_expanded.expand(-1, N_local, -1)

            # lDDT computation
            dists_to_score = (true_d < cutoff_expanded).to(compute_dtype) * pair_mask
            dist_l1 = torch.abs(true_d - pred_d)
            score = 0.25 * (
                (dist_l1 < 0.5).to(compute_dtype)
                + (dist_l1 < 1.0).to(compute_dtype)
                + (dist_l1 < 2.0).to(compute_dtype)
                + (dist_l1 < 4.0).to(compute_dtype)
            )

            # Per-token sums over the FULL R-set (R-set was all-gathered to full
            # above, so cdist column dim is N_R_full). Each rank's [B_l*m, N_l]
            # slot indexes its own LOCAL token subset along dim=1, so no
            # cross-cp reduction is required here — the R-set summation is
            # already complete per local token. (Earlier revisions all-reduced
            # these across cp, which incorrectly blended scores between the
            # different tokens that share the same N_l index across ranks.)
            out_num = (dists_to_score * score).sum(dim=-1)  # [B_l*m, N_l]
            out_denom = dists_to_score.sum(dim=-1)  # [B_l*m, N_l]
            mask_no_match = (out_denom != 0).to(compute_dtype)

            # Normalize to get target lDDT
            norm = 1.0 / (_EPS_LDDT + out_denom)
            target_lddt = norm * (_EPS_LDDT + out_num)
            combined_mask = mask_row * mask_no_match

        # Cross-entropy loss (gradient flows here)
        num_bins = pred_lddt_local.shape[-1]
        with torch.enable_grad():
            bin_index = torch.floor(target_lddt * num_bins).long().clamp(max=num_bins - 1)
            lddt_one_hot = F.one_hot(bin_index, num_classes=num_bins).to(compute_dtype)
            log_probs = F.log_softmax(pred_lddt_local.to(compute_dtype), dim=-1)
            errors = -(lddt_one_hot * log_probs).sum(dim=-1)

            masked_errors = errors * combined_mask
            numerator = masked_errors.sum(dim=-1)  # [B_l*m]
            denominator = combined_mask.sum(dim=-1)  # [B_l*m]

            num = numerator.clone()
            den = denominator.clone()
            with torch.no_grad():
                dist.all_reduce(num, op=dist.ReduceOp.SUM, group=cp_group)
                dist.all_reduce(den, op=dist.ReduceOp.SUM, group=cp_group)

            per_sample = num / den.clamp(min=_EPS_DENOM)
            loss_sum = per_sample.sum().clone()
            with torch.no_grad():
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=dp_group)

            loss_local = loss_sum / pred_lddt.shape[0]

        ctx.save_for_backward(pred_lddt_local, loss_local)
        ctx.device_mesh = device_mesh
        ctx.pred_placements = pred_lddt.placements
        ctx.pred_shape = pred_lddt.shape
        ctx.pred_stride = pred_lddt.stride()

        return DTensor.from_local(loss_local.detach(), device_mesh, _REPLICATED, shape=torch.Size(()), stride=())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_loss):
        pred_local, loss_local = ctx.saved_tensors
        if not pred_local.requires_grad:
            return (None,) * 12

        grad_local = grad_loss.to_local()
        (d_pred,) = torch.autograd.grad(
            outputs=[loss_local], inputs=[pred_local], grad_outputs=[grad_local], retain_graph=False
        )
        d_pred_dt = DTensor.from_local(
            d_pred, ctx.device_mesh, ctx.pred_placements, shape=ctx.pred_shape, stride=ctx.pred_stride
        )
        return (d_pred_dt,) + (None,) * 11


def plddt_loss_1d(
    pred_lddt: DTensor,
    pred_atom_coords: DTensor,
    true_atom_coords: DTensor,
    true_coords_resolved_mask: DTensor,
    feats: dict[str, DTensor],
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    multiplicity: int = 1,
) -> DTensor:
    """Compute pLDDT loss for 1D CP.

    Parameters
    ----------
    pred_lddt : DTensor
        [B*mult, N, bins], placements (Shard(0), Shard(1)).
    pred_atom_coords : DTensor
        [B*mult, N_atom, 3]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    true_atom_coords : DTensor
        [B*mult, N_atom, 3]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    true_coords_resolved_mask : DTensor
        [B*mult, N_atom]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    feats : dict
        Must contain token_to_rep_atom, r_set_to_rep_atom, atom_to_token, mol_type.
        ``atom_to_token`` may also arrive as either atom placement; redistributed
        internally to (Shard(0), Replicate()). ``feats`` is not mutated.

    Returns
    -------
    DTensor
        Scalar loss, placements (Replicate(), Replicate()).
    """
    # Redistribute atom-DTensor inputs to the canonical (Shard(0), Replicate())
    # before the autograd.Function boundary. The inner einsums project atoms
    # via the full N_atom axis (e.g. ``"btn,bmnc->bmtc"``), so atoms must be
    # replicated locally. ``feats["atom_to_token"]`` is read out without
    # mutating the caller's dict.
    pred_atom_coords = pred_atom_coords.redistribute(device_mesh, _ATOM_REPLICATED)
    true_atom_coords = true_atom_coords.redistribute(device_mesh, _ATOM_REPLICATED)
    true_coords_resolved_mask = true_coords_resolved_mask.redistribute(device_mesh, _ATOM_REPLICATED)
    atom_to_token = feats["atom_to_token"].redistribute(device_mesh, _ATOM_REPLICATED)
    return _PLDDTLoss1D.apply(
        pred_lddt,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats["token_to_rep_atom"],
        feats["r_set_to_rep_atom"],
        atom_to_token,
        feats["mol_type"],
        device_mesh,
        dp_group,
        cp_group,
        multiplicity,
    )


# ---------------------------------------------------------------------------
# pde_loss_1d
# ---------------------------------------------------------------------------


class _PDELoss1D(torch.autograd.Function):
    """Fused PDE loss for 1D CP.

    Computes pairwise distance error targets between predicted and true token
    coords, then cross-entropy against predicted PDE logits.

    Under 1D CP row-slab, each rank holds [B_local, N_row, N_full, bins] for
    pred_pde. Token coords are [B_local*mult, N_local, 3]. We all-gather
    token coords across cp to get the full column dimension.

    Communication budget:
      Forward:  1 all_gather (token coords + mask) + 2 all_reduce (cp + dp)
      Backward: 0 collectives
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred_pde: DTensor,
        pred_atom_coords: DTensor,
        true_atom_coords: DTensor,
        true_coords_resolved_mask: DTensor,
        token_to_rep_atom: DTensor,
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
        multiplicity: int,
        max_dist: float,
    ) -> DTensor:
        _SINGLE = (Shard(0), Shard(1))
        _ATOM = (Shard(0), Replicate())
        _validate_dtensor(pred_pde, "pred_pde", device_mesh, _SINGLE, expected_ndim=4)
        _validate_dtensor(pred_atom_coords, "pred_atom_coords", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(true_atom_coords, "true_atom_coords", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(true_coords_resolved_mask, "true_coords_resolved_mask", device_mesh, _ATOM, expected_ndim=2)
        _validate_dtensor(token_to_rep_atom, "token_to_rep_atom", device_mesh, _SINGLE, expected_ndim=3)

        cp_size = device_mesh.shape[1]
        compute_dtype = torch.promote_types(pred_pde.dtype, torch.float32)

        pred_pde_local = pred_pde.to_local().detach().requires_grad_(pred_pde.requires_grad)
        t2ra_local = token_to_rep_atom.to_local().to(compute_dtype)
        pred_coords_local = pred_atom_coords.to_local().to(compute_dtype)
        true_coords_local = true_atom_coords.to_local().to(compute_dtype)
        resolved_local = true_coords_resolved_mask.to_local().to(compute_dtype)

        B_local = t2ra_local.shape[0]
        N_local = t2ra_local.shape[1]

        with torch.no_grad():
            pred_reshaped = pred_coords_local.view(B_local, multiplicity, -1, 3)
            true_reshaped = true_coords_local.view(B_local, multiplicity, -1, 3)

            pred_token_local = torch.einsum("btn,bmnc->bmtc", t2ra_local, pred_reshaped).reshape(-1, N_local, 3)
            true_token_local = torch.einsum("btn,bmnc->bmtc", t2ra_local, true_reshaped).reshape(-1, N_local, 3)

            # All-gather token coords to get full column
            pred_token_full = all_gather_on_cp(pred_token_local, dim=1, cp_group=cp_group, cp_size=cp_size)
            true_token_full = all_gather_on_cp(true_token_local, dim=1, cp_group=cp_group, cp_size=cp_size)

            # Mask
            resolved_reshaped = resolved_local.view(B_local, multiplicity, -1)
            token_mask_local = torch.einsum("btn,bmn->bmt", t2ra_local, resolved_reshaped).reshape(-1, N_local)
            token_mask_full = all_gather_on_cp(token_mask_local, dim=1, cp_group=cp_group, cp_size=cp_size)

            mask = token_mask_local.unsqueeze(-1) * token_mask_full.unsqueeze(1)  # [B_l*m, N_l, N_full]

            # Target PDE
            true_d = torch.cdist(true_token_local, true_token_full)
            pred_d = torch.cdist(pred_token_local, pred_token_full)
            target_pde = torch.abs(true_d - pred_d)

            num_bins = pred_pde_local.shape[-1]
            bin_index = torch.floor(target_pde * num_bins / max_dist).long().clamp(max=num_bins - 1)

            # Compute mask sum for global normalization
            mask_sum_local = mask.sum(dim=(-2, -1))  # [B_l*m]
            mask_sum = mask_sum_local.clone()
            dist.all_reduce(mask_sum, op=dist.ReduceOp.SUM, group=cp_group)

        # Cross-entropy (gradient flows through pred_pde_local)
        with torch.enable_grad():
            log_probs = F.log_softmax(pred_pde_local.to(compute_dtype), dim=-1)
            target_log_prob = torch.gather(log_probs, dim=-1, index=bin_index.unsqueeze(-1)).squeeze(-1)
            errors = -target_log_prob

            masked_errors = errors * mask
            errors_sum_local = masked_errors.sum(dim=(-2, -1))  # [B_l*m]

            per_sample = errors_sum_local / (mask_sum + _EPS_DENOM)
            loss_sum = per_sample.sum().clone()
            with torch.no_grad():
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=dp_group)

            loss_local = loss_sum / pred_pde.shape[0]

        # Also compute fully reduced loss for the return value
        with torch.no_grad():
            errors_sum_global = errors_sum_local.clone()
            dist.all_reduce(errors_sum_global, op=dist.ReduceOp.SUM, group=cp_group)
            per_sample_final = errors_sum_global / (mask_sum + _EPS_DENOM)
            loss_scalar = per_sample_final.sum()
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.SUM, group=dp_group)
            loss_scalar = loss_scalar / pred_pde.shape[0]

        ctx.save_for_backward(pred_pde_local, loss_local)
        ctx.device_mesh = device_mesh
        ctx.pred_placements = pred_pde.placements
        ctx.pred_shape = pred_pde.shape
        ctx.pred_stride = pred_pde.stride()

        return DTensor.from_local(loss_scalar, device_mesh, _REPLICATED, shape=torch.Size(()), stride=())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_loss):
        pred_local, loss_local = ctx.saved_tensors
        if not pred_local.requires_grad:
            return (None,) * 10

        grad_local = grad_loss.to_local()
        (d_pred,) = torch.autograd.grad(
            outputs=[loss_local], inputs=[pred_local], grad_outputs=[grad_local], retain_graph=False
        )
        d_pred_dt = DTensor.from_local(
            d_pred, ctx.device_mesh, ctx.pred_placements, shape=ctx.pred_shape, stride=ctx.pred_stride
        )
        return (d_pred_dt,) + (None,) * 9


def pde_loss_1d(
    pred_pde: DTensor,
    pred_atom_coords: DTensor,
    true_atom_coords: DTensor,
    true_coords_resolved_mask: DTensor,
    feats: dict[str, DTensor],
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    multiplicity: int = 1,
    max_dist: float = 32.0,
) -> DTensor:
    """Compute PDE loss for 1D CP.

    Parameters
    ----------
    pred_pde : DTensor
        [B*mult, N, N, bins], placements (Shard(0), Shard(1)).
    pred_atom_coords : DTensor
        [B*mult, N_atom, 3]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    true_atom_coords : DTensor
        [B*mult, N_atom, 3]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    true_coords_resolved_mask : DTensor
        [B*mult, N_atom]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    feats : dict
        Must contain token_to_rep_atom.

    Returns
    -------
    DTensor
        Scalar loss, placements (Replicate(), Replicate()).
    """
    # Redistribute atom-DTensor inputs to the canonical (Shard(0), Replicate())
    # before the autograd.Function boundary; the inner einsums require the
    # full N_atom axis for the token projection.
    pred_atom_coords = pred_atom_coords.redistribute(device_mesh, _ATOM_REPLICATED)
    true_atom_coords = true_atom_coords.redistribute(device_mesh, _ATOM_REPLICATED)
    true_coords_resolved_mask = true_coords_resolved_mask.redistribute(device_mesh, _ATOM_REPLICATED)
    return _PDELoss1D.apply(
        pred_pde,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats["token_to_rep_atom"],
        device_mesh,
        dp_group,
        cp_group,
        multiplicity,
        max_dist,
    )


# ---------------------------------------------------------------------------
# pae_loss_1d
# ---------------------------------------------------------------------------


def _compute_frame_pred_1d(
    pred_atom_coords: DTensor,
    frames_idx_true: DTensor,
    feats: dict[str, DTensor],
    multiplicity: int,
    device_mesh: DeviceMesh,
    resolved_mask: Optional[DTensor] = None,
    inference: bool = False,
) -> tuple[DTensor, DTensor]:
    """Distributed wrapper around ``_compute_frame_pred`` for 1D CP DTensors.

    Gathers all inputs to local, runs the serial implementation, then wraps
    outputs as DTensors on the 2D mesh. ``inference=True`` matches the serial
    ``compute_ptms`` call site which derives ``resolved_pair`` from
    ``atom_pad_mask`` instead of ``atom_resolved_mask`` (see
    ``_compute_frame_pred_impl`` in ``confidencev2.py``).

    Returns
    -------
    frames_idx_pred : DTensor
        [B, mult, N, 3] with placements (Shard(0), Shard(2)).
        Note: Shard(2) shards the N_token dimension (dim 2 of 4D tensor).
    mask_collinear : DTensor
        [B, mult, N] with placements (Shard(0), Shard(2)).
    """
    feats_keys = {"asym_id", "atom_to_token", "atom_pad_mask", "atom_resolved_mask", "mol_type", "token_pad_mask"}
    for k in feats_keys:
        if k not in feats:
            raise ValueError(f"feats must contain key '{k}', got {sorted(feats.keys())}")

    if frames_idx_true.ndim == 4:
        raise ValueError(
            f"frames_idx_true has unsqueezed ensemble dim (ndim=4, shape={frames_idx_true.shape}). "
            "Only E=1 is supported; squeeze the ensemble dim before calling."
        )

    global_batch_size = feats["atom_pad_mask"].shape[0]
    _, num_tokens = feats["asym_id"].shape

    # Gather all inputs to local (replicate across cp)
    replicate_pl = (Shard(0), Replicate())

    pred_coords_gathered = pred_atom_coords.redistribute(device_mesh, placements=replicate_pl).to_local()
    frames_gathered = frames_idx_true.redistribute(device_mesh, placements=replicate_pl).to_local()

    asym_id_gathered = feats["asym_id"].redistribute(device_mesh, placements=replicate_pl).to_local()
    # feats["asym_id"] and atom_to_token may have (Shard(0), Shard(1)) in 1D CP
    # but single_repr_token_to_atom expects (Shard(0), Replicate()).  Redistribute
    # both before the call so placements match.
    asym_id_for_atom = feats["asym_id"].redistribute(device_mesh, placements=replicate_pl)
    atom_to_token_replicated = feats["atom_to_token"].redistribute(device_mesh, placements=replicate_pl)
    asym_id_atom_gathered = (
        single_repr_token_to_atom(asym_id_for_atom.float(), atom_to_token_replicated)
        .redistribute(device_mesh, placements=replicate_pl)
        .to_local()
        .to(torch.int64)
    )
    atom_pad_gathered = feats["atom_pad_mask"].redistribute(device_mesh, placements=replicate_pl).to_local()
    atom_resolved_gathered = feats["atom_resolved_mask"].redistribute(device_mesh, placements=replicate_pl).to_local()
    mol_type_gathered = feats["mol_type"].redistribute(device_mesh, placements=replicate_pl).to_local()
    token_pad_gathered = feats["token_pad_mask"].redistribute(device_mesh, placements=replicate_pl).to_local()

    resolved_gathered = None
    if resolved_mask is not None:
        resolved_gathered = resolved_mask.redistribute(device_mesh, placements=replicate_pl).to_local()

    feats_gathered = {
        "asym_id": asym_id_gathered,
        "atom_pad_mask": atom_pad_gathered,
        "atom_resolved_mask": atom_resolved_gathered,
        "mol_type": mol_type_gathered,
        "token_pad_mask": token_pad_gathered,
    }

    frames_pred_local, mask_collinear_local = _compute_frame_pred(
        pred_coords_gathered,
        frames_gathered,
        feats_gathered,
        asym_id_atom_gathered,
        multiplicity,
        resolved_mask=resolved_gathered,
        inference=inference,
    )

    # Broadcast from rank 0 to ensure consistency across cp ranks
    cp_submesh = device_mesh["cp"]
    shape_frames = torch.Size([global_batch_size, multiplicity, num_tokens, 3])
    frames_cp = distribute_tensor(frames_pred_local, cp_submesh, (Shard(2),), src_data_rank=0)
    frames_dt = DTensor.from_local(
        frames_cp.to_local(),
        device_mesh=device_mesh,
        placements=(Shard(0), Shard(2)),
        shape=shape_frames,
        stride=LayoutRightMap(shape_frames).strides,
    )

    shape_mask = torch.Size([global_batch_size, multiplicity, num_tokens])
    mask_cp = distribute_tensor(mask_collinear_local, cp_submesh, (Shard(2),), src_data_rank=0)
    mask_dt = DTensor.from_local(
        mask_cp.to_local(),
        device_mesh=device_mesh,
        placements=(Shard(0), Shard(2)),
        shape=shape_mask,
        stride=LayoutRightMap(shape_mask).strides,
    )

    return frames_dt, mask_dt


class _PAELoss1D(torch.autograd.Function):
    """Fused PAE loss for 1D CP.

    Computes frame-based PAE targets then cross-entropy against predicted
    PAE logits. Under 1D CP, pair logits [B*mult, N, N, bins] are row-slab
    sharded: each rank holds [B_l*m, N_l, N_full, bins].

    Communication budget:
      Forward:  all_gathers for frame computation + 1 all_gather (masks) +
                2 all_reduce (cp + dp)
      Backward: 0 collectives
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred_pae: DTensor,
        pred_atom_coords: DTensor,
        true_atom_coords: DTensor,
        true_coords_resolved_mask: DTensor,
        feats: dict[str, DTensor],
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
        multiplicity: int,
        max_dist: float,
    ) -> DTensor:
        _SINGLE = (Shard(0), Shard(1))
        _ATOM = (Shard(0), Replicate())
        _validate_dtensor(pred_pae, "pred_pae", device_mesh, _SINGLE, expected_ndim=4)
        _validate_dtensor(pred_atom_coords, "pred_atom_coords", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(true_atom_coords, "true_atom_coords", device_mesh, _ATOM, expected_ndim=3)
        _validate_dtensor(true_coords_resolved_mask, "true_coords_resolved_mask", device_mesh, _ATOM, expected_ndim=2)

        cp_size = device_mesh.shape[1]
        num_bins = pred_pae.shape[-1]

        pred_pae_local = pred_pae.to_local().detach()
        pred_pae_grad = pred_pae_local.unflatten(0, (-1, multiplicity)).requires_grad_(pred_pae.requires_grad)

        pred_coords_local = pred_atom_coords.to_local()
        true_coords_local = true_atom_coords.to_local()
        resolved_local = true_coords_resolved_mask.to_local()
        frame_resolved_local = feats["frame_resolved_mask"].to_local()
        token_pad_local = feats["token_pad_mask"].to_local()

        B_local_mult = pred_coords_local.shape[0]
        B_local = B_local_mult // multiplicity

        compute_dtype = torch.promote_types(pred_pae.dtype, torch.float32)

        with torch.no_grad():
            # Compute frames for true and pred coords.
            # _compute_frame_pred_1d returns DTensors with placements (Shard(0), Shard(2)),
            # sharded on the N_token dimension. The redistributes inside are on a
            # non-differentiable path (inside torch.no_grad), so the all-gathers are safe.
            # We use full_tensor() to gather to full N_token for the row-slab computation.
            frames_true_dt, mask_collinear_true_dt = _compute_frame_pred_1d(
                true_atom_coords,
                feats["frames_idx"],
                feats,
                multiplicity,
                device_mesh,
                resolved_mask=true_coords_resolved_mask,
            )
            # Gather frames across cp to get full N_token while keeping dp sharding.
            # Placements change from (Shard(0), Shard(2)) to (Shard(0), Replicate()),
            # which triggers an all-gather only on the cp dimension.
            _GATHER_PL = (Shard(0), Replicate())
            frames_true = frames_true_dt.redistribute(device_mesh, _GATHER_PL).to_local()
            # [B_l, mult, N_full, 3]
            # Use mask_collinear from _compute_frame_pred_1d (broadcast from rank 0)
            # instead of recomputing on each rank, which can produce different
            # boolean results at threshold boundaries due to FP ordering differences
            # in all-gather.
            mask_collinear_true = mask_collinear_true_dt.redistribute(device_mesh, _GATHER_PL).to_local()
            # [B_l, mult, N_full]

            frames_pred_dt, mask_collinear_pred_dt = _compute_frame_pred_1d(
                pred_atom_coords,
                feats["frames_idx"],
                feats,
                multiplicity,
                device_mesh,
            )
            frames_pred = frames_pred_dt.redistribute(device_mesh, _GATHER_PL).to_local()
            # [B_l, mult, N_full, 3]
            mask_collinear_pred = mask_collinear_pred_dt.redistribute(device_mesh, _GATHER_PL).to_local()
            # [B_l, mult, N_full]

            # Reshape atom coords (atoms are replicated, so to_local gives full atoms)
            true_reshaped = true_coords_local.reshape(B_local, multiplicity, -1, 3).to(compute_dtype)
            pred_reshaped = pred_coords_local.reshape(B_local, multiplicity, -1, 3).to(compute_dtype)

            # Express coordinates in frames — compute full N_full x N_full,
            # then extract the row-slab portion for this rank.
            true_transformed = _express_coordinate_in_frame_1d(
                true_reshaped, frames_true[:, :, :, 0], frames_true[:, :, :, 1], frames_true[:, :, :, 2]
            )  # [B_l, mult, N_full, N_full, 3]
            pred_transformed = _express_coordinate_in_frame_1d(
                pred_reshaped, frames_pred[:, :, :, 0], frames_pred[:, :, :, 1], frames_pred[:, :, :, 2]
            )

            target_pae_full = torch.sqrt(((true_transformed - pred_transformed) ** 2).sum(-1) + _EPS_PAE_DIST)
            # target_pae_full: [B_l, mult, N_full, N_full]

            # Extract row-slab portion
            cp_rank = dist.get_rank(cp_group)
            N_full = target_pae_full.shape[2]
            N_local = N_full // cp_size
            row_start = cp_rank * N_local
            row_end = row_start + N_local
            target_pae = target_pae_full[:, :, row_start:row_end, :]  # [B_l, mult, N_l, N_full]

            # mask_collinear_true and mask_collinear_pred are already computed
            # above from _compute_frame_pred_1d (broadcast from rank 0), so they
            # are consistent across CP ranks.  We still need token_pad_full for
            # the pair mask below.
            token_pad_full = all_gather_on_cp(token_pad_local, dim=1, cp_group=cp_group, cp_size=cp_size)

            # Build pair mask
            resolved_reshaped = resolved_local.reshape(B_local, multiplicity, -1)

            # Gather resolved status for frame b atoms (full N_token)
            # frames_true is [B_l, mult, N_full, 3] with global atom indices
            frame_b = frames_true[:, :, :, 1]  # [B_l, mult, N_full]
            b_resolved = torch.gather(resolved_reshaped.expand(-1, multiplicity, -1), dim=2, index=frame_b.long())
            # b_resolved: [B_l, mult, N_full]

            # frame_resolved: [B_l, N_local] -> all-gather to [B_l, N_full]
            frame_resolved_full = all_gather_on_cp(frame_resolved_local, dim=1, cp_group=cp_group, cp_size=cp_size)

            # Row-slab slice of masks: rows are local, columns are full
            mask_collinear_true_row = mask_collinear_true[:, :, row_start:row_end]
            mask_collinear_pred_row = mask_collinear_pred[:, :, row_start:row_end]
            frame_resolved_row = frame_resolved_full[:, row_start:row_end]
            token_pad_row = token_pad_full[:, row_start:row_end]

            pair_mask = (
                frame_resolved_row[:, None, :, None]
                * mask_collinear_true_row[:, :, :, None]
                * mask_collinear_pred_row[:, :, :, None]
                * b_resolved[:, :, None, :]
                * token_pad_row[:, None, :, None]
                * token_pad_full[:, None, None, :]
            )  # [B_l, mult, N_l, N_full]

            bin_index = torch.floor(target_pae * num_bins / max_dist).long().clamp(max=num_bins - 1)

            # Global mask sum
            mask_sum_local = pair_mask.sum(dim=(-2, -1))  # [B_l, mult]
            mask_sum = mask_sum_local.clone()
            dist.all_reduce(mask_sum, op=dist.ReduceOp.SUM, group=cp_group)

        # Cross-entropy (gradient through pred_pae)
        with torch.enable_grad():
            log_probs = F.log_softmax(pred_pae_grad.to(compute_dtype), dim=-1)
            target_log_prob = torch.gather(log_probs, dim=-1, index=bin_index.unsqueeze(-1)).squeeze(-1)
            errors = -target_log_prob

            masked_errors = errors * pair_mask
            errors_sum_local = masked_errors.sum(dim=(-2, -1))  # [B_l, mult]

            per_sample = errors_sum_local / (mask_sum + _EPS_DENOM)
            loss_sum = per_sample.sum().clone()
            with torch.no_grad():
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=dp_group)

            B_global_mult = pred_pae.shape[0]
            loss_local = loss_sum / B_global_mult

        # Compute fully reduced loss for return
        with torch.no_grad():
            errors_sum_global = errors_sum_local.clone()
            dist.all_reduce(errors_sum_global, op=dist.ReduceOp.SUM, group=cp_group)
            per_sample_final = errors_sum_global / (mask_sum + _EPS_DENOM)
            loss_scalar = per_sample_final.sum()
            dist.all_reduce(loss_scalar, op=dist.ReduceOp.SUM, group=dp_group)
            loss_scalar = loss_scalar / B_global_mult

        ctx.save_for_backward(pred_pae_grad, loss_local)
        ctx.device_mesh = device_mesh
        ctx.pred_placements = pred_pae.placements
        ctx.pred_shape = pred_pae.shape
        ctx.pred_stride = pred_pae.stride()
        ctx.multiplicity = multiplicity

        return DTensor.from_local(loss_scalar, device_mesh, _REPLICATED, shape=torch.Size(()), stride=())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_loss):
        pred_pae_grad, loss_local = ctx.saved_tensors
        if not pred_pae_grad.requires_grad:
            return (None,) * 10

        grad_local = grad_loss.to_local()
        (d_pred,) = torch.autograd.grad(
            outputs=[loss_local], inputs=[pred_pae_grad], grad_outputs=[grad_local], retain_graph=False
        )
        # Flatten (B_batch, mult, ...) back to (B_batch*mult, ...)
        d_pred_flat = d_pred.flatten(0, 1)
        d_pred_dt = DTensor.from_local(
            d_pred_flat, ctx.device_mesh, ctx.pred_placements, shape=ctx.pred_shape, stride=ctx.pred_stride
        )
        return (d_pred_dt,) + (None,) * 9


def _express_coordinate_in_frame_1d(
    atom_coords: torch.Tensor,
    frame_a: torch.Tensor,
    frame_b: torch.Tensor,
    frame_c: torch.Tensor,
) -> torch.Tensor:
    """Express coordinates in local frames.

    Parameters
    ----------
    atom_coords : Tensor
        [B, mult, N_atom, 3]
    frame_a, frame_b, frame_c : Tensor
        [B, mult, N_token] — global atom indices for frame atoms.

    Returns
    -------
    Tensor
        [B, mult, N_token, N_token, 3] — transformed coordinates.
    """
    batch, multiplicity = atom_coords.shape[0], atom_coords.shape[1]
    batch_idx = torch.arange(batch, device=atom_coords.device)[:, None, None]
    mult_idx = torch.arange(multiplicity, device=atom_coords.device)[None, :, None]

    a = atom_coords[batch_idx, mult_idx, frame_a]
    b = atom_coords[batch_idx, mult_idx, frame_b]
    c = atom_coords[batch_idx, mult_idx, frame_c]

    w1 = (a - b) / (torch.norm(a - b, dim=-1, keepdim=True) + _EPS_FRAME_NORM)
    w2 = (c - b) / (torch.norm(c - b, dim=-1, keepdim=True) + _EPS_FRAME_NORM)
    e1 = (w1 + w2) / (torch.norm(w1 + w2, dim=-1, keepdim=True) + _EPS_FRAME_NORM)
    e2 = (w2 - w1) / (torch.norm(w2 - w1, dim=-1, keepdim=True) + _EPS_FRAME_NORM)
    e3 = torch.linalg.cross(e1, e2)

    # d[i,j] = b[j] - b[i]
    d = b[:, :, None, :, :] - b[:, :, :, None, :]
    x_transformed = torch.cat(
        [
            torch.sum(d * e1[:, :, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e2[:, :, :, None, :], dim=-1, keepdim=True),
            torch.sum(d * e3[:, :, :, None, :], dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return x_transformed


def pae_loss_1d(
    pred_pae: DTensor,
    pred_atom_coords: DTensor,
    true_atom_coords: DTensor,
    true_coords_resolved_mask: DTensor,
    feats: dict[str, DTensor],
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    multiplicity: int = 1,
    max_dist: float = 32.0,
) -> DTensor:
    """Compute PAE loss for 1D CP.

    Parameters
    ----------
    pred_pae : DTensor
        [B*mult, N, N, bins], placements (Shard(0), Shard(1)).
    pred_atom_coords : DTensor
        [B*mult, N_atom, 3]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    true_atom_coords : DTensor
        [B*mult, N_atom, 3]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    true_coords_resolved_mask : DTensor
        [B*mult, N_atom]. Atoms may arrive as (Shard(0), Replicate()) OR
        (Shard(0), Shard(1)); redistributed internally to (Shard(0), Replicate()).
    feats : dict
        Must contain frames_idx, frame_resolved_mask, token_pad_mask, and
        keys required by _compute_frame_pred_1d.

    Returns
    -------
    DTensor
        Scalar loss, placements (Replicate(), Replicate()).
    """
    # Redistribute atom-DTensor inputs to the canonical (Shard(0), Replicate())
    # before the autograd.Function boundary. _express_coordinate_in_frame_1d
    # indexes atom_coords by global frame_a/b/c atom indices, so atoms must
    # be replicated locally.
    pred_atom_coords = pred_atom_coords.redistribute(device_mesh, _ATOM_REPLICATED)
    true_atom_coords = true_atom_coords.redistribute(device_mesh, _ATOM_REPLICATED)
    true_coords_resolved_mask = true_coords_resolved_mask.redistribute(device_mesh, _ATOM_REPLICATED)
    return _PAELoss1D.apply(
        pred_pae,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats,
        device_mesh,
        dp_group,
        cp_group,
        multiplicity,
        max_dist,
    )


# ---------------------------------------------------------------------------
# confidence_loss_1d — top-level aggregator
# ---------------------------------------------------------------------------


def confidence_loss_1d(
    model_out: dict[str, DTensor],
    feats: dict[str, DTensor],
    true_coords: DTensor,
    true_coords_resolved_mask: DTensor,
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    token_level_confidence: bool = True,
    multiplicity: int = 1,
    alpha_pae: float = 0.0,
) -> dict[str, DTensor | dict[str, DTensor]]:
    """Compute confidence loss for 1D CP.

    Aggregates plddt, pde, resolved, and (optionally) pae losses.
    This is the 1D CP equivalent of
    ``boltz.distributed.model.loss.confidencev2.confidence_loss``.

    Parameters
    ----------
    model_out : dict[str, DTensor]
        Model outputs:
        - "plddt_logits": [B*mult, N, bins], (Shard(0), Shard(1))
        - "pde_logits": [B*mult, N, N, bins], (Shard(0), Shard(1))
        - "resolved_logits": [B*mult, N, 2], (Shard(0), Shard(1))
        - "sample_atom_coords": [B*mult, N_atom, 3], (Shard(0), Replicate())
        - "pae_logits" (when alpha_pae > 0): [B*mult, N, N, bins], (Shard(0), Shard(1))
    feats : dict[str, DTensor]
        Input features.
    true_coords : DTensor
        [B*mult, N_atom, 3], (Shard(0), Replicate()).
    true_coords_resolved_mask : DTensor
        [B*mult, N_atom], (Shard(0), Replicate()).
    device_mesh : DeviceMesh
        2D mesh (dp, cp).
    dp_group, cp_group : ProcessGroup
    token_level_confidence : bool
        Must be True. The atom-level path is not implemented for DTensor.
    multiplicity : int
    alpha_pae : float

    Returns
    -------
    dict
        {"loss": scalar DTensor, "loss_breakdown": {sub-loss DTensors}}
    """
    if not token_level_confidence:
        raise NotImplementedError(
            "confidence_loss_1d only supports token_level_confidence=True. "
            "The atom-level confidence path is not implemented for DTensor."
        )

    from boltz.distributed.model.layers.elementwise_op import (
        ElementwiseOp,
        elementwise_op,
        scalar_tensor_op,
    )

    plddt = plddt_loss_1d(
        pred_lddt=model_out["plddt_logits"],
        pred_atom_coords=model_out["sample_atom_coords"],
        true_atom_coords=true_coords,
        true_coords_resolved_mask=true_coords_resolved_mask,
        feats=feats,
        device_mesh=device_mesh,
        dp_group=dp_group,
        cp_group=cp_group,
        multiplicity=multiplicity,
    )

    pde = pde_loss_1d(
        pred_pde=model_out["pde_logits"],
        pred_atom_coords=model_out["sample_atom_coords"],
        true_atom_coords=true_coords,
        true_coords_resolved_mask=true_coords_resolved_mask,
        feats=feats,
        device_mesh=device_mesh,
        dp_group=dp_group,
        cp_group=cp_group,
        multiplicity=multiplicity,
    )

    resolved = resolved_loss_1d(
        pred_resolved=model_out["resolved_logits"],
        feats=feats,
        true_coords_resolved_mask=true_coords_resolved_mask,
        device_mesh=device_mesh,
        dp_group=dp_group,
        cp_group=cp_group,
        multiplicity=multiplicity,
    )

    if alpha_pae > 0.0:
        pae = pae_loss_1d(
            pred_pae=model_out["pae_logits"],
            pred_atom_coords=model_out["sample_atom_coords"],
            true_atom_coords=true_coords,
            true_coords_resolved_mask=true_coords_resolved_mask,
            feats=feats,
            device_mesh=device_mesh,
            dp_group=dp_group,
            cp_group=cp_group,
            multiplicity=multiplicity,
        )
    else:
        pae = DTensor.from_local(
            torch.tensor(0.0, device=model_out["plddt_logits"].device),
            device_mesh=device_mesh,
            placements=_REPLICATED,
            shape=torch.Size(()),
            stride=(),
        )

    # Aggregate: loss = plddt + pde + resolved + alpha_pae * pae
    loss = elementwise_op(plddt, pde, ElementwiseOp.SUM)
    loss = elementwise_op(loss, resolved, ElementwiseOp.SUM)
    if alpha_pae > 0.0:
        pae_scaled = scalar_tensor_op(alpha_pae, pae, ElementwiseOp.PROD)
        loss = elementwise_op(loss, pae_scaled, ElementwiseOp.SUM)

    return {
        "loss": loss,
        "loss_breakdown": {
            "plddt_loss": plddt,
            "pde_loss": pde,
            "resolved_loss": resolved,
            "pae_loss": pae,
        },
    }
