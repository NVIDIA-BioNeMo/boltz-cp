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

"""DTensor-based context-parallel smooth LDDT loss for 1D CP (2D mesh ``(dp, cp)``).

Adapts the 2D CP smooth LDDT loss (diffusion.py) for 1D CP where:
- Mesh is 2D ``(dp, cp)`` instead of 3D ``(dp, cp_axis_0, cp_axis_1)``
- Atom coords ``[B*M, N, 3]`` use placement ``(Shard(0), Shard(1))`` --
  each rank holds ``[B_local*M, N/cp, 3]``
- Masks ``[B, N]`` use placement ``(Shard(0), Shard(1))``

Since each rank holds a row-slab of atoms, computing pairwise distances
requires an all-gather of coordinates across cp to obtain the full
coordinate set. The Triton kernel then operates on local rows vs full
columns: ``pred_coords_local [B*M, N/cp, 3]`` vs
``pred_coords_full [B*M, N, 3]``.

Communication budget:
  Forward (5 collective calls):
    1. all_gather over cp for pred_coords (1 call)
    2. all_gather over cp for true_coords (1 call)
    3. all_gather over cp for coords_mask (1 call)
    4. all_reduce over cp for num + den (1 call, packed)
    5. all_reduce over dp for batch mean (1 call)
  Backward (4 collective calls):
    1. all_gather over cp for pred_coords (recompute, 1 call)
    2. all_gather over cp for true_coords (recompute, 1 call)
    3. all_gather over cp for coords_mask (recompute, 1 call)
    4. reduce_scatter over cp for grad_pred_full -> grad_pred_local (1 call)

Memory per rank: O(N^2/cp) for the pairwise distance computation,
matching the forward per-rank budget.

Equivalence to serial code (src/boltz/model/loss/diffusion.py smooth_lddt_loss):
  The Triton kernel computes the same smooth LDDT formula. The all-gather
  reconstructs the full coordinate set so each rank computes its row-slab
  of the full pairwise distance matrix, then num/den are reduced across cp.
"""

import torch
import torch.distributed as dist
from torch.autograd.function import FunctionCtx
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh

try:
    from boltz.distributed.model.loss.triton.smooth_lddt_loss import (
        grid_launch_config,
        smooth_lddt_loss_bwd_kernel,
        smooth_lddt_loss_fwd_kernel,
    )

    has_smooth_lddt_loss_triton_kernels = True
except ImportError:
    has_smooth_lddt_loss_triton_kernels = False


def _all_gather_coords_1d(
    coords_local: torch.Tensor,
    cp_group: ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """All-gather coordinates across cp to reconstruct the full atom set.

    Parameters
    ----------
    coords_local : Tensor
        Local coordinate shard [B, N_local, 3].
    cp_group : ProcessGroup
        Process group for the cp mesh dimension.
    cp_size : int
        Number of ranks in the cp group.

    Returns
    -------
    Tensor
        Full coordinates [B, N_full, 3].
    """
    if cp_size == 1:
        return coords_local
    gathered = [torch.empty_like(coords_local) for _ in range(cp_size)]
    dist.all_gather(gathered, coords_local.contiguous(), group=cp_group)
    return torch.cat(gathered, dim=1)  # [B, N_full, 3]


def _smooth_lddt_forward_local(
    pred_coords_local: torch.Tensor,
    true_coords_local: torch.Tensor,
    pred_coords_full: torch.Tensor,
    true_coords_full: torch.Tensor,
    is_nucleotide_local: torch.Tensor,
    coords_mask_local: torch.Tensor,
    coords_mask_full: torch.Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
    multiplicity: int,
    cp_rank: int,
    n_local: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local forward computation for smooth LDDT loss under 1D CP.

    Computes the local numerator and denominator contributions from the
    row-slab [B*M, N/cp] vs the full column set [B*M, N].

    Parameters
    ----------
    pred_coords_local : Tensor
        Local predicted coords [B*M, N_local, 3].
    true_coords_local : Tensor
        Local true coords [B*M, N_local, 3].
    pred_coords_full : Tensor
        Full predicted coords [B*M, N_full, 3].
    true_coords_full : Tensor
        Full true coords [B*M, N_full, 3].
    is_nucleotide_local : Tensor
        Nucleotide flags [B, N_local] (pre-multiplicity).
    coords_mask_local : Tensor
        Atom mask [B, N_local] (pre-multiplicity).
    coords_mask_full : Tensor
        Full atom mask [B, N_full] (pre-multiplicity).
    nucleic_acid_cutoff : float
        Distance cutoff for nucleic acid atoms.
    other_cutoff : float
        Distance cutoff for non-nucleic acid atoms.
    multiplicity : int
        Number of diffusion samples per conformer.
    cp_rank : int
        Rank within the cp group (for diagonal zeroing).
    n_local : int
        Number of local atoms per rank (N/cp).

    Returns
    -------
    tuple[Tensor, Tensor]
        num_local [B*M], den_local [B*M].
    """
    dtype = true_coords_local.dtype

    # Expand masks for multiplicity: (B, N) -> (B*M, N)
    is_nucleotide_local = is_nucleotide_local.repeat_interleave(multiplicity, dim=0)
    coords_mask_local = coords_mask_local.repeat_interleave(multiplicity, dim=0)
    coords_mask_full = coords_mask_full.repeat_interleave(multiplicity, dim=0)

    n_full = pred_coords_full.shape[1]

    # Pairwise nucleotide mask: [B*M, N_local, N_full]
    is_nucleotide_pair = is_nucleotide_local.unsqueeze(-1).expand(-1, -1, n_full)

    # True pairwise distances: [B*M, N_local, N_full]
    true_dists = torch.cdist(true_coords_local, true_coords_full)

    # Cutoff mask
    mask = torch.where(
        is_nucleotide_pair.bool(),
        (true_dists < nucleic_acid_cutoff),
        (true_dists < other_cutoff),
    ).to(dtype=dtype)

    # Zero the diagonal on the self-block
    local_num_samples = pred_coords_local.shape[0]
    diag_offset = cp_rank * n_local
    row_idx = torch.arange(n_local, device=pred_coords_local.device)
    col_idx = row_idx + diag_offset
    # Build diag mask: 1 everywhere except the diagonal of the self-block
    diag_mask = torch.ones(n_local, n_full, device=pred_coords_local.device, dtype=dtype)
    diag_mask[row_idx, col_idx] = 0.0
    diag_mask = diag_mask.unsqueeze(0).expand(local_num_samples, -1, -1)
    mask = mask * diag_mask

    # Apply coordinate masks
    mask = mask * coords_mask_local.unsqueeze(-1)
    mask = mask * coords_mask_full.unsqueeze(1)

    # Predicted pairwise distances: [B*M, N_local, N_full]
    pred_dists = torch.cdist(pred_coords_local, pred_coords_full)
    dist_diff = (true_dists - pred_dists).abs()

    # Epsilon (smooth LDDT scoring)
    eps = torch.sigmoid(0.5 - dist_diff)
    for cutoff in (1.0, 2.0, 4.0):
        eps = eps + torch.sigmoid(cutoff - dist_diff)
    eps *= 0.25

    # Reduce over spatial dims
    num_local = (eps * mask).sum(dim=(1, 2))  # [B*M]
    den_local = mask.sum(dim=(1, 2))  # [B*M]

    return num_local, den_local


def _smooth_lddt_backward_local(
    grad_num_reduced: torch.Tensor,
    pred_coords_local: torch.Tensor,
    true_coords_local: torch.Tensor,
    pred_coords_full: torch.Tensor,
    true_coords_full: torch.Tensor,
    is_nucleotide_local: torch.Tensor,
    coords_mask_local: torch.Tensor,
    coords_mask_full: torch.Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
    multiplicity: int,
    cp_rank: int,
    n_local: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local backward computation for smooth LDDT loss under 1D CP.

    Recomputes the forward pass intermediates and computes gradients
    w.r.t. pred_coords_local and pred_coords_full.

    Parameters
    ----------
    grad_num_reduced : Tensor
        Gradient w.r.t. num_reduced [B*M].
    pred_coords_local : Tensor
        Local predicted coords [B*M, N_local, 3].
    true_coords_local : Tensor
        Local true coords [B*M, N_local, 3].
    pred_coords_full : Tensor
        Full predicted coords [B*M, N_full, 3].
    true_coords_full : Tensor
        Full true coords [B*M, N_full, 3].
    is_nucleotide_local : Tensor
        Nucleotide flags [B, N_local] (pre-multiplicity).
    coords_mask_local : Tensor
        Atom mask [B, N_local] (pre-multiplicity).
    coords_mask_full : Tensor
        Full atom mask [B, N_full] (pre-multiplicity).
    nucleic_acid_cutoff : float
        Distance cutoff for nucleic acid atoms.
    other_cutoff : float
        Distance cutoff for non-nucleic acid atoms.
    multiplicity : int
        Number of diffusion samples per conformer.
    cp_rank : int
        Rank within the cp group (for diagonal zeroing).
    n_local : int
        Number of local atoms per rank (N/cp).

    Returns
    -------
    tuple[Tensor, Tensor]
        grad_pred_local [B*M, N_local, 3], grad_pred_full [B*M, N_full, 3].
    """
    dtype = true_coords_local.dtype

    # Recompute masks (same as forward)
    is_nucleotide_local = is_nucleotide_local.repeat_interleave(multiplicity, dim=0)
    coords_mask_local = coords_mask_local.repeat_interleave(multiplicity, dim=0)
    coords_mask_full = coords_mask_full.repeat_interleave(multiplicity, dim=0)

    n_full = pred_coords_full.shape[1]
    is_nucleotide_pair = is_nucleotide_local.unsqueeze(-1).expand(-1, -1, n_full)

    true_dists = torch.cdist(true_coords_local, true_coords_full)

    mask = torch.where(
        is_nucleotide_pair.bool(),
        (true_dists < nucleic_acid_cutoff),
        (true_dists < other_cutoff),
    ).to(dtype=dtype)

    # Zero diagonal on self-block
    local_num_samples = pred_coords_local.shape[0]
    diag_offset = cp_rank * n_local
    row_idx = torch.arange(n_local, device=pred_coords_local.device)
    col_idx = row_idx + diag_offset
    diag_mask = torch.ones(n_local, n_full, device=pred_coords_local.device, dtype=dtype)
    diag_mask[row_idx, col_idx] = 0.0
    diag_mask = diag_mask.unsqueeze(0).expand(local_num_samples, -1, -1)
    mask = mask * diag_mask

    mask = mask * coords_mask_local.unsqueeze(-1)
    mask = mask * coords_mask_full.unsqueeze(1)

    # Recompute pred diffs and distances
    diff_vec = pred_coords_local.unsqueeze(2) - pred_coords_full.unsqueeze(1)  # [B*M, N_local, N_full, 3]
    pred_dists = diff_vec.norm(dim=-1)  # [B*M, N_local, N_full]
    dist_diff = (true_dists - pred_dists).abs()

    # d_eps / d(|pred-true|)
    d_eps_d_diff = torch.zeros_like(dist_diff)
    for cutoff in (0.5, 1.0, 2.0, 4.0):
        val = cutoff - dist_diff
        sig = torch.sigmoid(val)
        d_eps_d_diff -= sig * (1 - sig)
    d_eps_d_diff *= 0.25

    # d_L / d(pred_dists)
    grad_num_broadcast = grad_num_reduced.view(-1, 1, 1)
    d_L_d_pred_dists = grad_num_broadcast * mask * d_eps_d_diff * torch.sign(pred_dists - true_dists)

    # Gradient through distance computation
    pred_dists_safe = pred_dists.unsqueeze(-1) + 1e-8
    diff_dir = diff_vec / pred_dists_safe

    d_L_d_diff_vec = d_L_d_pred_dists.unsqueeze(-1) * diff_dir  # [B*M, N_local, N_full, 3]

    # grad_pred_local: sum over N_full (columns)
    grad_pred_local = d_L_d_diff_vec.sum(dim=2)  # [B*M, N_local, 3]
    # grad_pred_full: -sum over N_local (rows)
    grad_pred_full = -d_L_d_diff_vec.sum(dim=1)  # [B*M, N_full, 3]

    return grad_pred_local, grad_pred_full


def _smooth_lddt_triton_forward_local(
    pred_coords_local: torch.Tensor,
    true_coords_local: torch.Tensor,
    pred_coords_full: torch.Tensor,
    true_coords_full: torch.Tensor,
    is_nucleotide_local: torch.Tensor,
    coords_mask_local: torch.Tensor,
    coords_mask_full: torch.Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
    multiplicity: int,
    cp_rank: int,
    n_local: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-based local forward for smooth LDDT loss under 1D CP.

    Delegates to the shared Triton kernel that takes separate row/column
    coordinate tensors. For 1D CP, the row tensor is the local shard and
    the column tensor is the all-gathered full set.

    Parameters
    ----------
    pred_coords_local : Tensor
        Local predicted coords [B*M, N_local, 3].
    true_coords_local : Tensor
        Local true coords [B*M, N_local, 3].
    pred_coords_full : Tensor
        Full predicted coords [B*M, N_full, 3].
    true_coords_full : Tensor
        Full true coords [B*M, N_full, 3].
    is_nucleotide_local : Tensor
        Nucleotide flags [B, N_local] (pre-multiplicity).
    coords_mask_local : Tensor
        Atom mask [B, N_local] (pre-multiplicity).
    coords_mask_full : Tensor
        Full atom mask [B, N_full] (pre-multiplicity).
    nucleic_acid_cutoff : float
        Distance cutoff for nucleic acid atoms.
    other_cutoff : float
        Distance cutoff for non-nucleic acid atoms.
    multiplicity : int
        Number of diffusion samples per conformer.
    cp_rank : int
        Rank within the cp group (for diagonal zeroing).
    n_local : int
        Number of local atoms per rank (N/cp).

    Returns
    -------
    tuple[Tensor, Tensor]
        num_local [B*M], den_local [B*M].
    """
    if not has_smooth_lddt_loss_triton_kernels:
        raise ImportError("Smooth LDDT loss Triton kernels are not available.")

    if pred_coords_local.dtype in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Triton kernel for smooth LDDT loss does not support {pred_coords_local.dtype} "
            "due to precision issues. Please use float32."
        )

    # The Triton kernel assumes square [N, N] blocks and uses is_self_comm
    # to zero the diagonal (offs_m == offs_n). Passing the full [N_local, N_full]
    # pair directly would mis-identify the diagonal for cp_rank > 0.
    # Instead, decompose into cp_size square [N_local, N_local] blocks,
    # setting is_self_comm=True only for the self-block (k == cp_rank).
    # Total work is still O(N^2/cp).

    B_M = pred_coords_local.shape[0]
    n_full = pred_coords_full.shape[1]
    cp_size = n_full // n_local

    num_output = torch.zeros(B_M, device=pred_coords_local.device, dtype=pred_coords_local.dtype)
    den_output = torch.zeros(B_M, device=pred_coords_local.device, dtype=pred_coords_local.dtype)

    is_nucleotide_int8 = is_nucleotide_local.to(dtype=torch.int8)

    for k in range(cp_size):
        col_start = k * n_local
        col_end = col_start + n_local
        pred_col = pred_coords_full[:, col_start:col_end, :].contiguous()
        true_col = true_coords_full[:, col_start:col_end, :].contiguous()
        mask_col = coords_mask_full[:, col_start:col_end].contiguous()
        is_self = k == cp_rank

        num_block = torch.zeros(B_M, device=pred_coords_local.device, dtype=pred_coords_local.dtype)
        den_block = torch.zeros(B_M, device=pred_coords_local.device, dtype=pred_coords_local.dtype)

        smooth_lddt_loss_fwd_kernel[grid_launch_config](
            pred_coords_local,
            true_coords_local,
            pred_col,
            true_col,
            is_nucleotide_int8,
            coords_mask_local,
            mask_col,
            num_block,
            den_block,
            pred_coords_local.stride(0),
            pred_coords_local.stride(1),
            pred_coords_local.stride(2),
            true_coords_local.stride(0),
            true_coords_local.stride(1),
            true_coords_local.stride(2),
            pred_col.stride(0),
            pred_col.stride(1),
            pred_col.stride(2),
            true_col.stride(0),
            true_col.stride(1),
            true_col.stride(2),
            is_nucleotide_int8.stride(0),
            is_nucleotide_int8.stride(1),
            coords_mask_local.stride(0),
            coords_mask_local.stride(1),
            mask_col.stride(0),
            mask_col.stride(1),
            nucleic_acid_cutoff,
            other_cutoff,
            is_self,
            pred_coords_local.shape[0],
            pred_coords_local.shape[1],
            coords_mask_local.shape[0],
        )

        num_output += num_block
        den_output += den_block

    return num_output, den_output


def _smooth_lddt_triton_backward_local(
    grad_num_reduced: torch.Tensor,
    pred_coords_local: torch.Tensor,
    true_coords_local: torch.Tensor,
    pred_coords_full: torch.Tensor,
    true_coords_full: torch.Tensor,
    is_nucleotide_local: torch.Tensor,
    coords_mask_local: torch.Tensor,
    coords_mask_full: torch.Tensor,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
    multiplicity: int,
    cp_rank: int,
    n_local: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-based local backward for smooth LDDT loss under 1D CP.

    Decomposes the backward into cp_size blocks matching the forward
    decomposition. For each block, calls the Triton backward kernel
    and accumulates gradients.

    Parameters
    ----------
    grad_num_reduced : Tensor
        Gradient w.r.t. num_reduced [B*M].
    pred_coords_local : Tensor
        Local predicted coords [B*M, N_local, 3].
    true_coords_local : Tensor
        Local true coords [B*M, N_local, 3].
    pred_coords_full : Tensor
        Full predicted coords [B*M, N_full, 3].
    true_coords_full : Tensor
        Full true coords [B*M, N_full, 3].
    is_nucleotide_local : Tensor
        Nucleotide flags [B, N_local] (pre-multiplicity).
    coords_mask_local : Tensor
        Atom mask [B, N_local] (pre-multiplicity).
    coords_mask_full : Tensor
        Full atom mask [B, N_full] (pre-multiplicity).
    nucleic_acid_cutoff : float
        Distance cutoff for nucleic acid atoms.
    other_cutoff : float
        Distance cutoff for non-nucleic acid atoms.
    multiplicity : int
        Number of diffusion samples per conformer.
    cp_rank : int
        Rank within the cp group.
    n_local : int
        Number of local atoms per rank (N/cp).

    Returns
    -------
    tuple[Tensor, Tensor]
        grad_pred_local [B*M, N_local, 3], grad_pred_full [B*M, N_full, 3].
    """
    if not has_smooth_lddt_loss_triton_kernels:
        raise ImportError("Smooth LDDT loss Triton kernels are not available.")

    if pred_coords_local.dtype in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"Triton kernel for smooth LDDT loss does not support {pred_coords_local.dtype} "
            "due to precision issues. Please use float32."
        )

    B_M = pred_coords_local.shape[0]
    n_full = pred_coords_full.shape[1]
    cp_size = n_full // n_local

    grad_pred_local = torch.zeros_like(pred_coords_local)
    grad_pred_full = torch.zeros_like(pred_coords_full)

    # The Triton bwd kernel expects grad_den_reduced but we pass zeros
    # because den has no gradient path to pred_coords (mask is detached).
    grad_den_zeros = torch.zeros_like(grad_num_reduced)

    is_nucleotide_int8 = is_nucleotide_local.to(dtype=torch.int8)

    for k in range(cp_size):
        col_start = k * n_local
        col_end = col_start + n_local
        pred_col = pred_coords_full[:, col_start:col_end, :].contiguous()
        true_col = true_coords_full[:, col_start:col_end, :].contiguous()
        mask_col = coords_mask_full[:, col_start:col_end].contiguous()
        is_self = k == cp_rank

        grad_pred_block = torch.zeros_like(pred_coords_local)
        grad_col_block = torch.zeros(B_M, n_local, 3, device=pred_coords_local.device, dtype=pred_coords_local.dtype)

        smooth_lddt_loss_bwd_kernel[grid_launch_config](
            grad_num_reduced,
            grad_den_zeros,
            pred_coords_local,
            true_coords_local,
            pred_col,
            true_col,
            is_nucleotide_int8,
            coords_mask_local,
            mask_col,
            grad_pred_block,
            grad_col_block,
            pred_coords_local.stride(0),
            pred_coords_local.stride(1),
            pred_coords_local.stride(2),
            true_coords_local.stride(0),
            true_coords_local.stride(1),
            true_coords_local.stride(2),
            pred_col.stride(0),
            pred_col.stride(1),
            pred_col.stride(2),
            true_col.stride(0),
            true_col.stride(1),
            true_col.stride(2),
            is_nucleotide_int8.stride(0),
            is_nucleotide_int8.stride(1),
            coords_mask_local.stride(0),
            coords_mask_local.stride(1),
            mask_col.stride(0),
            mask_col.stride(1),
            grad_pred_block.stride(0),
            grad_pred_block.stride(1),
            grad_pred_block.stride(2),
            grad_col_block.stride(0),
            grad_col_block.stride(1),
            grad_col_block.stride(2),
            nucleic_acid_cutoff,
            other_cutoff,
            is_self,
            pred_coords_local.shape[0],
            pred_coords_local.shape[1],
            coords_mask_local.shape[0],
        )

        grad_pred_local += grad_pred_block
        grad_pred_full[:, col_start:col_end, :] += grad_col_block

    return grad_pred_local, grad_pred_full


class _SmoothLDDTLoss1DCP(torch.autograd.Function):
    """Fused autograd.Function for smooth LDDT loss under 1D CP.

    Forward: to_local() -> all_gather coords -> local Triton/PyTorch computation
             -> all_reduce num/den -> scalar loss -> from_local()
    Backward: all_gather coords -> local gradient computation
              -> reduce_scatter grad_pred_full -> sum with grad_pred_local -> from_local()

    Input placements (2D mesh ``(dp, cp)``):
      pred_coords:  (Shard(0), Shard(1))  -- [B*M, N, 3]
      true_coords:  (Shard(0), Shard(1))  -- [B*M, N, 3]
      is_nucleotide: (Shard(0), Shard(1)) -- [B, N]
      coords_mask:  (Shard(0), Shard(1))  -- [B, N]

    Output placements:
      loss: (Replicate(), Replicate()) -- scalar
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: FunctionCtx,
        pred_coords: DTensor,
        true_coords: DTensor,
        is_nucleotide: DTensor,
        coords_mask: DTensor,
        device_mesh: DeviceMesh,
        dp_group: ProcessGroup,
        cp_group: ProcessGroup,
        nucleic_acid_cutoff: float,
        other_cutoff: float,
        multiplicity: int,
        use_triton: bool,
    ) -> DTensor:
        """Forward pass.

        Parameters
        ----------
        pred_coords : DTensor
            Predicted atom coordinates [B*M, N, 3], placements (Shard(0), Shard(1)).
        true_coords : DTensor
            True atom coordinates [B*M, N, 3], placements (Shard(0), Shard(1)).
        is_nucleotide : DTensor
            Nucleotide flags [B, N], placements (Shard(0), Shard(1)).
        coords_mask : DTensor
            Atom mask [B, N], placements (Shard(0), Shard(1)).
        device_mesh : DeviceMesh
            2D device mesh (dp, cp).
        dp_group : ProcessGroup
            Process group for dp mesh dimension.
        cp_group : ProcessGroup
            Process group for cp mesh dimension.
        nucleic_acid_cutoff : float
            Distance cutoff for nucleic acid atoms.
        other_cutoff : float
            Distance cutoff for non-nucleic acid atoms.
        multiplicity : int
            Number of diffusion samples per conformer.
        use_triton : bool
            Whether to use Triton kernels.

        Returns
        -------
        DTensor
            Scalar loss, placements (Replicate(), Replicate()).
        """
        # --- Validate inputs ---
        for name, dt in [
            ("pred_coords", pred_coords),
            ("true_coords", true_coords),
            ("is_nucleotide", is_nucleotide),
            ("coords_mask", coords_mask),
        ]:
            if not isinstance(dt, DTensor):
                raise TypeError(f"{name} must be DTensor, got {type(dt)}")
            if dt.device_mesh != device_mesh:
                raise ValueError(f"{name} has different device_mesh than expected")

        expected = (Shard(0), Shard(1))
        for name, dt in [
            ("pred_coords", pred_coords),
            ("true_coords", true_coords),
            ("is_nucleotide", is_nucleotide),
            ("coords_mask", coords_mask),
        ]:
            if dt.placements != expected:
                raise ValueError(f"{name} placements {dt.placements} must be {expected}")
            for i_dim, placement in enumerate(dt.placements):
                if isinstance(placement, Partial):
                    raise ValueError(f"Partial placement on {name} mesh dim {i_dim} is not supported")
                elif isinstance(placement, Shard):
                    if dt.shape[placement.dim] % device_mesh.shape[i_dim] != 0:
                        raise ValueError(
                            f"Uneven sharding: {name} tensor dimension {placement.dim} of size "
                            f"{dt.shape[placement.dim]} along device mesh dimension {i_dim} "
                            f"of size {device_mesh.shape[i_dim]} is not supported"
                        )

        if pred_coords.ndim != 3:
            raise ValueError(f"pred_coords must be 3D [B*M, N, 3], got {pred_coords.ndim}D")
        if true_coords.ndim != 3:
            raise ValueError(f"true_coords must be 3D [B*M, N, 3], got {true_coords.ndim}D")

        # --- Extract local tensors ---
        pred_coords_local = pred_coords.to_local()  # [B_local*M, N_local, 3]
        true_coords_local = true_coords.to_local()  # [B_local*M, N_local, 3]
        is_nucleotide_local = is_nucleotide.to_local()  # [B_local, N_local]
        coords_mask_local = coords_mask.to_local()  # [B_local, N_local]

        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        n_local = pred_coords_local.shape[1]

        # --- All-gather coordinates across cp ---
        pred_coords_full = _all_gather_coords_1d(pred_coords_local, cp_group, cp_size)
        true_coords_full = _all_gather_coords_1d(true_coords_local, cp_group, cp_size)

        # All-gather mask across cp for column masking
        if cp_size == 1:
            coords_mask_full = coords_mask_local
        else:
            gathered_mask = [torch.empty_like(coords_mask_local) for _ in range(cp_size)]
            dist.all_gather(gathered_mask, coords_mask_local.contiguous(), group=cp_group)
            coords_mask_full = torch.cat(gathered_mask, dim=1)

        # --- Local computation ---
        if use_triton and has_smooth_lddt_loss_triton_kernels and pred_coords_local.is_cuda:
            num_local, den_local = _smooth_lddt_triton_forward_local(
                pred_coords_local,
                true_coords_local,
                pred_coords_full,
                true_coords_full,
                is_nucleotide_local,
                coords_mask_local,
                coords_mask_full,
                nucleic_acid_cutoff,
                other_cutoff,
                multiplicity,
                cp_rank,
                n_local,
            )
        else:
            num_local, den_local = _smooth_lddt_forward_local(
                pred_coords_local,
                true_coords_local,
                pred_coords_full,
                true_coords_full,
                is_nucleotide_local,
                coords_mask_local,
                coords_mask_full,
                nucleic_acid_cutoff,
                other_cutoff,
                multiplicity,
                cp_rank,
                n_local,
            )

        # --- Reduce num/den across cp ---
        metrics = torch.stack([num_local, den_local])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM, group=cp_group)
        num_reduced, den_reduced = metrics[0], metrics[1]

        # v2: add epsilon to denominator
        den_reduced = den_reduced + 1e-5

        # Compute LDDT per sample
        lddt_per_sample = 1.0 - num_reduced / den_reduced  # [B_local*M]

        # --- Global loss: mean over batch (all_reduce over DP) ---
        B_global = pred_coords.shape[0]
        lddt_sum = lddt_per_sample.sum()
        dist.all_reduce(lddt_sum, op=dist.ReduceOp.SUM, group=dp_group)
        lddt_final = lddt_sum / B_global

        # --- Save for backward ---
        ctx.nucleic_acid_cutoff = nucleic_acid_cutoff
        ctx.other_cutoff = other_cutoff
        ctx.multiplicity = multiplicity
        ctx.use_triton = use_triton and has_smooth_lddt_loss_triton_kernels and pred_coords_local.is_cuda
        ctx.device_mesh = device_mesh
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.n_local = n_local
        ctx.B_global = B_global
        ctx.pred_coords_shape = pred_coords.shape
        ctx.pred_coords_stride = pred_coords.stride()

        ctx.save_for_backward(
            pred_coords_local,
            true_coords_local,
            is_nucleotide_local,
            coords_mask_local,
            num_reduced,
            den_reduced,
        )

        return DTensor.from_local(
            lddt_final,
            device_mesh,
            (Replicate(), Replicate()),
            shape=torch.Size(()),
            stride=(),
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: FunctionCtx, grad_output: DTensor):
        """Backward pass.

        Recomputes the all-gather of coordinates, then computes local gradients
        using Triton or PyTorch. The gradient w.r.t. the full coordinate set
        is reduce-scattered back to local shards and summed with the direct
        local gradient.

        Communication: 4 collective calls (3 all_gathers + 1 reduce_scatter).
        """
        (
            pred_coords_local,
            true_coords_local,
            is_nucleotide_local,
            coords_mask_local,
            num_reduced,
            den_reduced,
        ) = ctx.saved_tensors

        # Chain rule: lddt_final = sum(1 - num/den) / B_global
        # d(lddt_final)/d(num_reduced) = -1 / (den_reduced * B_global)
        grad_output_local = grad_output.to_local().to(pred_coords_local.dtype)
        scale = grad_output_local.squeeze() / ctx.B_global

        inv_den = 1.0 / den_reduced
        grad_num_reduced = scale * (-inv_den)

        # Recompute all-gathers to avoid saving O(N) tensors
        pred_coords_full = _all_gather_coords_1d(pred_coords_local, ctx.cp_group, ctx.cp_size)
        true_coords_full = _all_gather_coords_1d(true_coords_local, ctx.cp_group, ctx.cp_size)
        if ctx.cp_size == 1:
            coords_mask_full = coords_mask_local
        else:
            gathered_mask = [torch.empty_like(coords_mask_local) for _ in range(ctx.cp_size)]
            dist.all_gather(gathered_mask, coords_mask_local.contiguous(), group=ctx.cp_group)
            coords_mask_full = torch.cat(gathered_mask, dim=1)

        if ctx.use_triton:
            grad_pred_local, grad_pred_full = _smooth_lddt_triton_backward_local(
                grad_num_reduced,
                pred_coords_local,
                true_coords_local,
                pred_coords_full,
                true_coords_full,
                is_nucleotide_local,
                coords_mask_local,
                coords_mask_full,
                ctx.nucleic_acid_cutoff,
                ctx.other_cutoff,
                ctx.multiplicity,
                ctx.cp_rank,
                ctx.n_local,
            )
        else:
            grad_pred_local, grad_pred_full = _smooth_lddt_backward_local(
                grad_num_reduced,
                pred_coords_local,
                true_coords_local,
                pred_coords_full,
                true_coords_full,
                is_nucleotide_local,
                coords_mask_local,
                coords_mask_full,
                ctx.nucleic_acid_cutoff,
                ctx.other_cutoff,
                ctx.multiplicity,
                ctx.cp_rank,
                ctx.n_local,
            )

        # Reduce-scatter grad_pred_full back to local shards
        if ctx.cp_size == 1:
            grad_from_full = grad_pred_full
        else:
            # Split full gradient into cp_size chunks along dim=1
            chunks = list(grad_pred_full.chunk(ctx.cp_size, dim=1))
            grad_from_full = torch.empty_like(pred_coords_local)
            dist.reduce_scatter(
                grad_from_full, [c.contiguous() for c in chunks], op=dist.ReduceOp.SUM, group=ctx.cp_group
            )

        # Total gradient: direct local + scattered from full
        total_grad_local = grad_pred_local + grad_from_full

        grad_pred = DTensor.from_local(
            total_grad_local,
            ctx.device_mesh,
            (Shard(0), Shard(1)),
            shape=ctx.pred_coords_shape,
            stride=ctx.pred_coords_stride,
        )

        return (
            grad_pred,  # pred_coords
            None,  # true_coords
            None,  # is_nucleotide
            None,  # coords_mask
            None,  # device_mesh
            None,  # dp_group
            None,  # cp_group
            None,  # nucleic_acid_cutoff
            None,  # other_cutoff
            None,  # multiplicity
            None,  # use_triton
        )


def smooth_lddt_loss_1d(
    pred_coords: DTensor,
    true_coords: DTensor,
    is_nucleotide: DTensor,
    coords_mask: DTensor,
    device_mesh: DeviceMesh,
    dp_group: ProcessGroup,
    cp_group: ProcessGroup,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
    use_triton: bool = True,
) -> DTensor:
    """Compute smooth LDDT loss for 1D CP on 2D mesh ``(dp, cp)``.

    Parameters
    ----------
    pred_coords : DTensor
        Predicted atom coordinates [B*M, N, 3], placements (Shard(0), Shard(1)).
    true_coords : DTensor
        True atom coordinates [B*M, N, 3], placements (Shard(0), Shard(1)).
    is_nucleotide : DTensor
        Nucleotide flags [B, N], placements (Shard(0), Shard(1)).
        Pre-multiplicity (the function handles repeat_interleave internally).
    coords_mask : DTensor
        Atom mask [B, N], placements (Shard(0), Shard(1)).
        Pre-multiplicity (the function handles repeat_interleave internally).
    device_mesh : DeviceMesh
        2D device mesh (dp, cp).
    dp_group : ProcessGroup
        Process group for the dp mesh dimension.
    cp_group : ProcessGroup
        Process group for the cp mesh dimension.
    nucleic_acid_cutoff : float
        Distance cutoff for nucleic acid atoms.
    other_cutoff : float
        Distance cutoff for non-nucleic acid atoms.
    multiplicity : int
        Number of diffusion samples per conformer.
    use_triton : bool
        Whether to use Triton kernels (falls back to PyTorch if unavailable).

    Returns
    -------
    DTensor
        Scalar loss, placements (Replicate(), Replicate()).
    """
    with torch.autocast("cuda", enabled=False):
        return _SmoothLDDTLoss1DCP.apply(
            pred_coords,
            true_coords,
            is_nucleotide,
            coords_mask,
            device_mesh,
            dp_group,
            cp_group,
            nucleic_acid_cutoff,
            other_cutoff,
            multiplicity,
            use_triton,
        )
