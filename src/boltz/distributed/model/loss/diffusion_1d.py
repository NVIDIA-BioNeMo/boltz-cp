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

import warnings

import torch
from einops import einsum
from torch.distributed.tensor import DTensor, Shard


def weighted_rigid_align_1d(
    true_coords: DTensor,
    pred_coords: DTensor,
    weights: DTensor,
    mask: DTensor,
) -> DTensor:
    """Compute weighted alignment and return the aligned true coordinates using 1D-CP DTensor.

    1D-CP variant of :func:`weighted_rigid_align`. Operates on a 2-axis ``(dp, cp)``
    mesh with placements ``(Shard(0), Shard(1))`` — the atom dimension is sharded
    over the flat ``cp`` group, and the coordinate axis is local (no inner
    Replicate broadcast axis exists in 1D-CP).

    Algorithm parity with :func:`boltz.model.loss.diffusionv2.weighted_rigid_align`
    is preserved by reducing weighted centroids and the covariance matrix over the
    ``cp`` group, then computing SVD redundantly on every rank (inputs to SVD are
    bit-identical after the reduction). This is NOT differentiable
    (``torch.no_grad()`` internally).

    Parameters
    ----------
    true_coords : DTensor
        Ground truth atom coordinates, shape (B, N, 3).
        Placements: (Shard(0), Shard(1)).
    pred_coords : DTensor
        Predicted atom coordinates, shape (B, N, 3).
        Placements: (Shard(0), Shard(1)).
    weights : DTensor
        Alignment weights, shape (B, N).
        Placements: (Shard(0), Shard(1)).
    mask : DTensor
        Atom mask, shape (B, N).
        Placements: (Shard(0), Shard(1)).

    Returns
    -------
    DTensor
        Aligned true coordinates with same placements as input true_coords.
    """
    # Ndim checks (3D coords, 2D weights/mask)
    if true_coords.ndim != 3:
        raise ValueError(f"true_coords must be 3D (B, N, 3), got ndim={true_coords.ndim}")
    if pred_coords.ndim != 3:
        raise ValueError(f"pred_coords must be 3D (B, N, 3), got ndim={pred_coords.ndim}")
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D (B, N), got ndim={weights.ndim}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (B, N), got ndim={mask.ndim}")

    # Shape checks
    if true_coords.shape != pred_coords.shape:
        raise ValueError(f"true_coords shape {true_coords.shape} != pred_coords shape {pred_coords.shape}")
    if weights.shape != mask.shape:
        raise ValueError(f"weights shape {weights.shape} != mask shape {mask.shape}")
    if weights.shape != true_coords.shape[:2]:
        raise ValueError(f"weights shape {weights.shape} != expected {true_coords.shape[:2]}")

    # Device mesh checks
    if true_coords.device_mesh != pred_coords.device_mesh:
        raise ValueError("true_coords and pred_coords must be on the same device_mesh")
    if true_coords.device_mesh != weights.device_mesh:
        raise ValueError("true_coords and weights must be on the same device_mesh")
    if true_coords.device_mesh != mask.device_mesh:
        raise ValueError("true_coords and mask must be on the same device_mesh")

    # Placement checks: 1D-CP expects (Shard(0), Shard(1)) on a 2-axis (dp, cp) mesh
    placements = (Shard(0), Shard(1))
    if true_coords.device_mesh.ndim != 2:
        raise ValueError(
            f"weighted_rigid_align_1d expects a 2-axis device_mesh (dp, cp), got ndim={true_coords.device_mesh.ndim}"
        )
    if true_coords.placements != placements:
        raise ValueError(f"true_coords placements {true_coords.placements} != expected {placements}")
    if pred_coords.placements != placements:
        raise ValueError(f"pred_coords placements {pred_coords.placements} != expected {placements}")
    if weights.placements != placements:
        raise ValueError(f"weights placements {weights.placements} != expected {placements}")
    if mask.placements != placements:
        raise ValueError(f"mask placements {mask.placements} != expected {placements}")

    # Even sharding guards
    device_mesh = true_coords.device_mesh
    size_cp = device_mesh.shape[1]
    if true_coords.shape[1] % size_cp != 0:
        raise ValueError(
            f"weighted_rigid_align_1d requires even sharding on atom dim: "
            f"shape[1]={true_coords.shape[1]} not divisible by cp size={size_cp}"
        )

    with torch.no_grad():
        # Convert to local tensors
        true_coords_local = true_coords.to_local()
        pred_coords_local = pred_coords.to_local()
        weights_local = weights.to_local()
        mask_local = mask.to_local()

        # Mesh axis 1: reduce along atom dimension (cp axis, flat)
        group_reduce_atoms = device_mesh.get_group(1)

        batch_size, num_points, dim = true_coords_local.shape
        weights_expanded = (mask_local * weights_local).unsqueeze(-1)

        # Scalar degenerate check
        total_num_points = num_points * size_cp
        if total_num_points < (dim + 1):
            warnings.warn(
                "The size of one of the point clouds is <= dim+1. "
                "`WeightedRigidAlign` cannot return a unique rotation.",
                UserWarning,
                stacklevel=1,
            )

        rank_coord = device_mesh.get_coordinate()
        assert rank_coord is not None

        # Per-batch degenerate check (mask.sum(dim=-1) < (dim + 1) after cp reduction).
        # After the all_reduce, mask_count_global is bit-identical across cp ranks
        # within a dp shard, so every rank would emit identical warnings. Gate to
        # the first cp rank (rank_coord[1] == 0) to deduplicate; mirror of the 2D
        # variant at L164/L167.
        mask_count_global = mask_local.sum(dim=1).clone()
        torch.distributed.all_reduce(
            mask_count_global,
            op=torch.distributed.ReduceOp.SUM,
            group=group_reduce_atoms,
        )
        is_first_cp_rank = rank_coord[1] == 0
        if is_first_cp_rank:
            degenerate_batch_indices = torch.where(mask_count_global < (dim + 1))[0]
            if degenerate_batch_indices.numel() > 0:
                warnings.warn(
                    f"[rank_coord:{rank_coord}] "
                    "The size of one of the point clouds is <= dim+1. "
                    "`WeightedRigidAlign` cannot return a unique rotation. "
                    f"Batch indices (subset): {degenerate_batch_indices.tolist()}",
                    UserWarning,
                    stacklevel=1,
                )

        # Overlapped async reductions for centroids (dim=1 = atoms, sharded on cp)
        weights_sum_local = weights_expanded.sum(dim=1, keepdim=True)
        req_reduce_weights = torch.distributed.all_reduce(
            weights_sum_local, op=torch.distributed.ReduceOp.SUM, group=group_reduce_atoms, async_op=True
        )

        true_coords_weighted_sum_local = (true_coords_local * weights_expanded).sum(dim=1, keepdim=True)
        req_reduce_true_coords = torch.distributed.all_reduce(
            true_coords_weighted_sum_local,
            op=torch.distributed.ReduceOp.SUM,
            group=group_reduce_atoms,
            async_op=True,
        )

        pred_coords_weighted_sum_local = (pred_coords_local * weights_expanded).sum(dim=1, keepdim=True)
        req_reduce_pred_coords = torch.distributed.all_reduce(
            pred_coords_weighted_sum_local,
            op=torch.distributed.ReduceOp.SUM,
            group=group_reduce_atoms,
            async_op=True,
        )

        req_reduce_weights.wait()
        req_reduce_true_coords.wait()
        true_centroid = true_coords_weighted_sum_local / weights_sum_local

        req_reduce_pred_coords.wait()
        pred_centroid = pred_coords_weighted_sum_local / weights_sum_local

        # Center the coordinates
        true_coords_centered = true_coords_local - true_centroid
        pred_coords_centered = pred_coords_local - pred_centroid

        # Compute the weighted covariance matrix; reduce over cp atoms
        cov_matrix_local = einsum(
            weights_expanded * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
        )
        cov_matrix_local = cov_matrix_local.contiguous()
        torch.distributed.all_reduce(
            cov_matrix_local,
            op=torch.distributed.ReduceOp.SUM,
            group=group_reduce_atoms,
        )
        original_dtype = cov_matrix_local.dtype

        # SVD in float64 for numerical stability. After the all-reduce the inputs are
        # bit-identical across cp ranks, so SVD computed independently on each rank
        # yields the same rotation matrix without further communication.
        cov_matrix_64 = cov_matrix_local.to(dtype=torch.float64)
        U, S, V = torch.linalg.svd(cov_matrix_64, driver="gesvd" if cov_matrix_64.is_cuda else None)
        V = V.mH

        # SVD low-rank warning: after the all_reduce above, cov_matrix_64 is
        # bit-identical across cp ranks within a dp shard, so every rank would
        # emit identical warnings. Gate to the first cp rank to deduplicate;
        # mirror of the 2D variant at L253-261.
        if is_first_cp_rank and (S.abs() <= 1e-15).any() and not (total_num_points < (dim + 1)):
            warnings.warn(
                f"[rank_coord:{rank_coord}] "
                "Excessively low rank of "
                "cross-correlation between aligned point clouds. "
                "`WeightedRigidAlign` cannot return a unique rotation.",
                UserWarning,
                stacklevel=1,
            )

        # Rotation matrix with proper determinant
        rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V)
        F = torch.eye(dim, dtype=cov_matrix_64.dtype, device=cov_matrix_64.device)[None].repeat(batch_size, 1, 1)
        F[:, -1, -1] = torch.det(rot_matrix)
        rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
        rot_matrix = rot_matrix.to(dtype=original_dtype)

        # Apply rotation and translation
        aligned_coords_local = einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j") + pred_centroid

        # Convert back to DTensor
        aligned_coords = DTensor.from_local(
            aligned_coords_local,
            device_mesh=device_mesh,
            placements=true_coords.placements,
            shape=true_coords.shape,
            stride=true_coords.stride(),
        )

        return aligned_coords
