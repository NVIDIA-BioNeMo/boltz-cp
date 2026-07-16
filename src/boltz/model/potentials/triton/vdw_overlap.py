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

"""Triton kernel for VDW overlap potential energy and gradient computation.

This kernel is the CUDA fast-path for ``VDWOverlapPotential``.  On CPU the
potential falls back to the generic ``Potential.compute()`` / ``compute_gradient()``
pipeline, which materialises O(N^2) pair indices via ``torch.triu_indices``.
The Triton kernel tiles over the upper triangle in BLOCK x BLOCK chunks and
accumulates via ``tl.atomic_add``, keeping memory at O(N).

A single fused kernel (``_vdw_overlap_kernel``) computes energy and optionally
gradient, controlled by the ``COMPUTE_GRADIENT: tl.constexpr`` flag.  When
``COMPUTE_GRADIENT=False`` the gradient branch is eliminated at compile time.

Class hierarchy and dispatch
----------------------------
::

    Potential (ABC)
    ├── compute()           — generic pipeline (see below)
    ├── compute_gradient()  — generic pipeline with derivatives
    ├── compute_args()      — abstract: build pair indices + args
    ├── compute_variable()  — abstract: coords + index → value
    └── compute_function()  — abstract: value → energy

    FlatBottomPotential(Potential)   — compute_function: flat-bottom energy
    DistancePotential(Potential)     — compute_variable: pairwise distance

    VDWOverlapPotential(FlatBottomPotential, DistancePotential)
    ├── _extract_vdw_features()   — shared O(N) feature extraction
    ├── compute()                 — CUDA → Triton; CPU → super().compute()
    ├── compute_gradient()        — CUDA → Triton; CPU → super().compute_gradient()
    └── compute_args()            — O(N^2) pair indices (CPU fallback only)

Generic pipeline (``Potential.compute`` / ``compute_gradient``)
---------------------------------------------------------------
::

    compute_args()            → (index [2, P], (k, lower_bounds, upper_bounds), ...)
        │                        VDWOverlapPotential builds triu pair indices,
        │                        filters by pad/ion/connection masks
        ▼
    compute_variable()        → dist [P]    (DistancePotential: ||coords[i] - coords[j]||)
        │                        gradient mode also returns r_hat [2, P, 3]
        ▼
    compute_function()        → energy [P]  (FlatBottomPotential: k*(bound - dist) if overlap)
        │                        gradient mode also returns dEnergy [P] = -k
        ▼
    chain rule + scatter_reduce → grad [N, 3]
        grad[i] += dEnergy * r_hat_ij    (= -r_hat when overlapping)
        grad[j] += dEnergy * -r_hat_ij   (= +r_hat when overlapping)

Triton fast-path (this module)
------------------------------
Replaces the entire pipeline above with a single fused kernel that tiles over
the upper triangle without materialising pair indices.  The upper triangle
is linearised into ``G*(G+1)/2`` blocks (where ``G = ceil(N/BLOCK)``) so
no blocks are launched for the lower triangle.

Energy is always computed.  When ``COMPUTE_GRADIENT=True``, the gradient is
computed on top of the energy using already-computed ``diff`` and ``dist``
tensors — the extra cost is minimal (unit vector + 2D atomic scatter).

**Energy** (always computed):
    For each valid pair (i, j) with i < j, computes:
        lower_bound = (vdw_radii[i] + vdw_radii[j]) * (1 - buffer)
        dist = ||coords[i] - coords[j]||
        if dist < lower_bound: energy += lower_bound - dist
    Accumulates energy via ``tl.atomic_add`` to a scalar output.

**Gradient** (when ``COMPUTE_GRADIENT=True``):
    For each valid pair (i, j) with i < j where dist < lower_bound:
        r_hat = (coords[i] - coords[j]) / dist
        grad[i] += -r_hat   (force pushes atom i away)
        grad[j] +=  r_hat   (force pushes atom j away)
    Accumulates per-atom gradients via 2D masked ``tl.atomic_add`` to [N, 3].

**Pair filtering** (applied in both modes):
    A pair (i, j) is valid iff:
    1. Both atoms are unpadded: atom_pad_mask[i] AND atom_pad_mask[j]
    2. Both from multi-atom chains: single_ion_mask[i] AND single_ion_mask[j]
    3. Chains are not connected: NOT connected_chain_matrix[chain_id[i], chain_id[j]]

Batching
--------
The kernel supports an optional batch dimension on ``coords`` and outputs.
Features (``vdw_radii``, ``atom_pad_mask``, ``single_ion_mask``,
``atom_chain_id``, ``connected_chain_matrix``) are **shared** across the batch
— only coordinates and outputs are per-batch-element.  The grid is
``(num_triu_blocks, B)`` where ``num_triu_blocks = G*(G+1)/2``.

Input shapes:
    coords:                  [B, N, 3] or [N, 3]    float32+
    vdw_radii:               [N]       float32+
    atom_pad_mask:           [N]       bool
    single_ion_mask:         [N]       bool
    atom_chain_id:           [N]       int64
    connected_chain_matrix:  [C, C]    bool  (C = num_chains), passed flattened
    buffer:                  scalar    float

Output shapes:
    energy:   [B] or scalar  (always)
    grad:     [B, N, 3] or [N, 3]  (when COMPUTE_GRADIENT=True)
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Upper-triangle linearization helpers
# ---------------------------------------------------------------------------


@triton.jit
def _linear_to_triu(pid, G):
    """Map a linear index to (row, col) in the upper triangle of a G×G grid.

    The upper triangle (including diagonal) has G*(G+1)/2 entries, enumerated
    row-by-row:  (0,0), (0,1), ..., (0,G-1), (1,1), (1,2), ..., (G-1,G-1).

    Given a linear ``pid`` in [0, G*(G+1)/2), returns ``(pid_m, pid_n)`` with
    ``0 <= pid_m <= pid_n < G``.
    """
    # Invert the triangular number: pid_m = G - 1 - floor((-1 + sqrt(1 + 8*k)) / 2)
    # where k = num_triu - 1 - pid, num_triu = G*(G+1)//2.
    # fp64 required: float32 sqrt loses precision at G >= ~2048.
    num_triu = G * (G + 1) // 2
    k = num_triu - 1 - pid
    pid_m = G - 1 - tl.math.floor((-1.0 + tl.math.sqrt(1.0 + 8.0 * k.to(tl.float64))) / 2.0).to(tl.int32)
    # Column offset within the row
    row_start = pid_m * G - pid_m * (pid_m - 1) // 2
    pid_n = pid - row_start + pid_m
    return pid_m, pid_n


# ---------------------------------------------------------------------------
# Fused kernel: energy + optional gradient
# ---------------------------------------------------------------------------


@triton.jit
def _vdw_overlap_kernel(
    coords_ptr,
    vdw_radii_ptr,
    atom_pad_mask_ptr,
    single_ion_mask_ptr,
    atom_chain_id_ptr,
    connected_chain_matrix_ptr,
    energy_ptr,
    grad_ptr,
    N,
    num_chains,
    num_blocks,
    buffer_val,
    stride_coords_b,
    stride_coords_n,
    stride_coords_d,
    stride_grad_b,
    stride_grad_n,
    stride_grad_d,
    SIZE_DIM_D: tl.constexpr,
    BLOCK: tl.constexpr,
    ORDER_COORDS_0: tl.constexpr,
    ORDER_COORDS_1: tl.constexpr,
    COMPUTE_GRADIENT: tl.constexpr,
):
    """Tile over upper triangle, accumulate VDW overlap energy and optionally gradient."""
    BLOCK_D: tl.constexpr = 4  # next power of 2 for SIZE_DIM_D=3

    pid_triu = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_m, pid_n = _linear_to_triu(pid_triu, num_blocks)

    # Offset coords and energy by batch element
    coords_batch_ptr = coords_ptr + pid_b * stride_coords_b
    energy_batch_ptr = energy_ptr + pid_b

    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)

    mask_m = offs_m < N
    mask_n = offs_n < N

    # --- Load masks and chain ids for both row and col (shared across batch) ---
    pad_m = tl.load(atom_pad_mask_ptr + offs_m, mask=mask_m, other=0).to(tl.int1)
    pad_n = tl.load(atom_pad_mask_ptr + offs_n, mask=mask_n, other=0).to(tl.int1)

    ion_m = tl.load(single_ion_mask_ptr + offs_m, mask=mask_m, other=0).to(tl.int1)
    ion_n = tl.load(single_ion_mask_ptr + offs_n, mask=mask_n, other=0).to(tl.int1)

    chain_m = tl.load(atom_chain_id_ptr + offs_m, mask=mask_m, other=0)
    chain_n = tl.load(atom_chain_id_ptr + offs_n, mask=mask_n, other=0)

    # --- Load VDW radii (shared across batch) ---
    radii_m = tl.load(vdw_radii_ptr + offs_m, mask=mask_m, other=0.0)
    radii_n = tl.load(vdw_radii_ptr + offs_n, mask=mask_n, other=0.0)

    # --- Load coordinates via make_block_ptr [N, 3] -> (BLOCK, BLOCK_D) ---
    coords_m_block = tl.make_block_ptr(
        base=coords_batch_ptr,
        shape=(N, SIZE_DIM_D),
        strides=(stride_coords_n, stride_coords_d),
        offsets=(pid_m * BLOCK, 0),
        block_shape=(BLOCK, BLOCK_D),
        order=(ORDER_COORDS_0, ORDER_COORDS_1),
    )
    coords_n_block = tl.make_block_ptr(
        base=coords_batch_ptr,
        shape=(N, SIZE_DIM_D),
        strides=(stride_coords_n, stride_coords_d),
        offsets=(pid_n * BLOCK, 0),
        block_shape=(BLOCK, BLOCK_D),
        order=(ORDER_COORDS_0, ORDER_COORDS_1),
    )
    coords_m = tl.load(coords_m_block, boundary_check=(0, 1), padding_option="zero")
    coords_n = tl.load(coords_n_block, boundary_check=(0, 1), padding_option="zero")

    # --- Pairwise distances: (BLOCK, BLOCK) ---
    diff = coords_m[:, None, :] - coords_n[None, :, :]  # (BLOCK, BLOCK, BLOCK_D)
    dist_sq = tl.sum(diff * diff, axis=2)
    dist = tl.sqrt(dist_sq + 1e-30)  # avoid sqrt(0) edge case

    # --- Lower bounds: (BLOCK, BLOCK) ---
    lower_bound = (radii_m[:, None] + radii_n[None, :]) * (1.0 - buffer_val)

    # --- Pair validity mask (BLOCK, BLOCK) ---
    valid = pad_m[:, None] & pad_n[None, :]
    valid = valid & ion_m[:, None] & ion_n[None, :]

    # Chains not connected: look up connected_chain_matrix[chain_m, chain_n]
    connected = tl.load(
        connected_chain_matrix_ptr + chain_m[:, None] * num_chains + chain_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=1,  # default: treat as connected (skip pair)
    ).to(tl.int1)
    valid = valid & ~connected

    # Upper triangle: i < j
    if pid_m == pid_n:
        valid = valid & (offs_m[:, None] < offs_n[None, :])

    # --- Energy contribution: mask multiply instead of tl.where ---
    overflow = lower_bound - dist
    active = valid & (overflow > 0.0)
    energy_contrib = overflow * active.to(overflow.dtype)

    # Sum within block and atomic-add to this batch element's energy
    block_energy = tl.sum(energy_contrib)
    tl.atomic_add(energy_batch_ptr, block_energy.to(energy_batch_ptr.dtype.element_ty))

    # --- Gradient contribution (conditional on COMPUTE_GRADIENT constexpr) ---
    if COMPUTE_GRADIENT:
        grad_batch_ptr = grad_ptr + pid_b * stride_grad_b

        # For active pairs (dist < lower_bound):
        #   grad_i += -r_hat,  grad_j += r_hat
        inv_dist = 1.0 / dist
        r_hat = diff * inv_dist[:, :, None]  # (BLOCK, BLOCK, BLOCK_D)

        # Mask inactive pairs via cast instead of tl.where
        active_f = active.to(r_hat.dtype)[:, :, None]  # (BLOCK, BLOCK, 1)

        # grad_i: sum over j of -r_hat for active pairs -> (BLOCK, BLOCK_D)
        grad_m = tl.sum(-r_hat * active_f, axis=1)

        # grad_j: sum over i of r_hat for active pairs -> (BLOCK, BLOCK_D)
        grad_n = tl.sum(r_hat * active_f, axis=0)

        # 2D masked atomic_add over (BLOCK, BLOCK_D); the 4th column (padding)
        # is masked out by d_valid so it generates no memory traffic.
        offs_d = tl.arange(0, BLOCK_D)
        d_valid = offs_d < 3
        ptrs_m = grad_batch_ptr + offs_m[:, None] * stride_grad_n + offs_d[None, :] * stride_grad_d
        ptrs_n = grad_batch_ptr + offs_n[:, None] * stride_grad_n + offs_d[None, :] * stride_grad_d
        mask_2d_m = mask_m[:, None] & d_valid[None, :]
        mask_2d_n = mask_n[:, None] & d_valid[None, :]
        tl.atomic_add(ptrs_m, grad_m, mask=mask_2d_m)
        tl.atomic_add(ptrs_n, grad_n, mask=mask_2d_n)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 32


def _validate_inputs(coords: torch.Tensor, vdw_radii: torch.Tensor) -> None:
    """Validate that floating-point inputs are at least float32 precision."""
    for name, tensor in [("coords", coords), ("vdw_radii", vdw_radii)]:
        if tensor.is_floating_point() and tensor.dtype in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"{name} dtype {tensor.dtype} is below float32; "
                "VDW overlap kernel requires at least float32 precision"
            )


def _vdw_overlap_impl(
    coords: torch.Tensor,
    vdw_radii: torch.Tensor,
    atom_pad_mask: torch.Tensor,
    single_ion_mask: torch.Tensor,
    atom_chain_id: torch.Tensor,
    connected_chain_matrix: torch.Tensor,
    buffer: float,
    compute_gradient: bool,
) -> Tuple[torch.Tensor, ...]:
    """Shared implementation for energy and gradient computation.

    Returns ``(energy,)`` when ``compute_gradient=False``,
    ``(energy, grad)`` when ``compute_gradient=True``.
    The ``batched`` flag is encoded in the tensor shapes (B present or not).
    """
    _validate_inputs(coords, vdw_radii)

    batched = coords.dim() == 3
    if not batched:
        coords = coords.unsqueeze(0)

    B, N, _ = coords.shape
    compute_dtype = torch.promote_types(coords.dtype, torch.float32)
    if N == 0:
        energy = torch.zeros(B, device=coords.device, dtype=compute_dtype)
        if compute_gradient:
            grad = torch.zeros((B, 0, 3), device=coords.device, dtype=compute_dtype)
            return (
                energy if batched else energy.squeeze(0),
                grad if batched else grad.squeeze(0),
            )
        return (energy if batched else energy.squeeze(0),)

    coords = coords.to(compute_dtype).contiguous()
    vdw_radii = vdw_radii.to(compute_dtype).contiguous()
    atom_pad_mask = atom_pad_mask.bool().contiguous()
    single_ion_mask = single_ion_mask.bool().contiguous()
    atom_chain_id = atom_chain_id.long().contiguous()
    connected_chain_matrix = connected_chain_matrix.bool().contiguous()

    num_chains = connected_chain_matrix.shape[0]
    connected_flat = connected_chain_matrix.flatten()

    energy = torch.zeros(B, device=coords.device, dtype=compute_dtype)
    if compute_gradient:
        grad = torch.zeros((B, N, 3), device=coords.device, dtype=compute_dtype)

    num_blocks = triton.cdiv(N, _BLOCK_SIZE)
    num_triu_blocks = num_blocks * (num_blocks + 1) // 2
    grid = (num_triu_blocks, B)

    # Strides of the [N, 3] sub-tensor (last two dims), pure Python sort
    strides_2d = coords.stride()[1:]
    order_coords = tuple(sorted(range(len(strides_2d)), key=lambda i: strides_2d[i], reverse=True))

    _vdw_overlap_kernel[grid](
        coords,
        vdw_radii,
        atom_pad_mask,
        single_ion_mask,
        atom_chain_id,
        connected_flat,
        energy,
        grad if compute_gradient else torch.empty(0, device=coords.device),
        N,
        num_chains,
        num_blocks,
        buffer,
        coords.stride(0),
        coords.stride(1),
        coords.stride(2),
        grad.stride(0) if compute_gradient else 0,
        grad.stride(1) if compute_gradient else 0,
        grad.stride(2) if compute_gradient else 0,
        SIZE_DIM_D=3,
        BLOCK=_BLOCK_SIZE,
        ORDER_COORDS_0=order_coords[0],
        ORDER_COORDS_1=order_coords[1],
        COMPUTE_GRADIENT=compute_gradient,
    )

    energy = energy if batched else energy.squeeze(0)
    if compute_gradient:
        grad = grad if batched else grad.squeeze(0)
        return energy, grad
    return (energy,)


def vdw_overlap_energy(
    coords: torch.Tensor,
    vdw_radii: torch.Tensor,
    atom_pad_mask: torch.Tensor,
    single_ion_mask: torch.Tensor,
    atom_chain_id: torch.Tensor,
    connected_chain_matrix: torch.Tensor,
    buffer: float,
) -> torch.Tensor:
    """Compute VDW overlap flat-bottom potential energy using a Triton kernel.

    Supports batched coords ``[B, N, 3]`` with shared features (masks, radii,
    chain ids, connected matrix are the same for all batch elements).
    Unbatched ``[N, 3]`` coords are also accepted and return a scalar.
    """
    (energy,) = _vdw_overlap_impl(
        coords,
        vdw_radii,
        atom_pad_mask,
        single_ion_mask,
        atom_chain_id,
        connected_chain_matrix,
        buffer,
        compute_gradient=False,
    )
    return energy


def vdw_overlap_gradient(
    coords: torch.Tensor,
    vdw_radii: torch.Tensor,
    atom_pad_mask: torch.Tensor,
    single_ion_mask: torch.Tensor,
    atom_chain_id: torch.Tensor,
    connected_chain_matrix: torch.Tensor,
    buffer: float,
) -> torch.Tensor:
    """Compute VDW overlap flat-bottom potential gradient using a Triton kernel.

    Supports batched coords ``[B, N, 3]`` with shared features.
    Unbatched ``[N, 3]`` coords are also accepted and return ``[N, 3]``.
    """
    _, grad = _vdw_overlap_impl(
        coords,
        vdw_radii,
        atom_pad_mask,
        single_ion_mask,
        atom_chain_id,
        connected_chain_matrix,
        buffer,
        compute_gradient=True,
    )
    return grad
