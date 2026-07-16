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

"""Triton kernel for grouped clash denom computation.

Computes per-token neighbour counts within a distance cutoff, segregated by
chain identity (``asym_id``) and entity identity (``entity_id``), in a single
O(N^2) pass with O(N) memory.  Three per-token denoms are produced:

* **total** -- all valid non-self pairs within cutoff.
* **same_chain** -- subset where ``asym_id[row] == asym_id[col]``.
* **same_entity** -- subset where ``entity_id[row] == entity_id[col]``.

From these the caller derives intra/inter/homo/hetero counts:

    inter       = total - same_chain
    homo_inter  = same_entity - same_chain
    hetero_inter = inter - homo_inter
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _clash_denom_grouped_kernel(
    # Pointers
    coords_ptr,
    mask_ptr,
    chain_id_ptr,
    entity_id_ptr,
    out_total_ptr,
    out_chain_ptr,
    out_entity_ptr,
    # Shapes
    B_mul,
    N_tok,
    SIZE_DIM_D: tl.constexpr,
    # Strides -- coords [B*mul, N, 3]
    stride_coords_b,
    stride_coords_n,
    stride_coords_d,
    # Strides -- mask [B, N]
    stride_mask_b,
    stride_mask_n,
    # Strides -- chain_id [B, N]
    stride_cid_b,
    stride_cid_n,
    # Strides -- entity_id [B, N]
    stride_eid_b,
    stride_eid_n,
    # Strides -- outputs [B*mul, N]
    stride_out_b,
    stride_out_n,
    # Constants
    cutoff,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MULTIPLICITY: tl.constexpr,
    # Memory layout orders for make_block_ptr (computed from argsort of strides)
    ORDER_COORDS_0: tl.constexpr,
    ORDER_COORDS_1: tl.constexpr,
    ORDER_COORDS_2: tl.constexpr,
    ORDER_MASK_0: tl.constexpr,
    ORDER_MASK_1: tl.constexpr,
    ORDER_CID_0: tl.constexpr,
    ORDER_CID_1: tl.constexpr,
    ORDER_EID_0: tl.constexpr,
    ORDER_EID_1: tl.constexpr,
):
    BLOCK_D: tl.constexpr = 4  # next power-of-2 for dim=3

    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    batch_idx = pid_batch // MULTIPLICITY

    # -- load coordinates [BLOCK_M/N, BLOCK_D] via make_block_ptr ---------------
    row_coords_ptr = tl.make_block_ptr(
        base=coords_ptr,
        shape=(B_mul, N_tok, SIZE_DIM_D),
        strides=(stride_coords_b, stride_coords_n, stride_coords_d),
        offsets=(pid_batch, pid_m * BLOCK_M, 0),
        block_shape=(1, BLOCK_M, BLOCK_D),
        order=(ORDER_COORDS_0, ORDER_COORDS_1, ORDER_COORDS_2),
    )
    col_coords_ptr = tl.make_block_ptr(
        base=coords_ptr,
        shape=(B_mul, N_tok, SIZE_DIM_D),
        strides=(stride_coords_b, stride_coords_n, stride_coords_d),
        offsets=(pid_batch, pid_n * BLOCK_N, 0),
        block_shape=(1, BLOCK_N, BLOCK_D),
        order=(ORDER_COORDS_0, ORDER_COORDS_1, ORDER_COORDS_2),
    )

    row_coords = tl.reshape(
        tl.load(row_coords_ptr, boundary_check=(1, 2), padding_option="zero"),
        (BLOCK_M, BLOCK_D),
    )
    col_coords = tl.reshape(
        tl.load(col_coords_ptr, boundary_check=(1, 2), padding_option="zero"),
        (BLOCK_N, BLOCK_D),
    )

    # -- load masks [BLOCK_M], [BLOCK_N] via make_block_ptr ---------------------
    mask_row_ptr = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B_mul, N_tok),
        strides=(stride_mask_b, stride_mask_n),
        offsets=(batch_idx, pid_m * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(ORDER_MASK_0, ORDER_MASK_1),
    )
    mask_col_ptr = tl.make_block_ptr(
        base=mask_ptr,
        shape=(B_mul, N_tok),
        strides=(stride_mask_b, stride_mask_n),
        offsets=(batch_idx, pid_n * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(ORDER_MASK_0, ORDER_MASK_1),
    )

    m_row = (
        tl.reshape(
            tl.load(mask_row_ptr, boundary_check=(1,), padding_option="zero"),
            (BLOCK_M,),
        )
        > 0
    )
    m_col = (
        tl.reshape(
            tl.load(mask_col_ptr, boundary_check=(1,), padding_option="zero"),
            (BLOCK_N,),
        )
        > 0
    )

    # -- load group IDs [BLOCK_M], [BLOCK_N] via make_block_ptr -----------------
    cid_row_ptr = tl.make_block_ptr(
        base=chain_id_ptr,
        shape=(B_mul, N_tok),
        strides=(stride_cid_b, stride_cid_n),
        offsets=(batch_idx, pid_m * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(ORDER_CID_0, ORDER_CID_1),
    )
    cid_col_ptr = tl.make_block_ptr(
        base=chain_id_ptr,
        shape=(B_mul, N_tok),
        strides=(stride_cid_b, stride_cid_n),
        offsets=(batch_idx, pid_n * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(ORDER_CID_0, ORDER_CID_1),
    )
    eid_row_ptr = tl.make_block_ptr(
        base=entity_id_ptr,
        shape=(B_mul, N_tok),
        strides=(stride_eid_b, stride_eid_n),
        offsets=(batch_idx, pid_m * BLOCK_M),
        block_shape=(1, BLOCK_M),
        order=(ORDER_EID_0, ORDER_EID_1),
    )
    eid_col_ptr = tl.make_block_ptr(
        base=entity_id_ptr,
        shape=(B_mul, N_tok),
        strides=(stride_eid_b, stride_eid_n),
        offsets=(batch_idx, pid_n * BLOCK_N),
        block_shape=(1, BLOCK_N),
        order=(ORDER_EID_0, ORDER_EID_1),
    )

    cid_row = tl.reshape(tl.load(cid_row_ptr, boundary_check=(1,), padding_option="zero"), (BLOCK_M,))
    cid_col = tl.reshape(tl.load(cid_col_ptr, boundary_check=(1,), padding_option="zero"), (BLOCK_N,))
    eid_row = tl.reshape(tl.load(eid_row_ptr, boundary_check=(1,), padding_option="zero"), (BLOCK_M,))
    eid_col = tl.reshape(tl.load(eid_col_ptr, boundary_check=(1,), padding_option="zero"), (BLOCK_N,))

    # -- pairwise distances [BLOCK_M, BLOCK_N] --------------------------------
    delta = row_coords[:, None, :] - col_coords[None, :, :]
    d_sq = tl.sum(delta * delta, axis=2)
    d = tl.sqrt(d_sq)

    # -- validity mask ---------------------------------------------------------
    valid = m_row[:, None] & m_col[None, :]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    not_diag = offs_m[:, None] != offs_n[None, :]
    active = valid & not_diag & (d < cutoff)

    # -- group comparisons -----------------------------------------------------
    same_chain = cid_row[:, None] == cid_col[None, :]
    same_entity = eid_row[:, None] == eid_col[None, :]

    # -- per-row accumulation --------------------------------------------------
    total_per_row = tl.sum(active.to(tl.float32), axis=1)
    chain_per_row = tl.sum((active & same_chain).to(tl.float32), axis=1)
    entity_per_row = tl.sum((active & same_entity).to(tl.float32), axis=1)

    # -- atomic store (manual indexing required for tl.atomic_add) -------------
    bound_m = offs_m < N_tok
    row_out_offs = pid_batch * stride_out_b + offs_m * stride_out_n
    tl.atomic_add(out_total_ptr + row_out_offs, total_per_row, mask=bound_m, sem="relaxed")
    tl.atomic_add(out_chain_ptr + row_out_offs, chain_per_row, mask=bound_m, sem="relaxed")
    tl.atomic_add(out_entity_ptr + row_out_offs, entity_per_row, mask=bound_m, sem="relaxed")


def clash_denom_grouped(
    coords: torch.Tensor,
    mask: torch.Tensor,
    chain_id: torch.Tensor,
    entity_id: torch.Tensor,
    multiplicity: int,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-token clash neighbour counts grouped by chain and entity.

    Single-pass Triton kernel that returns three ``[B*mul, N]`` denom tensors:
    ``(total, same_chain, same_entity)``.  The caller can derive all five
    clash categories from these three tensors without additional kernel calls.

    Parameters
    ----------
    coords : torch.Tensor
        Representative atom coordinates, shape ``[B*mul, N, 3]``.
    mask : torch.Tensor
        Token padding mask (float), shape ``[B, N]``.
    chain_id : torch.Tensor
        Per-token chain identifier, shape ``[B, N]``.
    entity_id : torch.Tensor
        Per-token entity identifier, shape ``[B, N]``.
    multiplicity : int
        Diffusion multiplicity (``B_mul = B * multiplicity``).
    cutoff : float
        Distance cutoff in Angstrom.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(total_denom, chain_denom, entity_denom)`` each shape ``[B*mul, N]``.
    """
    B_mul, N, dim_d = coords.shape
    if dim_d != 3:
        raise ValueError(f"Coordinate dimension must be 3, got {dim_d}")
    if B_mul % multiplicity != 0:
        raise ValueError(f"Coordinate batch ({B_mul}) must be divisible by multiplicity ({multiplicity})")

    device = coords.device
    compute_dtype = torch.promote_types(coords.dtype, torch.float32)
    if coords.dtype != compute_dtype:
        coords = coords.to(compute_dtype)

    out_total = torch.zeros(B_mul, N, device=device, dtype=compute_dtype)
    out_chain = torch.zeros(B_mul, N, device=device, dtype=compute_dtype)
    out_entity = torch.zeros(B_mul, N, device=device, dtype=compute_dtype)

    BLOCK_M = 32
    BLOCK_N = 32
    grid = (B_mul, triton.cdiv(N, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Compute memory layout order for make_block_ptr
    # order = argsort of strides (ascending), giving fastest-varying dim first
    order_coords = tuple(torch.tensor(coords.stride()).argsort().tolist())
    order_mask = tuple(torch.tensor(mask.stride()).argsort().tolist())
    order_cid = tuple(torch.tensor(chain_id.stride()).argsort().tolist())
    order_eid = tuple(torch.tensor(entity_id.stride()).argsort().tolist())

    _clash_denom_grouped_kernel[grid](
        coords,
        mask,
        chain_id,
        entity_id,
        out_total,
        out_chain,
        out_entity,
        B_mul,
        N,
        dim_d,
        coords.stride(0),
        coords.stride(1),
        coords.stride(2),
        mask.stride(0),
        mask.stride(1),
        chain_id.stride(0),
        chain_id.stride(1),
        entity_id.stride(0),
        entity_id.stride(1),
        out_total.stride(0),
        out_total.stride(1),
        float(cutoff),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MULTIPLICITY=multiplicity,
        ORDER_COORDS_0=order_coords[0],
        ORDER_COORDS_1=order_coords[1],
        ORDER_COORDS_2=order_coords[2],
        ORDER_MASK_0=order_mask[0],
        ORDER_MASK_1=order_mask[1],
        ORDER_CID_0=order_cid[0],
        ORDER_CID_1=order_cid[1],
        ORDER_EID_0=order_eid[0],
        ORDER_EID_1=order_eid[1],
    )

    return out_total, out_chain, out_entity
