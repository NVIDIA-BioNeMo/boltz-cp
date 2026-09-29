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

"""1D CP ConfidenceModule and ConfidenceHeads for 2D mesh ``(dp, cp)``.

Adapts the 2D CP confidence module (confidencev2.py / confidence_utils.py) for
1D context parallelism. The key difference is that tensors use 2-element
placements instead of 3-element:

- Single ``s [B, N, C_s]``: ``(Shard(0), Shard(1))``
- Pair ``z [B, N, N, C_z]``: ``(Shard(0), Shard(1))`` (row-slab)
- Scalar metrics: ``(Shard(0), Replicate())``

Row-slab sharding means each CP rank holds ``[B_local, N/cp, N_full, ...]``
for pair tensors.  Outer ops require an all-gather of column data across cp
(instead of TransposeComm).  PDE transpose ``z + z^T`` requires an all-gather.

Communication budget per forward pass:
- LayerNorms / linears: 0 collectives (params Replicate)
- Outer sum s -> z: 1 all-gather (via ``outer_sum_1d``)
- Optional outer product: 1 all-gather
- Distogram cdist: 1 all-gather
- RelativePositionEncoder: 6 all-gathers (one per feature key)
- PairformerModule1D: O(depth) collectives (ring attention + triangle)
- ConfidenceHeads1D:
  - PDE transpose: 1 all-to-all (NCCL) / 1 all-gather (Gloo)
  - pLDDT / iPLDDT: 1 all-reduce each over cp
  - PDE / iPDE: 1 all-reduce each over cp
  - pTM / ipTM: O(num_chains^2) all-reduces
"""

import warnings
from math import pi

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from boltz.data import const
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, elementwise_op
from boltz.distributed.model.layers.embedding import EmbeddingParamsReplicated
from boltz.distributed.model.layers.layernorm import LayerNormParamsReplicated
from boltz.distributed.model.layers.linear import LinearParamsReplicated
from boltz.distributed.model.layers.outer_sum_1d import outer_sum_1d
from boltz.distributed.model.layers.pairformer_1d import PairformerModule1D
from boltz.distributed.model.layers.repeat_interleave import shardwise_repeat_interleave
from boltz.distributed.model.layers.shardwise_op import shardwise_distogram
from boltz.distributed.model.modules.confidence_utils import compute_aggregated_metric
from boltz.distributed.model.modules.encoders_1d import RelativePositionEncoder1D
from boltz.distributed.utils import all_gather_on_cp, update_exhaustive_strides
from boltz.model.modules.confidence_utils import tm_function as serial_tm_function
from boltz.model.modules.confidencev2 import (
    IPLDDT_INTERFACE_WEIGHT,
    IPLDDT_LIGAND_WEIGHT,
    IPLDDT_NON_INTERFACE_WEIGHT,
)
from boltz.model.modules.confidencev2 import ConfidenceHeads as SerialConfidenceHeads
from boltz.model.modules.confidencev2 import ConfidenceModule as SerialConfidenceModule

# 1D CP placements on 2D mesh (dp, cp)
SINGLE_PLACEMENTS_1D = (Shard(0), Shard(1))
PAIR_PLACEMENTS_1D = (Shard(0), Shard(1))
SCALAR_PLACEMENTS_1D = (Shard(0), Replicate())

# Small constant added to denominators to avoid division by zero
_EPS = 1e-5

# Sentinel value for chain_pair_iptm entries where the chain pair does not exist
CHAIN_IPTM_SENTINEL = -1.0

# x_pred atoms are sharded on cp (block-diagonal atom_to_token padding ensures
# N_atoms_per_shard is consistent across ranks): [B*mult, N_atoms, 3] → (Shard(0), Shard(1))
ATOM_PLACEMENTS_1D = (Shard(0), Shard(1))


def _rep_atom_to_token_1d(
    x_pred: DTensor,
    token_to_rep_atom: DTensor,
    device_mesh: DeviceMesh,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Project atom coordinates to token level via representative atoms for 1D CP.

    Computes ``token_to_rep_atom @ x_pred`` to produce token-level coordinates.
    Under 1D CP, tokens are sharded on cp and ``token_to_rep_atom`` keeps the
    full atom dimension (``[B_local, N_tokens/cp, N_atoms]``). ``x_pred`` atoms
    are sharded on cp (``[B_local*mult, N_atoms/cp, 3]``), so we all-gather
    x_pred along the atom dim before the matmul.

    Communication: 1 all-gather on x_pred along dim 1 (atom dim).

    Parameters
    ----------
    x_pred : DTensor
        ``[B*mult, N_atoms, 3]`` with placements ``(Shard(0), Shard(1))``.
    token_to_rep_atom : DTensor
        ``[B, N_tokens, N_atoms]`` with placements ``(Shard(0), Shard(1))``.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : ProcessGroup
        The cp process group.

    Returns
    -------
    DTensor
        ``[B*mult, N_tokens, 3]`` with placements ``(Shard(0), Shard(1))``.
    """
    cp_size = device_mesh.shape[1]
    t2ra_local = token_to_rep_atom.to_local()  # [B_local, N_tokens/cp, N_atoms]
    x_local = x_pred.to_local()  # [B_local*mult, N_atoms/cp, 3]

    # All-gather x_pred along atom dim to match token_to_rep_atom's full N_atoms.
    x_full = all_gather_on_cp(x_local, dim=1, cp_group=cp_group, cp_size=cp_size)
    # x_full: [B_local*mult, N_atoms, 3]

    B_local = t2ra_local.shape[0]
    mult = x_full.shape[0] // B_local

    compute_dtype = torch.promote_types(t2ra_local.dtype, torch.float32)
    t2ra_f = t2ra_local.to(dtype=compute_dtype)
    x_f = x_full.to(dtype=compute_dtype)

    # Reshape x_pred to [B_local, mult, N_atoms, 3] for broadcasting over mult
    x_reshaped = x_f.reshape(B_local, mult, *x_f.shape[1:])
    # [B_local, N_tokens/cp, N_atoms] @ [B_local, mult, N_atoms, 3]
    # → [B_local, mult, N_tokens/cp, 3]
    out = torch.einsum("btn,bmnc->bmtc", t2ra_f, x_reshaped)
    # Flatten back to [B_local*mult, N_tokens/cp, 3]
    out = out.reshape(B_local * mult, *out.shape[2:])

    B_global = token_to_rep_atom.shape[0]
    N_global = token_to_rep_atom.shape[1]
    out_shape = torch.Size([B_global * mult, N_global, 3])
    out_stride = update_exhaustive_strides(out.shape, out.stride(), out_shape)
    return DTensor.from_local(
        out,
        device_mesh=device_mesh,
        placements=SINGLE_PLACEMENTS_1D,
        shape=out_shape,
        stride=out_stride,
    )


def _outer_cdist_1d(
    x: DTensor,
    device_mesh: DeviceMesh,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Compute pairwise L2 distance matrix for 1D CP producing row-slab pair.

    Parameters
    ----------
    x : DTensor
        ``[B, N, 3]`` with placements ``(Shard(0), Shard(1))``.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : ProcessGroup
        The cp process group.

    Returns
    -------
    DTensor
        Distance matrix ``[B, N, N]`` with placements ``(Shard(0), Shard(1))``.
    """
    cp_size = device_mesh.shape[1]
    x_local = x.to_local()  # [B_local, N/cp, 3]
    x_full = all_gather_on_cp(x_local, dim=1, cp_group=cp_group, cp_size=cp_size)
    # [B_local, N, 3]

    d_local = torch.cdist(x_local, x_full)  # [B_local, N/cp, N]

    B_global = x.shape[0]
    N_global = x.shape[1]
    output_shape = torch.Size([B_global, N_global, N_global])
    output_stride = update_exhaustive_strides(d_local.shape, d_local.stride(), output_shape)

    return DTensor.from_local(
        d_local,
        device_mesh=device_mesh,
        placements=PAIR_PLACEMENTS_1D,
        shape=output_shape,
        stride=output_stride,
    )


def _transpose_via_all_gather(
    z_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """Transpose a row-slab pair tensor using all_gather + local transpose + slice.

    Gloo-compatible fallback for the all_to_all variant.

    Input:  z_local [B, N_row, N_full, C] — this rank's row slab
    Output: [B, N_row, N_full, C] — this rank's row slab of z^T

    Communication: one all_gather (N^2*C total across ranks).
    """
    if cp_size == 1:
        return z_local.transpose(1, 2)

    # All-gather row slabs from all ranks -> full [B, N, N, C]
    gathered = [torch.empty_like(z_local) for _ in range(cp_size)]
    dist.all_gather(gathered, z_local.contiguous(), group=cp_group)
    z_full = torch.cat(gathered, dim=1)  # [B, N, N, C]

    # Transpose the full pair matrix
    z_full_t = z_full.transpose(1, 2).contiguous()

    # Slice out this rank's row slab of the transposed matrix
    cp_rank = dist.get_rank(cp_group)
    n_row = z_local.shape[1]
    return z_full_t[:, cp_rank * n_row : (cp_rank + 1) * n_row, :, :]


def _transpose_via_all_to_all(
    z_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """Transpose a row-slab pair tensor using all_to_all + local transpose.

    NCCL-backend path. Mathematically identical to ``_transpose_via_all_gather`` but
    Pareto-better at ``cp_size >= 2``: each rank sends/receives only the ``1/cp_size``
    column chunk it needs, so peak extra memory is ``(P-1) * (N/P)^2 * C`` vs the
    all_gather's ``(P-1) * N^2/P * C`` (P× less), and total bytes on wire scale by
    the same factor.

    Input:  z_local [B, N_row, N_full, C] — this rank's row slab
    Output: [B, N_row, N_full, C] — this rank's row slab of z^T

    Communication: one all_to_all (N^2*C / cp_size total per rank).
    """
    if cp_size == 1:
        return z_local.transpose(1, 2).contiguous()

    _, _, n_full, _ = z_local.shape
    if n_full % cp_size != 0:
        raise ValueError(f"Uneven sharding: column axis size n_full ({n_full}) not divisible by cp_size ({cp_size})")
    n_col_chunk = n_full // cp_size

    # Each rank r should receive column chunk r from every other rank's row slab.
    # Send: split our row slab into cp_size column chunks; chunk r goes to rank r.
    # Recv: gather the column chunk assigned to this rank from every rank.
    send_list = [z_local[:, :, r * n_col_chunk : (r + 1) * n_col_chunk, :].contiguous() for r in range(cp_size)]
    recv_list = [torch.empty_like(send_list[0]) for _ in range(cp_size)]
    dist.all_to_all(recv_list, send_list, group=cp_group)

    # Concatenate received chunks along the row axis to form this rank's column slab
    # of the global pair tensor: shape [B, N_full, N_row, C].
    z_col_slab = torch.cat(recv_list, dim=1)
    del recv_list  # free P chunks before transpose allocation to bound peak memory

    # Local transpose -> this rank's row slab of z^T: [B, N_row, N_full, C].
    return z_col_slab.transpose(1, 2).contiguous()


def _transpose_slab(
    z_local: torch.Tensor,
    cp_group: dist.ProcessGroup,
    cp_size: int,
) -> torch.Tensor:
    """Dispatch to the all_to_all (NCCL) or all_gather (Gloo) row-slab transpose.

    Centralising the backend decision here keeps the forward and backward paths of
    ``_RowSlabTransposeAdd1D`` in lockstep — a single change updates both call sites.
    """
    backend = dist.get_backend(cp_group)
    if backend == "nccl":
        return _transpose_via_all_to_all(z_local, cp_group, cp_size)
    return _transpose_via_all_gather(z_local, cp_group, cp_size)


class _RowSlabTransposeAdd1D(torch.autograd.Function):
    """Autograd-safe ``z + z^T`` for row-slab pair tensors under 1D CP.

    Forward: ``result = z + transpose(z)`` via all-to-all (NCCL) or all-gather (Gloo).
    Backward: ``grad_z = grad + transpose(grad)`` (same operation on gradient).

    Communication budget:
      Forward: 1 all-to-all (NCCL) / 1 all-gather (Gloo) over cp (when cp > 1).
      Backward: 1 all-to-all (NCCL) / 1 all-gather (Gloo) over cp (when cp > 1).
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        z: DTensor,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> DTensor:
        if not isinstance(z, DTensor):
            raise TypeError(f"z must be a DTensor, got {type(z)}")
        if z.placements != PAIR_PLACEMENTS_1D:
            raise ValueError(f"z must have placements {PAIR_PLACEMENTS_1D}, got {z.placements}")
        # Even-sharding checks
        dp_size = device_mesh.shape[0]
        cp_size = device_mesh.shape[1]
        if z.shape[0] % dp_size != 0:
            raise ValueError(f"Uneven sharding: z dim 0 ({z.shape[0]}) not divisible by dp_size ({dp_size})")
        if z.shape[1] % cp_size != 0:
            raise ValueError(f"Uneven sharding: z dim 1 ({z.shape[1]}) not divisible by cp_size ({cp_size})")

        z_local = z.to_local()

        z_t_local = _transpose_slab(z_local, cp_group, cp_size)
        result_local = z_local + z_t_local

        if z.requires_grad:
            ctx.device_mesh = device_mesh
            ctx.cp_group = cp_group
            ctx.cp_size = cp_size
            ctx.z_placements = z.placements
            ctx.z_shape = z.shape
            ctx.z_stride = z.stride()

        return DTensor.from_local(
            result_local,
            device_mesh=device_mesh,
            placements=z.placements,
            shape=z.shape,
            stride=z.stride(),
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        grad_local = grad_output.to_local() if isinstance(grad_output, DTensor) else grad_output

        # d(z + z^T)/dz applied to grad = grad + grad^T
        grad_t_local = _transpose_slab(grad_local, ctx.cp_group, ctx.cp_size)
        grad_z_local = grad_local + grad_t_local

        grad_z = DTensor.from_local(
            grad_z_local,
            device_mesh=ctx.device_mesh,
            placements=ctx.z_placements,
            shape=ctx.z_shape,
            stride=ctx.z_stride,
        )
        return grad_z, None, None


def _row_slab_transpose_add_1d(
    z: DTensor,
    device_mesh: DeviceMesh,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Compute ``z + z^T`` for row-slab pair tensor under 1D CP.

    For row-slab ``[B, N/cp, N, C]``, the transpose ``z^T`` has layout
    ``[B, N/cp, N, C]`` where row i gets column i from all ranks.
    This requires an all-gather to exchange blocks.

    Parameters
    ----------
    z : DTensor
        ``[B, N, N, C]`` with placements ``(Shard(0), Shard(1))``.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : ProcessGroup
        The cp process group.

    Returns
    -------
    DTensor
        ``z + z^T`` with same shape and placements.

    Communication budget:
      Forward: 1 all-gather over cp (when cp > 1).
      Backward: 1 all-gather over cp (when cp > 1).
    """
    return _RowSlabTransposeAdd1D.apply(z, device_mesh, cp_group)


class _OuterProduct1D(torch.autograd.Function):
    """Autograd function for 1D CP outer product producing row-slab pair repr.

    Computes ``z_out[b, i, j, c] = z1[b, i, c] * z2[b, j, c]`` in row-slab format.

    Input placements (on 2D mesh ``(dp, cp)``):
        z1: ``(Shard(0), Shard(1))``  -- single repr ``[B, N, C]``
        z2: ``(Shard(0), Shard(1))``  -- single repr ``[B, N, C]``

    Output placement:
        z_out: ``(Shard(0), Shard(1))`` -- pair repr ``[B, N, N, C]`` (row-slab)

    Forward:
        1. All-gather z2 along cp -> z2_full ``[B, N, C]``
        2. z_out = z1_local[:, :, None, :] * z2_full[:, None, :, :]

    Backward:
        1. All-gather z2 along cp (recomputed from saved z2_local for O(N/cp) memory)
        2. grad_z1 = sum(grad_output * z2_full, dim=2) -- local
        3. grad_z2_full = sum(grad_output * z1_local, dim=1) -- [B_local, N, C]
        4. Reduce-scatter grad_z2_full along cp -> [B_local, N/cp, C]

    Communication budget:
      Forward: 1 all-gather of z2 along cp.
      Backward: 1 all-gather of z2 along cp (recomputed), 1 reduce-scatter of grad_z2 along cp.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        z1: DTensor,
        z2: DTensor,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ) -> DTensor:
        if not isinstance(z1, DTensor):
            raise TypeError(f"z1 must be a DTensor, got {type(z1)}")
        if not isinstance(z2, DTensor):
            raise TypeError(f"z2 must be a DTensor, got {type(z2)}")
        if z1.placements != SINGLE_PLACEMENTS_1D:
            raise ValueError(f"z1 must have placements {SINGLE_PLACEMENTS_1D}, got {z1.placements}")
        if z2.placements != SINGLE_PLACEMENTS_1D:
            raise ValueError(f"z2 must have placements {SINGLE_PLACEMENTS_1D}, got {z2.placements}")
        if z1.device_mesh != z2.device_mesh:
            raise ValueError("z1 and z2 must share the same device_mesh")
        # Even-sharding checks
        dp_size = device_mesh.shape[0]
        cp_size = device_mesh.shape[1]
        if z1.shape[0] % dp_size != 0:
            raise ValueError(f"Uneven sharding: z1 dim 0 ({z1.shape[0]}) not divisible by dp_size ({dp_size})")
        if z1.shape[1] % cp_size != 0:
            raise ValueError(f"Uneven sharding: z1 dim 1 ({z1.shape[1]}) not divisible by cp_size ({cp_size})")

        z1_local = z1.to_local()
        z2_local = z2.to_local()

        z2_full = all_gather_on_cp(z2_local, dim=1, cp_group=cp_group, cp_size=cp_size)

        output_local = z1_local.unsqueeze(2) * z2_full.unsqueeze(1)
        # [B_local, N/cp, N, C]

        B_global = z1.shape[0]
        N_global = z1.shape[1]
        output_shape = torch.Size([B_global, N_global, N_global, z1.shape[2]])
        output_stride = update_exhaustive_strides(output_local.shape, output_local.stride(), output_shape)

        # Save for backward -- save z2_local (O(N/cp)) not z2_full (O(N))
        # to keep backward memory at O(N/cp); recompute all-gather in backward.
        ctx.save_for_backward(z1_local, z2_local)
        ctx.device_mesh = device_mesh
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.z1_shape = z1.shape
        ctx.z1_stride = z1.stride()
        ctx.z2_shape = z2.shape
        ctx.z2_stride = z2.stride()
        ctx.z1_requires_grad = z1.requires_grad
        ctx.z2_requires_grad = z2.requires_grad

        return DTensor.from_local(
            output_local,
            device_mesh=device_mesh,
            placements=PAIR_PLACEMENTS_1D,
            shape=output_shape,
            stride=output_stride,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        z1_local, z2_local = ctx.saved_tensors
        go_local = grad_output.to_local() if isinstance(grad_output, DTensor) else grad_output

        # Recompute z2_full via all-gather from saved z2_local (O(N/cp) memory savings)
        z2_full = all_gather_on_cp(z2_local, dim=1, cp_group=ctx.cp_group, cp_size=ctx.cp_size)

        grad_z1 = None
        grad_z2 = None

        if ctx.z1_requires_grad:
            # grad_z1[b, i, c] = sum_j(grad[b, i, j, c] * z2_full[b, j, c])
            grad_z1_local = (go_local * z2_full.unsqueeze(1)).sum(dim=2)
            grad_z1 = DTensor.from_local(
                grad_z1_local,
                device_mesh=ctx.device_mesh,
                placements=SINGLE_PLACEMENTS_1D,
                shape=ctx.z1_shape,
                stride=ctx.z1_stride,
            )

        if ctx.z2_requires_grad:
            # grad_z2_full[b, j, c] = sum_i(grad[b, i, j, c] * z1_local[b, i, c])
            grad_z2_full = (go_local * z1_local.unsqueeze(2)).sum(dim=1)
            # Reduce-scatter along cp to get grad_z2_local
            if ctx.cp_size == 1:
                grad_z2_local = grad_z2_full
            else:
                grad_z2_chunks = [c.contiguous() for c in grad_z2_full.chunk(ctx.cp_size, dim=1)]
                grad_z2_local = torch.empty_like(grad_z2_chunks[0])
                dist.reduce_scatter(grad_z2_local, grad_z2_chunks, op=dist.ReduceOp.SUM, group=ctx.cp_group)
            grad_z2 = DTensor.from_local(
                grad_z2_local,
                device_mesh=ctx.device_mesh,
                placements=SINGLE_PLACEMENTS_1D,
                shape=ctx.z2_shape,
                stride=ctx.z2_stride,
            )

        return grad_z1, grad_z2, None, None


def _outer_product_1d(
    z1: DTensor,
    z2: DTensor,
    device_mesh: DeviceMesh,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Compute elementwise outer product of two single representations for 1D CP.

    Produces ``z_out[b, i, j, c] = z1[b, i, c] * z2[b, j, c]`` in row-slab format.

    Parameters
    ----------
    z1, z2 : DTensor
        ``[B, N, C]`` with placements ``(Shard(0), Shard(1))``.
    device_mesh : DeviceMesh
        2D mesh ``(dp, cp)``.
    cp_group : ProcessGroup
        The cp process group.

    Returns
    -------
    DTensor
        ``[B, N, N, C]`` with placements ``(Shard(0), Shard(1))``.

    Communication budget:
      Forward: 1 all-gather of z2 along cp.
      Backward: 1 reduce-scatter of grad_z2 along cp.
    """
    return _OuterProduct1D.apply(z1, z2, device_mesh, cp_group)


class ConfidenceHeads1D(nn.Module):
    """1D CP confidence heads for 2D mesh ``(dp, cp)``.

    Wraps the serial ``ConfidenceHeads`` layer, distributing parameters with
    ``LinearParamsReplicated`` and adding 1D-sharded metric computation.

    Only ``token_level_confidence=True`` is supported.

    Placements:
    - s input: ``(Shard(0), Shard(1))`` — single ``[B*mult, N, D_s]``
    - z input: ``(Shard(0), Shard(1))`` — pair ``[B*mult, N, N, D_z]``
    - d input: ``(Shard(0), Shard(1))`` — distance ``[B*mult, N, N]``
    - logit outputs: same placements as inputs
    - scalar metrics: ``(Shard(0), Replicate())``
    """

    def __init__(
        self,
        layer: SerialConfidenceHeads,
        device_mesh: DeviceMesh,
        cp_group: dist.ProcessGroup,
    ):
        super().__init__()

        if not layer.token_level_confidence:
            raise NotImplementedError(
                "ConfidenceHeads1D only supports token_level_confidence=True. "
                "The atom-level confidence path is not implemented for 1D CP."
            )

        self.device_mesh = device_mesh
        self.cp_group = cp_group
        self.token_level_confidence = layer.token_level_confidence
        self.use_separate_heads = layer.use_separate_heads

        # PAE / PDE heads
        if self.use_separate_heads:
            self.to_pae_intra_logits = LinearParamsReplicated(layer.to_pae_intra_logits, device_mesh)
            self.to_pae_inter_logits = LinearParamsReplicated(layer.to_pae_inter_logits, device_mesh)
            self.to_pde_intra_logits = LinearParamsReplicated(layer.to_pde_intra_logits, device_mesh)
            self.to_pde_inter_logits = LinearParamsReplicated(layer.to_pde_inter_logits, device_mesh)
        else:
            self.to_pae_logits = LinearParamsReplicated(layer.to_pae_logits, device_mesh)
            self.to_pde_logits = LinearParamsReplicated(layer.to_pde_logits, device_mesh)

        # pLDDT / resolved heads
        self.to_plddt_logits = LinearParamsReplicated(layer.to_plddt_logits, device_mesh)
        self.to_resolved_logits = LinearParamsReplicated(layer.to_resolved_logits, device_mesh)

    def _validate_inputs(
        self,
        s: DTensor,
        z: DTensor,
        x_pred: DTensor,
        d: DTensor,
        feats: dict,
        pred_distogram_logits: DTensor,
    ) -> None:
        """Validate DTensor types, placements, and even sharding for all inputs."""
        for name, tensor in [("s", s), ("z", z), ("x_pred", x_pred), ("d", d)]:
            if not isinstance(tensor, DTensor):
                raise TypeError(f"Expected DTensor for {name}, got {type(tensor)}")

        expected = {
            "s": SINGLE_PLACEMENTS_1D,
            "z": PAIR_PLACEMENTS_1D,
            "x_pred": ATOM_PLACEMENTS_1D,
            "d": PAIR_PLACEMENTS_1D,
        }
        for name, tensor in [("s", s), ("z", z), ("x_pred", x_pred), ("d", d)]:
            if tensor.placements != expected[name]:
                raise ValueError(f"Expected {name} placements {expected[name]}, got {tensor.placements}")

        if not isinstance(pred_distogram_logits, DTensor):
            raise TypeError(f"Expected DTensor for pred_distogram_logits, got {type(pred_distogram_logits)}")
        if pred_distogram_logits.placements != PAIR_PLACEMENTS_1D:
            raise ValueError(
                f"Expected pred_distogram_logits placements {PAIR_PLACEMENTS_1D}, "
                f"got {pred_distogram_logits.placements}"
            )

        for key in ("token_pad_mask", "asym_id", "mol_type"):
            feat = feats[key]
            if not isinstance(feat, DTensor):
                raise TypeError(f"Expected DTensor for feats['{key}'], got {type(feat)}")
            if feat.placements != SINGLE_PLACEMENTS_1D:
                raise ValueError(f"Expected feats['{key}'] placements {SINGLE_PLACEMENTS_1D}, got {feat.placements}")

        # Even-sharding checks
        dp_size = self.device_mesh.shape[0]
        cp_size = self.device_mesh.shape[1]
        for name, tensor in [("s", s), ("z", z)]:
            if tensor.shape[0] % dp_size != 0:
                raise ValueError(
                    f"Uneven sharding: {name} dim 0 ({tensor.shape[0]}) not divisible by dp_size ({dp_size})"
                )
            if tensor.shape[1] % cp_size != 0:
                raise ValueError(
                    f"Uneven sharding: {name} dim 1 ({tensor.shape[1]}) not divisible by cp_size ({cp_size})"
                )

        N_global = feats["token_pad_mask"].shape[1]
        if s.shape[1] != N_global:
            raise ValueError(f"Token dim mismatch: s.shape[1]={s.shape[1]} vs N_global={N_global}")
        if z.shape[1] != N_global or z.shape[2] != N_global:
            raise ValueError(
                f"Pair dims must equal N_global={N_global}, got z.shape[1]={z.shape[1]}, z.shape[2]={z.shape[2]}"
            )

    def forward(
        self,
        s: DTensor,
        z: DTensor,
        x_pred: DTensor,
        d: DTensor,
        feats: dict[str, DTensor],
        pred_distogram_logits: DTensor,
        multiplicity: int = 1,
    ) -> dict[str, DTensor]:
        """Compute confidence logits and aggregated metrics.

        Parameters
        ----------
        s : DTensor
            Single representation ``[B*mult, N, D_s]``, placements ``(Shard(0), Shard(1))``.
        z : DTensor
            Pair representation ``[B*mult, N, N, D_z]``, placements ``(Shard(0), Shard(1))``.
        x_pred : DTensor
            Predicted atom coordinates ``[B*mult, N_atoms, 3]``, placements
            ``ATOM_PLACEMENTS_1D == (Shard(0), Shard(1))``.  Atoms ARE sharded
            across cp, mirroring the data-pipeline placement of ``atom_to_token``
            in ``INFERENCE_FEATURE_PLACEMENTS_1D``.
        d : DTensor
            Token-level distance matrix ``[B*mult, N, N]``, placements ``(Shard(0), Shard(1))``.
        feats : dict[str, DTensor]
            Feature dictionary with ``token_pad_mask``, ``asym_id``, ``mol_type``.
        pred_distogram_logits : DTensor
            Predicted distogram logits ``[B, N, N, bins]`` or ``[B, N, N, K, bins]``.
        multiplicity : int
            Number of diffusion samples per input.

        Returns
        -------
        dict[str, DTensor]
        """
        self._validate_inputs(s, z, x_pred, d, feats, pred_distogram_logits)

        plddt_logits = self.to_plddt_logits(s)
        resolved_logits = self.to_resolved_logits(s)

        # Build same_chain mask for separate heads and iPLDDT
        with torch.no_grad():
            asym_id_local = feats["asym_id"].to_local()  # [B_local, N_local]
            cp_size = self.device_mesh.shape[1]
            asym_id_full = all_gather_on_cp(asym_id_local, 1, self.cp_group, cp_size)
            same_chain_base = torch.eq(asym_id_local[:, :, None], asym_id_full[:, None, :])  # [B_local, N_row, N_full]

        if self.use_separate_heads:
            same_chain = same_chain_base.repeat_interleave(multiplicity, dim=0) if multiplicity > 1 else same_chain_base

            pae_intra = self.to_pae_intra_logits(z)
            pae_inter = self.to_pae_inter_logits(z)

            # Build DTensor masks matching pae logits shape to preserve gradient flow.
            # same_chain is [B_local, N_local, N_full]; expand to [B_local, N_local, N_full, C].
            pae_num_bins = pae_intra.shape[-1]
            same_chain_pae_local = same_chain.unsqueeze(-1).expand(-1, -1, -1, pae_num_bins).float()
            not_same_chain_pae_local = (~same_chain).unsqueeze(-1).expand(-1, -1, -1, pae_num_bins).float()
            same_chain_pae_dt = DTensor.from_local(
                same_chain_pae_local,
                device_mesh=self.device_mesh,
                placements=pae_intra.placements,
                shape=pae_intra.shape,
                stride=update_exhaustive_strides(
                    same_chain_pae_local.shape, same_chain_pae_local.stride(), pae_intra.shape
                ),
            )
            not_same_chain_pae_dt = DTensor.from_local(
                not_same_chain_pae_local,
                device_mesh=self.device_mesh,
                placements=pae_intra.placements,
                shape=pae_intra.shape,
                stride=update_exhaustive_strides(
                    not_same_chain_pae_local.shape, not_same_chain_pae_local.stride(), pae_intra.shape
                ),
            )
            pae_logits = elementwise_op(
                elementwise_op(pae_intra, same_chain_pae_dt, ElementwiseOp.PROD),
                elementwise_op(pae_inter, not_same_chain_pae_dt, ElementwiseOp.PROD),
                ElementwiseOp.SUM,
            )

            pde_intra = _row_slab_transpose_add_1d(self.to_pde_intra_logits(z), self.device_mesh, self.cp_group)
            pde_inter = _row_slab_transpose_add_1d(self.to_pde_inter_logits(z), self.device_mesh, self.cp_group)

            # Build DTensor masks matching pde logits shape.
            pde_num_bins = pde_intra.shape[-1]
            same_chain_pde_local = same_chain.unsqueeze(-1).expand(-1, -1, -1, pde_num_bins).float()
            not_same_chain_pde_local = (~same_chain).unsqueeze(-1).expand(-1, -1, -1, pde_num_bins).float()
            same_chain_pde_dt = DTensor.from_local(
                same_chain_pde_local,
                device_mesh=self.device_mesh,
                placements=pde_intra.placements,
                shape=pde_intra.shape,
                stride=update_exhaustive_strides(
                    same_chain_pde_local.shape, same_chain_pde_local.stride(), pde_intra.shape
                ),
            )
            not_same_chain_pde_dt = DTensor.from_local(
                not_same_chain_pde_local,
                device_mesh=self.device_mesh,
                placements=pde_intra.placements,
                shape=pde_intra.shape,
                stride=update_exhaustive_strides(
                    not_same_chain_pde_local.shape, not_same_chain_pde_local.stride(), pde_intra.shape
                ),
            )
            pde_logits = elementwise_op(
                elementwise_op(pde_intra, same_chain_pde_dt, ElementwiseOp.PROD),
                elementwise_op(pde_inter, not_same_chain_pde_dt, ElementwiseOp.PROD),
                ElementwiseOp.SUM,
            )
        else:
            pae_logits = self.to_pae_logits(z)
            pde_logits = _row_slab_transpose_add_1d(self.to_pde_logits(z), self.device_mesh, self.cp_group)

        out_dict: dict[str, DTensor] = {
            "plddt_logits": plddt_logits,
            "pde_logits": pde_logits,
            "resolved_logits": resolved_logits,
            "pae_logits": pae_logits,
        }

        # No-grad aggregated metrics (inference / logging only)
        with torch.no_grad():
            self._compute_aggregated_metrics(
                out_dict,
                plddt_logits,
                pde_logits,
                pae_logits,
                pred_distogram_logits,
                d,
                x_pred,
                feats,
                same_chain_base,
                multiplicity,
            )

        return out_dict

    def _compute_aggregated_metrics(
        self,
        out_dict: dict[str, DTensor],
        plddt_logits: DTensor,
        pde_logits: DTensor,
        pae_logits: DTensor,
        pred_distogram_logits: DTensor,
        d: DTensor,
        x_pred: DTensor,
        feats: dict[str, DTensor],
        same_chain_base: torch.Tensor,
        multiplicity: int,
    ) -> None:
        """Compute no-grad aggregated metrics and add to out_dict."""
        token_pad_mask = feats["token_pad_mask"]
        mask_local = token_pad_mask.to_local()  # [B_local, N_local]
        B_local = mask_local.shape[0]
        N_local = mask_local.shape[1]
        cp_size = self.device_mesh.shape[1]

        # pLDDT
        plddt = compute_aggregated_metric(plddt_logits)
        plddt_local = plddt.to_local()  # [B_local*mult, N_local]
        plddt_reshaped = plddt_local.reshape(B_local, multiplicity, N_local)

        masked_plddt = plddt_reshaped * mask_local.unsqueeze(1)
        num_local = masked_plddt.sum(dim=-1)  # [B_local, mult]
        den_local = mask_local.sum(dim=-1, keepdim=True)  # [B_local, 1]

        dist.all_reduce(num_local, op=dist.ReduceOp.SUM, group=self.cp_group)
        dist.all_reduce(den_local, op=dist.ReduceOp.SUM, group=self.cp_group)

        complex_plddt_local = (num_local / den_local).reshape(B_local * multiplicity)
        complex_plddt = DTensor.from_local(
            complex_plddt_local,
            device_mesh=self.device_mesh,
            placements=SCALAR_PLACEMENTS_1D,
            shape=(plddt.shape[0],),
            stride=(1,),
        )

        # iPLDDT
        mol_type_local = feats["mol_type"].to_local()
        is_ligand_local = (mol_type_local == const.chain_type_ids["NONPOLYMER"]).float()

        d_local = d.to_local()  # [B_local*mult, N_row, N_full]
        is_contact_local = (d_local < 8).float()

        is_diff_chain_local = (~same_chain_base).float()

        N_row = d_local.shape[1]
        N_full = d_local.shape[2]
        is_contact_4d = is_contact_local.reshape(B_local, multiplicity, N_row, N_full)
        is_diff_chain_4d = is_diff_chain_local.unsqueeze(1)
        non_ligand_4d = (1 - is_ligand_local).unsqueeze(1).unsqueeze(-1)

        interface_product = is_contact_4d * is_diff_chain_4d * non_ligand_4d
        # Max over the full column dimension. Each rank already holds all N_full
        # columns in its row-slab, so no cross-rank reduction is needed.
        token_interface_mask_local = interface_product.max(dim=-1).values  # [B_local, mult, N_row]

        is_ligand_3d = is_ligand_local.unsqueeze(1)
        token_non_interface_mask = (1 - token_interface_mask_local) * (1 - is_ligand_3d)
        iplddt_weight_local = (
            is_ligand_3d * IPLDDT_LIGAND_WEIGHT
            + token_interface_mask_local * IPLDDT_INTERFACE_WEIGHT
            + token_non_interface_mask * IPLDDT_NON_INTERFACE_WEIGHT
        )

        masked_iplddt_w = mask_local.unsqueeze(1) * iplddt_weight_local
        num_iplddt = (plddt_reshaped * masked_iplddt_w).sum(dim=-1)
        den_iplddt = masked_iplddt_w.sum(dim=-1)

        dist.all_reduce(num_iplddt, op=dist.ReduceOp.SUM, group=self.cp_group)
        dist.all_reduce(den_iplddt, op=dist.ReduceOp.SUM, group=self.cp_group)

        complex_iplddt_local = (num_iplddt / den_iplddt).reshape(B_local * multiplicity)
        complex_iplddt = DTensor.from_local(
            complex_iplddt_local,
            device_mesh=self.device_mesh,
            placements=SCALAR_PLACEMENTS_1D,
            shape=(plddt.shape[0],),
            stride=(1,),
        )

        # PDE / iPDE
        pde = compute_aggregated_metric(pde_logits, end=32)

        pred_disto_local = pred_distogram_logits.to_local()
        if pred_disto_local.ndim == 5:  # noqa: PLR2004
            if pred_disto_local.shape[-2] != 1:
                raise ValueError(f"ConfidenceHeads1D expects num_distograms=1, got shape {pred_disto_local.shape}")
            pred_disto_local = pred_disto_local.squeeze(-2)
        pred_disto_prob = torch.softmax(pred_disto_local, dim=-1)
        contacts_mask = torch.zeros((1, 1, 1, 64), dtype=pred_disto_prob.dtype, device=pred_disto_prob.device)
        contacts_mask[:, :, :, :20] = 1.0
        prob_contact_local = (pred_disto_prob * contacts_mask).sum(-1)
        # prob_contact_local: [B_local, N_row, N_full]

        pde_local = pde.to_local().reshape(B_local, multiplicity, N_row, N_full)

        # Column mask: all-gather token_pad_mask across cp
        col_mask = all_gather_on_cp(mask_local, 1, self.cp_group, cp_size)  # [B_local, N_full]

        prob_contact_local = prob_contact_local * mask_local[:, :, None] * col_mask[:, None, :]

        # Zero diagonal on self-block
        cp_rank = dist.get_rank(self.cp_group)
        diag_start = cp_rank * N_local
        diag_idx_row = torch.arange(N_local, device=prob_contact_local.device)
        diag_idx_col = diag_idx_row + diag_start
        prob_contact_local[:, diag_idx_row, diag_idx_col] = 0

        prob_contact_4d = prob_contact_local.unsqueeze(1)  # [B_local, 1, N_row, N_full]

        num_pde = (pde_local * prob_contact_4d).sum(dim=(2, 3))
        den_pde = prob_contact_4d.sum(dim=(2, 3))

        dist.all_reduce(num_pde, op=dist.ReduceOp.SUM, group=self.cp_group)
        dist.all_reduce(den_pde, op=dist.ReduceOp.SUM, group=self.cp_group)

        complex_pde_local = (num_pde / den_pde).reshape(B_local * multiplicity)
        complex_pde = DTensor.from_local(
            complex_pde_local,
            device_mesh=self.device_mesh,
            placements=SCALAR_PLACEMENTS_1D,
            shape=(pde.shape[0],),
            stride=(1,),
        )

        # iPDE
        is_diff_chain_4d_contact = is_diff_chain_local.unsqueeze(1)
        token_intf_pair = prob_contact_4d * is_diff_chain_4d_contact
        num_ipde = (pde_local * token_intf_pair).sum(dim=(2, 3))
        den_ipde = token_intf_pair.sum(dim=(2, 3))

        dist.all_reduce(num_ipde, op=dist.ReduceOp.SUM, group=self.cp_group)
        dist.all_reduce(den_ipde, op=dist.ReduceOp.SUM, group=self.cp_group)

        complex_ipde_local = (num_ipde / (den_ipde + _EPS)).reshape(B_local * multiplicity)
        complex_ipde = DTensor.from_local(
            complex_ipde_local,
            device_mesh=self.device_mesh,
            placements=SCALAR_PLACEMENTS_1D,
            shape=(pde.shape[0],),
            stride=(1,),
        )

        # PAE
        pae = compute_aggregated_metric(pae_logits, end=32)

        out_dict["plddt"] = plddt
        out_dict["pde"] = pde
        out_dict["pae"] = pae
        out_dict["complex_plddt"] = complex_plddt
        out_dict["complex_iplddt"] = complex_iplddt
        out_dict["complex_pde"] = complex_pde
        out_dict["complex_ipde"] = complex_ipde

        # PTM / iPTM
        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm = _compute_ptms_1d(
            pae_logits,
            x_pred,
            feats,
            multiplicity,
            self.device_mesh,
            self.cp_group,
        )
        out_dict["ptm"] = ptm
        out_dict["iptm"] = iptm
        out_dict["ligand_iptm"] = ligand_iptm
        out_dict["protein_iptm"] = protein_iptm
        out_dict["pair_chains_iptm"] = pair_chains_iptm


def _compute_ptms_1d(
    logits: DTensor,
    x_pred: DTensor,
    feats: dict[str, DTensor],
    multiplicity: int,
    device_mesh: DeviceMesh,
    cp_group: dist.ProcessGroup,
) -> tuple[DTensor, DTensor, DTensor, DTensor, dict[int, dict[int, DTensor]]]:
    """Compute pTM and ipTM scores for 1D CP.

    Adapts the 2D CP ``_ComputePtmsImpl`` for the simpler 1D row-slab layout.
    Under 1D CP, pair masks are built by all-gathering column data instead of
    using TransposeComm.

    Includes the collinearity mask from ``_compute_frame_pred_1d`` to match
    the serial ``compute_ptms`` which masks out tokens with degenerate frames
    (e.g. single-atom ligand chains).

    Parameters
    ----------
    logits : DTensor
        PAE logits ``[B*mult, N, N, bins]`` with placements ``(Shard(0), Shard(1))``.
    x_pred : DTensor
        Predicted atom coordinates ``[B*mult, N_atoms, 3]``.
    feats : dict[str, DTensor]
        Feature dictionary with ``token_pad_mask``, ``asym_id``, ``mol_type``,
        ``frames_idx``, ``atom_to_token``, ``atom_pad_mask``, ``atom_resolved_mask``.
    multiplicity : int
        Number of diffusion samples.
    device_mesh : DeviceMesh
        2D mesh ``(dp, cp)``.
    cp_group : ProcessGroup
        The cp process group.

    Returns
    -------
    tuple[DTensor, DTensor, DTensor, DTensor, dict]
        ptm, iptm, ligand_iptm, protein_iptm, pair_chains_iptm.
    """
    from boltz.distributed.model.loss.confidence_1d import _compute_frame_pred_1d

    cp_size = device_mesh.shape[1]

    # Compute collinearity mask via compute_frame_pred. Inference mode matches
    # the serial ``compute_ptms`` call site, which derives ``resolved_pair`` from
    # ``atom_pad_mask`` rather than ``atom_resolved_mask``.
    # frames_idx may have been flattened to 3D [B*mult, N, 3] by the caller;
    # _compute_frame_pred_1d expects [B, N, 3] (no mult dim).
    frames_idx = feats["frames_idx"]
    _, mask_collinear_dt = _compute_frame_pred_1d(
        x_pred,
        frames_idx,
        feats,
        multiplicity,
        device_mesh,
        inference=True,
    )
    # mask_collinear_dt: [B, mult, N] with placements (Shard(0), Shard(2))
    # Reshape to [B*mult, N_local] for row-slab masking
    mc_local = mask_collinear_dt.to_local()  # [B_local, mult, N_local]
    mask_collinear_local = mc_local.reshape(-1, mc_local.shape[-1])  # [B_local*mult, N_local]

    mask_pad_local = feats["token_pad_mask"].to_local().bool()  # [B_local, N_local]
    mask_pad_mult = mask_pad_local.repeat_interleave(multiplicity, dim=0)
    asym_id_local = feats["asym_id"].to_local()
    asym_id_mult = asym_id_local.repeat_interleave(multiplicity, dim=0)
    mol_type_local = feats["mol_type"].to_local()
    mol_type_mult = mol_type_local.repeat_interleave(multiplicity, dim=0)

    # All-gather column data for pair mask construction
    mask_pad_full = all_gather_on_cp(mask_pad_mult, 1, cp_group, cp_size)
    asym_id_full = all_gather_on_cp(asym_id_mult, 1, cp_group, cp_size)
    mol_type_full = all_gather_on_cp(mol_type_mult, 1, cp_group, cp_size)

    # Build pair masks: [B_local*mult, N_row, N_full]
    # mask_collinear applies to the row (alignment target) dimension, matching serial compute_ptms
    mask_collinear_bool = mask_collinear_local.bool()
    pair_mask_ptm = mask_collinear_bool[:, :, None] & mask_pad_mult[:, :, None] & mask_pad_full[:, None, :]
    pair_mask_iptm_equal = asym_id_mult[:, :, None] == asym_id_full[:, None, :]
    pair_mask_iptm = pair_mask_ptm & (~pair_mask_iptm_equal)

    is_ligand_row = mol_type_mult == const.chain_type_ids["NONPOLYMER"]
    is_protein_row = mol_type_mult == const.chain_type_ids["PROTEIN"]
    is_ligand_col = mol_type_full == const.chain_type_ids["NONPOLYMER"]
    is_protein_col = mol_type_full == const.chain_type_ids["PROTEIN"]

    ligand_iptm_mask = (
        (is_ligand_row[:, :, None] & is_protein_col[:, None, :])
        | (is_protein_row[:, :, None] & is_ligand_col[:, None, :])
    ) & pair_mask_iptm
    protein_iptm_mask = (is_protein_row[:, :, None] & is_protein_col[:, None, :]) & pair_mask_iptm

    # Compute TM expected values from logits
    logits_local = logits.to_local().detach()
    num_bins = logits_local.shape[-1]
    bin_width = 32.0 / num_bins
    compute_dtype = torch.promote_types(logits_local.dtype, torch.float32)

    # Compute n_res (global token count) in the same dtype as the TM
    # computation to avoid an implicit fp32 down-cast that loses precision
    # for fp64 testing paths.
    n_res_local = mask_pad_mult.to(compute_dtype).sum(dim=-1, keepdim=True)
    dist.all_reduce(n_res_local, op=dist.ReduceOp.SUM, group=cp_group)
    pae_value = (torch.arange(num_bins, device=logits_local.device, dtype=compute_dtype) + 0.5) * bin_width
    pae_value = pae_value.unsqueeze(0)
    tm_value = serial_tm_function(pae_value, n_res_local).unsqueeze(1).unsqueeze(2)
    probs = F.softmax(logits_local.to(compute_dtype), dim=-1)
    tm_expected_value = torch.sum(probs * tm_value, dim=-1)
    # tm_expected_value: [B_local*mult, N_row, N_full]

    # Helper to compute TM-style metric: sum(tm * mask) / sum(mask), then max over alignment target.
    # Under row-slab sharding each rank holds full N columns, so the column-sum
    # (dim=-1) is complete locally — no all-reduce needed for numerator/denominator.
    # Only the max over the row (token) dimension requires an all-reduce(MAX)
    # across cp because each rank holds only N/cp rows.
    def _compute_tm_metric(mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        mask_compute = mask_bool.to(compute_dtype)
        # Sum over column dim (full N) for each row token — local, no communication
        numerator = (tm_expected_value * mask_compute).sum(dim=-1)  # [B_local*mult, N_row]
        denominator = mask_compute.sum(dim=-1)  # [B_local*mult, N_row]
        per_token = numerator / (denominator + _EPS)  # [B_local*mult, N_row]
        # Max over alignment target (token dim) — need all-reduce over cp
        local_max = per_token.max(dim=-1).values  # [B_local*mult]
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=cp_group)
        return local_max

    ptm_local = _compute_tm_metric(pair_mask_ptm)
    iptm_local = _compute_tm_metric(pair_mask_iptm)
    ligand_local = _compute_tm_metric(ligand_iptm_mask)
    protein_local = _compute_tm_metric(protein_iptm_mask)

    # Build output DTensors
    B_mult_global = logits.shape[0]
    output_shape = torch.Size([B_mult_global])
    output_stride = (1,)
    output_placements = SCALAR_PLACEMENTS_1D

    ptm = DTensor.from_local(
        ptm_local, device_mesh=device_mesh, placements=output_placements, shape=output_shape, stride=output_stride
    )
    iptm = DTensor.from_local(
        iptm_local, device_mesh=device_mesh, placements=output_placements, shape=output_shape, stride=output_stride
    )
    ligand_iptm = DTensor.from_local(
        ligand_local, device_mesh=device_mesh, placements=output_placements, shape=output_shape, stride=output_stride
    )
    protein_iptm = DTensor.from_local(
        protein_local, device_mesh=device_mesh, placements=output_placements, shape=output_shape, stride=output_stride
    )

    # Per-chain-pair iptm
    chain_pair_iptm: dict[int, dict[int, DTensor]] = {}
    # Only consider asym_ids from non-padding tokens (padding tokens can have
    # spurious asym_id values from the data pipeline's even-sharding padding).
    mask_pad_base = feats["token_pad_mask"].to_local().bool()  # [B_local, N_local]
    local_asym_ids = set(torch.unique(asym_id_local[mask_pad_base]).tolist())

    # Gather all chain IDs across cp and dp
    cp_obj_list = [None] * dist.get_world_size(group=cp_group)
    dist.all_gather_object(cp_obj_list, local_asym_ids, group=cp_group)
    cp_asym_ids = set().union(*cp_obj_list)

    dp_group = device_mesh.get_group(0)
    dp_obj_list = [None] * dist.get_world_size(group=dp_group)
    dist.all_gather_object(dp_obj_list, cp_asym_ids, group=dp_group)
    world_asym_ids_list = sorted(set().union(*dp_obj_list))

    for idx1 in world_asym_ids_list:
        chain_iptm: dict[int, DTensor] = {}
        for idx2 in world_asym_ids_list:
            if idx1 not in cp_asym_ids or idx2 not in cp_asym_ids:
                iptm_chain_local = torch.full(
                    (mask_pad_mult.size(0),),
                    CHAIN_IPTM_SENTINEL,
                    device=mask_pad_mult.device,
                    dtype=compute_dtype,
                )
            else:
                mask_pair_chain_row = (asym_id_mult == idx2) & mask_pad_mult & mask_collinear_bool
                mask_pair_chain_col = all_gather_on_cp(
                    ((asym_id_mult == idx1) & mask_pad_mult).float(), 1, cp_group, cp_size
                ).bool()
                mask_pair_chain = mask_pair_chain_row[:, :, None] & mask_pair_chain_col[:, None, :]
                iptm_chain_local = _compute_tm_metric(mask_pair_chain)

            chain_iptm[idx2] = DTensor.from_local(
                iptm_chain_local,
                device_mesh=device_mesh,
                placements=output_placements,
                shape=output_shape,
                stride=output_stride,
            )
        chain_pair_iptm[idx1] = chain_iptm

    return ptm, iptm, ligand_iptm, protein_iptm, chain_pair_iptm


class _ContactConditioning1D(nn.Module):
    """1D CP wrapper for ContactConditioning, preserving serial parameter names.

    Wraps the serial ``ContactConditioning`` module for 1D context parallelism on
    a 2D mesh ``(dp, cp)``.  All operations are elementwise on the last dimension
    of pair features ``[B, N, N, *]``, so no cross-shard communication is needed.

    The parameter names match the serial module exactly:
    ``encoder.*``, ``encoding_unspecified``, ``encoding_unselected``.
    """

    def __init__(self, module, device_mesh: DeviceMesh) -> None:
        super().__init__()
        self.device_mesh = device_mesh

        self.fourier_embedding = module.fourier_embedding
        self.encoder = LinearParamsReplicated(module.encoder, device_mesh)

        all_replicate = [Replicate()] * device_mesh.ndim
        self.encoding_unspecified = nn.Parameter(
            distribute_tensor(module.encoding_unspecified.data, device_mesh, all_replicate),
            requires_grad=module.encoding_unspecified.requires_grad,
        )
        self.encoding_unselected = nn.Parameter(
            distribute_tensor(module.encoding_unselected.data, device_mesh, all_replicate),
            requires_grad=module.encoding_unselected.requires_grad,
        )

        self.cutoff_min = module.cutoff_min
        self.cutoff_max = module.cutoff_max

    def forward(self, feats: dict[str, DTensor]) -> DTensor:
        """Compute contact conditioning pairwise embeddings for 1D CP.

        Parameters
        ----------
        feats : dict[str, DTensor]
            Must contain ``contact_conditioning`` and ``contact_threshold``.

        Returns
        -------
        DTensor
            Pair tensor ``[B, N, N, token_z]`` with placements ``(Shard(0), Shard(1))``.
        """
        cc_dt: DTensor = feats["contact_conditioning"]
        ct_dt: DTensor = feats["contact_threshold"]

        cc_local = cc_dt.to_local()
        ct_local = ct_dt.to_local()

        ct_norm = (ct_local - self.cutoff_min) / (self.cutoff_max - self.cutoff_min)
        ct_flat = ct_norm.flatten()
        fourier_flat = torch.cos(2 * pi * self.fourier_embedding.proj(ct_flat.unsqueeze(-1)))
        ct_fourier = fourier_flat.reshape(ct_norm.shape + (-1,))

        cc_features = cc_local[:, :, :, 2:]
        combined = torch.cat(
            [cc_features, ct_norm.unsqueeze(-1), ct_fourier],
            dim=-1,
        )

        combined_shape = cc_dt.shape[:-1] + (combined.shape[-1],)
        combined_contiguous = combined.contiguous()
        combined_stride = update_exhaustive_strides(
            combined_contiguous.shape, combined_contiguous.stride(), combined_shape
        )
        combined_dt = DTensor.from_local(
            combined_contiguous,
            self.device_mesh,
            cc_dt.placements,
            shape=combined_shape,
            stride=combined_stride,
        )
        encoded_dt = self.encoder(combined_dt)

        # Apply unspecified/unselected masks using native DTensor ops.
        # Safe because all multiplications are Replicate x Shard(0,1) = local elementwise.
        unspec_flag = cc_dt[:, :, :, 0:1]
        unsel_flag = cc_dt[:, :, :, 1:2]
        mask_factor = 1.0 - (unspec_flag + unsel_flag)

        result = (
            encoded_dt * mask_factor + self.encoding_unspecified * unspec_flag + self.encoding_unselected * unsel_flag
        )
        return result


class ConfidenceModule1D(nn.Module):
    """1D CP ConfidenceModule for 2D mesh ``(dp, cp)``.

    Wraps the serial :class:`~boltz.model.modules.confidencev2.ConfidenceModule`,
    distributing submodule parameters and using 1D-specific sharded operations.

    Only ``token_level_confidence=True`` is supported.

    Parameters
    ----------
    module : SerialConfidenceModule
        Initialised serial module whose weights are wrapped / transferred.
    dist_manager : DistributedManager
        Distributed manager defining the distributed computation topology
        and groups.
    """

    def __init__(
        self,
        module: SerialConfidenceModule,
        dist_manager: "DistributedManager",
    ) -> None:
        super().__init__()

        if not module.token_level_confidence:
            raise NotImplementedError(
                "ConfidenceModule1D only supports token_level_confidence=True. "
                "The atom-level confidence path is not implemented for 1D CP."
            )

        self.dist_manager = dist_manager
        self.device_mesh = dist_manager.device_mesh
        self.cp_group = dist_manager.group["cp"]
        device_mesh = self.device_mesh
        cp_group = self.cp_group

        self.no_update_s = module.no_update_s
        self.add_s_to_z_prod = module.add_s_to_z_prod
        self.add_s_input_to_s = module.add_s_input_to_s
        self.add_z_input_to_z = module.add_z_input_to_z
        self.return_latent_feats = module.return_latent_feats

        # Buffer (plain tensor, not DTensor)
        self.register_buffer("boundaries", module.boundaries)

        # LayerNorms
        self.s_inputs_norm = LayerNormParamsReplicated(module.s_inputs_norm, device_mesh)
        if not self.no_update_s:
            self.s_norm = LayerNormParamsReplicated(module.s_norm, device_mesh)
        self.z_norm = LayerNormParamsReplicated(module.z_norm, device_mesh)

        # s -> z projections
        self.s_to_z = LinearParamsReplicated(module.s_to_z, device_mesh)
        self.s_to_z_transpose = LinearParamsReplicated(module.s_to_z_transpose, device_mesh)

        if self.add_s_to_z_prod:
            self.s_to_z_prod_in1 = LinearParamsReplicated(module.s_to_z_prod_in1, device_mesh)
            self.s_to_z_prod_in2 = LinearParamsReplicated(module.s_to_z_prod_in2, device_mesh)
            self.s_to_z_prod_out = LinearParamsReplicated(module.s_to_z_prod_out, device_mesh)

        # Optional s_input -> s
        if self.add_s_input_to_s:
            self.s_input_to_s = LinearParamsReplicated(module.s_input_to_s, device_mesh)

        # Optional z-input conditioning
        if self.add_z_input_to_z:
            # Shared 1D-CP relative-position encoder (same impl as the trunk).
            # Preserves the serial param path (rel_pos.linear_layer.*).
            self.rel_pos = RelativePositionEncoder1D(module.rel_pos, device_mesh, cp_group)
            self.token_bonds = LinearParamsReplicated(module.token_bonds, device_mesh)
            self.bond_type_feature = getattr(module, "bond_type_feature", False)
            if self.bond_type_feature:
                self.token_bonds_type = EmbeddingParamsReplicated(module.token_bonds_type, device_mesh)
            # ContactConditioning: wrap as submodule to match serial param names
            # (contact_conditioning.encoder.*, contact_conditioning.encoding_unspecified, etc.)
            cc = _ContactConditioning1D(module.contact_conditioning, device_mesh)
            self.contact_conditioning = cc

        # Distogram embedding
        self.dist_bin_pairwise_embed = EmbeddingParamsReplicated(module.dist_bin_pairwise_embed, device_mesh)

        # Pairformer
        self.pairformer_stack = PairformerModule1D(module.pairformer_stack, dist_manager)

        # Confidence heads
        self.confidence_heads = ConfidenceHeads1D(module.confidence_heads, device_mesh, cp_group)

    def _contact_conditioning_1d(self, feats: dict[str, DTensor]) -> DTensor:
        """Compute contact conditioning pairwise embeddings for 1D CP.

        Delegates to the ``_ContactConditioning1D`` submodule which preserves
        the serial parameter naming path (``contact_conditioning.*``).
        """
        return self.contact_conditioning(feats)

    def _validate_inputs(
        self,
        s_inputs: DTensor,
        s: DTensor,
        z: DTensor,
        x_pred: DTensor,
    ) -> None:
        """Validate DTensor types, placements, and even sharding for primary inputs.

        Note: ``x_pred`` has atom-level shape ``[B*mult, N_atoms, 3]`` with
        placements ``ATOM_PLACEMENTS_1D`` (``(Shard(0), Shard(1))``), matching
        the producer placements emitted by the 1D-CP diffusion module. The
        expected-placements dict at the call site below uses
        ``ATOM_PLACEMENTS_1D`` directly.
        """
        for name, tensor in [("s_inputs", s_inputs), ("s", s), ("z", z), ("x_pred", x_pred)]:
            if not isinstance(tensor, DTensor):
                raise TypeError(f"Expected DTensor for {name}, got {type(tensor)}")

        expected = {
            "s_inputs": SINGLE_PLACEMENTS_1D,
            "s": SINGLE_PLACEMENTS_1D,
            "z": PAIR_PLACEMENTS_1D,
            "x_pred": ATOM_PLACEMENTS_1D,
        }
        for name, tensor in [("s_inputs", s_inputs), ("s", s), ("z", z), ("x_pred", x_pred)]:
            if tensor.placements != expected[name]:
                raise ValueError(f"Expected {name} placements {expected[name]}, got {tensor.placements}")

        # Even-sharding checks
        dp_size = self.device_mesh.shape[0]
        cp_size = self.device_mesh.shape[1]
        for name, tensor in [("s_inputs", s_inputs), ("s", s), ("z", z), ("x_pred", x_pred)]:
            if tensor.shape[0] % dp_size != 0:
                raise ValueError(
                    f"Uneven sharding: {name} dim 0 ({tensor.shape[0]}) not divisible by dp_size ({dp_size})"
                )
            if tensor.shape[1] % cp_size != 0:
                raise ValueError(
                    f"Uneven sharding: {name} dim 1 ({tensor.shape[1]}) not divisible by cp_size ({cp_size})"
                )

    def forward(
        self,
        s_inputs: DTensor,
        s: DTensor,
        z: DTensor,
        x_pred: DTensor,
        feats: dict[str, DTensor],
        pred_distogram_logits: DTensor,
        multiplicity: int = 1,
        run_sequentially: bool = False,
    ) -> dict[str, DTensor]:
        """Forward pass through the 1D CP confidence module.

        Parameters
        ----------
        s_inputs : DTensor
            Input single representation ``[B, N, D_s]``, placements ``(Shard(0), Shard(1))``.
        s : DTensor
            Trunk single representation ``[B, N, D_s]``, same placements.
        z : DTensor
            Trunk pair representation ``[B, N, N, D_z]``, placements ``(Shard(0), Shard(1))``.
        x_pred : DTensor
            Predicted atom coordinates ``[B*mult, N_atoms, 3]``, placements
            ``ATOM_PLACEMENTS_1D == (Shard(0), Shard(1))``.  Atoms ARE sharded
            across cp, mirroring the data-pipeline placement of ``atom_to_token``
            in ``INFERENCE_FEATURE_PLACEMENTS_1D``.
        feats : dict[str, DTensor]
            Feature dictionary.
        pred_distogram_logits : DTensor
            Predicted distogram logits.
        multiplicity : int
            Number of diffusion samples.
        run_sequentially : bool
            If True and multiplicity > 1, run each sample one at a time.

        Returns
        -------
        dict[str, DTensor]
            Confidence outputs.
        """
        self._validate_inputs(s_inputs, s, z, x_pred)

        if run_sequentially and multiplicity > 1:
            return self._forward_sequentially(s_inputs, s, z, x_pred, feats, pred_distogram_logits, multiplicity)

        # 1. Normalize inputs
        s_inputs = self.s_inputs_norm(s_inputs)
        if not self.no_update_s:
            s = self.s_norm(s)

        # 2. Optional s_input addition to s
        if self.add_s_input_to_s:
            s = elementwise_op(s, self.s_input_to_s(s_inputs), ElementwiseOp.SUM)

        # 3. Normalize z
        z = self.z_norm(z)

        # 4. Optional z-input conditioning
        if self.add_z_input_to_z:
            rel_pos = self.rel_pos(feats)
            z = elementwise_op(z, rel_pos, ElementwiseOp.SUM)

            safe_dtype = z.dtype if z.dtype.is_floating_point else torch.float32
            z = elementwise_op(
                z,
                self.token_bonds(feats["token_bonds"].to(dtype=safe_dtype)),
                ElementwiseOp.SUM,
            )
            if self.bond_type_feature:
                z = elementwise_op(
                    z,
                    self.token_bonds_type(feats["type_bonds"].long()),
                    ElementwiseOp.SUM,
                )
            z = elementwise_op(z, self._contact_conditioning_1d(feats), ElementwiseOp.SUM)

        # 5. Repeat s for multiplicity
        s = shardwise_repeat_interleave(s, multiplicity, dim=0)

        # 6. Outer-sum s -> z
        s_to_z_pair = outer_sum_1d(
            self.s_to_z(s_inputs),
            self.s_to_z_transpose(s_inputs),
            self.device_mesh,
            self.cp_group,
        )
        z = elementwise_op(z, s_to_z_pair, ElementwiseOp.SUM)

        # 7. Optional outer-product s -> z
        if self.add_s_to_z_prod:
            z_prod = _outer_product_1d(
                self.s_to_z_prod_in1(s_inputs),
                self.s_to_z_prod_in2(s_inputs),
                self.device_mesh,
                self.cp_group,
            )
            z = elementwise_op(z, self.s_to_z_prod_out(z_prod), ElementwiseOp.SUM)

        # 8. Distogram: x_pred -> representative-atom repr -> cdist -> bin -> embed
        token_to_rep_atom = feats["token_to_rep_atom"]
        x_pred_repr = _rep_atom_to_token_1d(x_pred, token_to_rep_atom, self.device_mesh, self.cp_group)

        d = _outer_cdist_1d(x_pred_repr, self.device_mesh, self.cp_group)
        distogram = shardwise_distogram(d, self.boundaries)
        distogram = self.dist_bin_pairwise_embed(distogram)

        # 9. Repeat z for multiplicity and add distogram
        z = shardwise_repeat_interleave(z, multiplicity, dim=0)
        z = elementwise_op(z, distogram, ElementwiseOp.SUM)

        # 10. Masks for pairformer
        mask = shardwise_repeat_interleave(feats["token_pad_mask"], multiplicity, dim=0)
        pair_mask = shardwise_repeat_interleave(feats["token_pair_pad_mask"], multiplicity, dim=0)
        mask = mask.to(dtype=s.dtype)
        pair_mask = pair_mask.to(dtype=z.dtype)

        # 11. Pairformer
        # TODO(confidence-1d-parity): at cp>1, the confidence pairformer's layer 4
        # diverges — post-pairformer s produces near-random plddt logits (argmax ~24
        # vs ~48 at cp=1). Layers 0-3 are fine. The trunk pairformer works correctly
        # at cp>1 with the same PairformerModule1D code. 2D CP also passes. Root
        # cause is unidentified; next step is sub-operation instrumentation of layer 4.
        s, z = self.pairformer_stack(s, z, mask=mask, pair_mask=pair_mask)

        # 12. Output dict
        out_dict: dict[str, DTensor] = {}
        if self.return_latent_feats:
            out_dict["s_conf"] = s
            out_dict["z_conf"] = z

        # 13. Confidence heads
        out_dict.update(
            self.confidence_heads(
                s=s,
                z=z,
                x_pred=x_pred,
                d=d,
                feats=feats,
                pred_distogram_logits=pred_distogram_logits,
                multiplicity=multiplicity,
            )
        )
        return out_dict

    def _forward_sequentially(
        self,
        s_inputs: DTensor,
        s: DTensor,
        z: DTensor,
        x_pred: DTensor,
        feats: dict[str, DTensor],
        pred_distogram_logits: DTensor,
        multiplicity: int,
    ) -> dict[str, DTensor]:
        """Run the confidence module one multiplicity sample at a time."""
        x_pred_local = x_pred.to_local()
        if x_pred_local.shape[0] % multiplicity != 0:
            raise ValueError(
                f"x_pred.shape[0] must be divisible by multiplicity, "
                f"got {x_pred.shape[0]} and multiplicity {multiplicity}"
            )
        B_local = x_pred_local.shape[0] // multiplicity
        B_global = x_pred.shape[0] // multiplicity

        if B_local > 1:
            warnings.warn(
                "B_local > 1 could cause deadlocking issues with pair_chains_iptm "
                "when chain counts differ across dp groups",
                stacklevel=2,
            )

        x_pred_single_shape = torch.Size([B_global, *x_pred.shape[1:]])
        x_pred_unflat = x_pred_local.unflatten(0, (B_local, multiplicity))
        x_pred_single_stride = update_exhaustive_strides(x_pred.shape, x_pred.stride(), x_pred_single_shape)

        out_dicts: list[dict] = []
        for mult_idx in range(multiplicity):
            x_pred_sample = DTensor.from_local(
                x_pred_unflat[:, mult_idx : mult_idx + 1].flatten(0, 1),
                device_mesh=x_pred.device_mesh,
                placements=x_pred.placements,
                shape=x_pred_single_shape,
                stride=x_pred_single_stride,
            )
            out_dicts.append(
                self.forward(
                    s_inputs,
                    s,
                    z,
                    x_pred_sample,
                    feats,
                    pred_distogram_logits,
                    multiplicity=1,
                    run_sequentially=False,
                )
            )

        out_dict: dict[str, DTensor] = {}
        B_global_mult = x_pred.shape[0]
        for key in out_dicts[0]:
            if key != "pair_chains_iptm":
                ref = out_dicts[0][key]
                stacked = torch.stack([o[key].to_local() for o in out_dicts], dim=1)
                stacked_flattened = stacked.flatten(0, 1)
                out_shape = torch.Size([B_global_mult, *ref.shape[1:]])
                out_dict[key] = DTensor.from_local(
                    stacked_flattened,
                    device_mesh=ref.device_mesh,
                    placements=ref.placements,
                    shape=out_shape,
                    stride=update_exhaustive_strides(ref.shape, ref.stride(), out_shape),
                )
            else:
                pair_chains_iptm: dict = {}
                for idx1 in out_dicts[0][key]:
                    chain_iptm: dict = {}
                    for idx2 in out_dicts[0][key][idx1]:
                        ref = out_dicts[0][key][idx1][idx2]
                        stacked = torch.stack([o[key][idx1][idx2].to_local() for o in out_dicts], dim=1)
                        stacked_flattened = stacked.flatten(0, 1)
                        ref_shape = torch.Size([B_global_mult, *ref.shape[1:]])
                        chain_iptm[idx2] = DTensor.from_local(
                            stacked_flattened,
                            device_mesh=ref.device_mesh,
                            placements=ref.placements,
                            shape=ref_shape,
                            stride=update_exhaustive_strides(ref.shape, ref.stride(), ref_shape),
                        )
                    pair_chains_iptm[idx1] = chain_iptm
                out_dict[key] = pair_chains_iptm

        return out_dict
