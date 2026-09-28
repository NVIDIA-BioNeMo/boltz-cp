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

"""Outer sum for 1D context parallelism on a 2D mesh ``(dp, cp)``.

Computes the pairwise outer sum used to initialise the pair representation:

    z_init = z1[:, :, None, :] + z2[:, None, :, :]

where ``z1`` and ``z2`` are single representations with shape ``[B, N, C]``.

Under 1D CP, both inputs are sharded along the token dimension (dim 1) on the
cp axis.  The result is the row-slab pair representation ``[B, N/cp, N, C]``
with placement ``(Shard(0), Shard(1))``.

Communication budget
--------------------
Forward:  1 all-gather of ``z2`` along cp  (``N/cp * C`` per rank).
Backward: 1 reduce-scatter of ``grad_z2`` along cp (``N/cp * C`` output per rank).
          ``grad_z1`` is fully local (sum over full-N column dimension).
"""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from boltz.distributed.utils import update_exhaustive_strides


class _OuterSum1D(torch.autograd.Function):
    """Autograd function for 1D CP outer sum producing row-slab pair repr.

    Input placements (on 2D mesh ``(dp, cp)``):
        z1: ``(Shard(0), Shard(1))``  — single repr ``[B, N, C]``
        z2: ``(Shard(0), Shard(1))``  — single repr ``[B, N, C]``

    Output placement:
        z_out: ``(Shard(0), Shard(1))`` — pair repr ``[B, N, N, C]`` (row-slab)

    Forward:
        1. All-gather z2 along cp → z2_full ``[B, N, C]``
        2. z_out = z1_local[:, :, None, :] + z2_full[:, None, :, :]
           Result shape: ``[B_local, N/cp, N, C]``

    Backward:
        1. grad_z1 = grad_output.sum(dim=2) → ``[B_local, N/cp, C]`` (local)
        2. grad_z2_full = grad_output.sum(dim=1) → ``[B_local, N, C]``
        3. Reduce-scatter grad_z2_full along cp → ``[B_local, N/cp, C]``
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
        # --- Input validation ---
        if not isinstance(z1, DTensor):
            raise TypeError(f"z1 must be a DTensor, got {type(z1)}")
        if not isinstance(z2, DTensor):
            raise TypeError(f"z2 must be a DTensor, got {type(z2)}")

        expected_placements = (Shard(0), Shard(1))
        if z1.placements != expected_placements:
            raise ValueError(f"z1 must have placements {expected_placements}, got {z1.placements}")
        if z2.placements != expected_placements:
            raise ValueError(f"z2 must have placements {expected_placements}, got {z2.placements}")
        if z1.device_mesh != z2.device_mesh:
            raise ValueError("z1 and z2 must share the same device_mesh")

        # Even sharding checks
        dp_size = device_mesh.shape[0]
        cp_size = device_mesh.shape[1]
        if z1.shape[0] % dp_size != 0:
            raise ValueError(f"Uneven sharding: z1 dim 0 ({z1.shape[0]}) not divisible by dp_size ({dp_size})")
        if z2.shape[0] % dp_size != 0:
            raise ValueError(f"Uneven sharding: z2 dim 0 ({z2.shape[0]}) not divisible by dp_size ({dp_size})")
        if z1.shape[1] % cp_size != 0:
            raise ValueError(f"Uneven sharding: z1 dim 1 ({z1.shape[1]}) not divisible by cp_size ({cp_size})")
        if z2.shape[1] % cp_size != 0:
            raise ValueError(f"Uneven sharding: z2 dim 1 ({z2.shape[1]}) not divisible by cp_size ({cp_size})")

        ctx.device_mesh = device_mesh
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.z1_shape = z1.shape
        ctx.z2_shape = z2.shape
        ctx.z1_requires_grad = z1.requires_grad
        ctx.z2_requires_grad = z2.requires_grad

        z1_local = z1.to_local()  # [B_local, N/cp, C]
        z2_local = z2.to_local()  # [B_local, N/cp, C]

        # All-gather z2 along cp (skip if cp_size == 1)
        if cp_size == 1:
            z2_full = z2_local
        else:
            z2_gathered_list = [torch.empty_like(z2_local) for _ in range(cp_size)]
            dist.all_gather(z2_gathered_list, z2_local.contiguous(), group=cp_group)
            z2_full = torch.cat(z2_gathered_list, dim=1)  # [B_local, N, C]

        # Outer sum: z1[:, :, None, :] + z2_full[:, None, :, :]
        output_local = z1_local.unsqueeze(2) + z2_full.unsqueeze(1)
        # Shape: [B_local, N/cp, N, C]

        # Build output DTensor with pair repr global shape [B, N, N, C]
        B_global = z1.shape[0]
        N_global = z1.shape[1]
        C = z1.shape[2]
        output_global_shape = torch.Size([B_global, N_global, N_global, C])
        output_stride = update_exhaustive_strides(output_local.shape, output_local.stride(), output_global_shape)

        output = DTensor.from_local(
            output_local,
            device_mesh=device_mesh,
            placements=(Shard(0), Shard(1)),
            shape=output_global_shape,
            stride=output_stride,
        )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: DTensor):
        device_mesh = ctx.device_mesh
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size

        grad_local = grad_output.to_local()  # [B_local, N/cp, N, C]

        grad_z1 = None
        grad_z2 = None

        if ctx.z1_requires_grad:
            # grad_z1 = grad_output.sum(dim=2): sum over full-N column dim (local)
            grad_z1_local = grad_local.sum(dim=2)  # [B_local, N/cp, C]
            grad_z1_stride = update_exhaustive_strides(grad_z1_local.shape, grad_z1_local.stride(), ctx.z1_shape)
            grad_z1 = DTensor.from_local(
                grad_z1_local,
                device_mesh=device_mesh,
                placements=(Shard(0), Shard(1)),
                shape=ctx.z1_shape,
                stride=grad_z1_stride,
            )

        if ctx.z2_requires_grad:
            # grad_z2_full = grad_output.sum(dim=1): sum over N/cp row dim (local)
            # This gives [B_local, N, C] — each rank has a partial sum over its row shard.
            # Promote to >=fp32 for the cross-rank reduce-scatter accumulation per
            # CLAUDE.md "Reduce gradients in fp32"; cast back to input_dtype AFTER
            # the collective so bf16/fp16 autocast does not lose precision in the
            # accumulation.  Mirrors _AllGatherTriangleAttentionStartingNode1DImpl
            # in triangular_attention_1d.py.
            input_dtype = grad_local.dtype
            compute_dtype = torch.promote_types(input_dtype, torch.float32)
            grad_z2_full = grad_local.sum(dim=1, dtype=compute_dtype)  # [B_local, N, C]

            # Reduce-scatter along cp (skip if cp_size == 1)
            if cp_size == 1:
                grad_z2_local = grad_z2_full
            else:
                grad_z2_chunks = list(grad_z2_full.chunk(cp_size, dim=1))
                grad_z2_chunks = [c.contiguous() for c in grad_z2_chunks]
                grad_z2_local = torch.empty(
                    grad_z2_chunks[0].shape,
                    dtype=grad_z2_chunks[0].dtype,
                    device=grad_z2_chunks[0].device,
                )
                dist.reduce_scatter(grad_z2_local, grad_z2_chunks, op=dist.ReduceOp.SUM, group=cp_group)

            # Cast back to input_dtype after the fp32 reduce-scatter.
            grad_z2_local = grad_z2_local.to(input_dtype)

            # Ensure contiguous strides for update_exhaustive_strides validation.
            # torch.empty_like(chunk_view) can inherit non-exhaustive strides from
            # chunk views; use .contiguous() as a safety net.
            grad_z2_local = grad_z2_local.contiguous()
            grad_z2_stride = update_exhaustive_strides(grad_z2_local.shape, grad_z2_local.stride(), ctx.z2_shape)
            grad_z2 = DTensor.from_local(
                grad_z2_local,
                device_mesh=device_mesh,
                placements=(Shard(0), Shard(1)),
                shape=ctx.z2_shape,
                stride=grad_z2_stride,
            )

        return grad_z1, grad_z2, None, None


def outer_sum_1d(
    z1: DTensor,
    z2: DTensor,
    device_mesh: DeviceMesh,
    cp_group: dist.ProcessGroup,
) -> DTensor:
    """Compute outer sum of two single representations for 1D CP.

    Produces the row-slab pair representation:
        ``z_out[b, i, j, c] = z1[b, i, c] + z2[b, j, c]``

    Parameters
    ----------
    z1 : DTensor
        Single representation ``[B, N, C]`` with placement ``(Shard(0), Shard(1))``.
    z2 : DTensor
        Single representation ``[B, N, C]`` with placement ``(Shard(0), Shard(1))``.
    device_mesh : DeviceMesh
        2D device mesh ``(dp, cp)``.
    cp_group : dist.ProcessGroup
        Process group for the cp axis.

    Returns
    -------
    DTensor
        Pair representation ``[B, N, N, C]`` with placement ``(Shard(0), Shard(1))``.
    """
    return _OuterSum1D.apply(z1, z2, device_mesh, cp_group)
