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

import unittest.mock

import pytest
import torch

# Sibling-import the topology-agnostic worker from the 2D scatter test.
# See tests/distributed/test_dtensor_boltz2_1d_train.py:77 for precedent.
from tests.distributed.model.layers.test_dtensor_scatter import parallel_assert_scatter_reduce

from boltz.distributed.manager import DistributedManager
from boltz.testing.utils import spawn_multiprocessing


def _parallel_assert_scatter_reduce_1d(rank, grid_group_sizes, device_type, backend, env_map, dtype, reduce_op):
    """Thin wrapper: patches device_mesh_subgroups → device_mesh for 1D CP.

    Under 1D CP, DistributedManager._device_mesh_subgroups is None (no tuple
    CP sizes means no subgroup mesh is created). The 2D worker iterates
    [device_mesh_subgroups, device_mesh]; patching None → device_mesh makes
    the iteration degenerate (same mesh twice) but correct, which is the
    spec's stated intent.
    """
    import boltz.distributed.manager as manager_module

    original_getattr = DistributedManager.__getattr__

    def patched_getattr(self, name):
        if name == "device_mesh_subgroups":
            val = original_getattr(self, name)
            if val is None:
                return original_getattr(self, "device_mesh")
            return val
        return original_getattr(self, name)

    with unittest.mock.patch.object(manager_module.DistributedManager, "__getattr__", patched_getattr):
        parallel_assert_scatter_reduce(rank, grid_group_sizes, device_type, backend, env_map, dtype, reduce_op)


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
        ((2, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp:{x[0][0]}, cp:{x[0][1]}, specify_method:{x[1]}, device_type:{x[2]}, method_init:{x[3]}",
)
@pytest.mark.parametrize("reduce_op", ["sum", "mean"])
def test_distributed_scatter_reduce_1d(setup_env, reduce_op):
    """Test distributed_scatter_reduce under 1D CP placements.

    Exercises the forward/backward all_gather and batch_isend_irecv peer-selection
    through the genuine 1D CP process-group routing (flat int cp=2 on a (dp, cp) mesh),
    which the 2D test does not cover independently.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        _parallel_assert_scatter_reduce_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        torch.float64,
        reduce_op,
    )
