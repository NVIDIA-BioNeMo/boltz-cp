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

"""Parity tests for outer_sum_1d under 1D CP (2D mesh ``(dp, cp)``)."""

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.outer_sum_1d import outer_sum_1d
from boltz.testing.utils import (
    assert_no_percentile_upshift,
    seed_by_rank,
    spawn_multiprocessing,
)


def _worker_outer_sum_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    z1_global_host,
    z2_global_host,
    output_expected_global_host,
    d_output_global_host,
    d_z1_expected_global_host,
    d_z2_expected_global_host,
    symmetric,
    output_global_fp32_host=None,
    d_z1_global_fp32_host=None,
    d_z2_global_fp32_host=None,
):
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    check_error_hist = output_global_fp32_host is not None

    placements_single = (Shard(0), Shard(1))
    placements_pair = (Shard(0), Shard(1))

    # Distribute inputs
    z1_dtensor = distribute_tensor(
        z1_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_single,
    ).requires_grad_(True)

    z2_input = (
        z1_dtensor
        if symmetric
        else distribute_tensor(
            z2_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
        ).requires_grad_(True)
    )

    # Get cp group
    cp_group = manager.group["cp"]

    # Forward
    output_dtensor = outer_sum_1d(z1_dtensor, z2_input, manager.device_mesh, cp_group)

    # Distribute expected outputs for comparison
    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_pair,
        src_data_rank=None,
    )
    d_output_dtensor = distribute_tensor(
        d_output_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_pair,
    )

    if check_error_hist:
        # Backward
        output_dtensor.backward(d_output_dtensor)

        output_fp32_dtensor = distribute_tensor(
            output_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )
        d_z1_fp32_dtensor = distribute_tensor(
            d_z1_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            output_dtensor.to_local(),
            output_expected_dtensor.to_local(),
            output_fp32_dtensor.to_local(),
            names_input=("output_cp_fp32", "output_serial_fp64", "output_serial_fp32"),
        )
        assert_no_percentile_upshift(
            z1_dtensor.grad.to_local(),
            distribute_tensor(
                d_z1_expected_global_host.to(dtype=dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_single,
                src_data_rank=None,
            ).to_local(),
            d_z1_fp32_dtensor.to_local(),
            names_input=("d_z1_cp_fp32", "d_z1_serial_fp64", "d_z1_serial_fp32"),
        )

        if not symmetric:
            d_z2_fp32_dtensor = distribute_tensor(
                d_z2_global_fp32_host.to(device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_single,
                src_data_rank=None,
            )
            assert_no_percentile_upshift(
                z2_input.grad.to_local(),
                distribute_tensor(
                    d_z2_expected_global_host.to(dtype=dtype, device=manager.device),
                    device_mesh=manager.device_mesh,
                    placements=placements_single,
                    src_data_rank=None,
                ).to_local(),
                d_z2_fp32_dtensor.to_local(),
                names_input=("d_z2_cp_fp32", "d_z2_serial_fp64", "d_z2_serial_fp32"),
            )
    else:
        # Forward parity (local shard comparison)
        torch.testing.assert_close(output_dtensor.to_local(), output_expected_dtensor.to_local())

        # Full tensor parity against serial reference
        output_global_result = output_dtensor.full_tensor().cpu()
        torch.testing.assert_close(output_global_result, output_expected_global_host.to(dtype=dtype))

        # Backward
        output_dtensor.backward(d_output_dtensor)

        # Gradient parity
        d_z1_expected_dtensor = distribute_tensor(
            d_z1_expected_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        torch.testing.assert_close(z1_dtensor.grad.to_local(), d_z1_expected_dtensor.to_local())
        torch.testing.assert_close(z1_dtensor.grad.full_tensor().cpu(), d_z1_expected_global_host.to(dtype=dtype))

        if not symmetric:
            d_z2_expected_dtensor = distribute_tensor(
                d_z2_expected_global_host.to(dtype=dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_single,
                src_data_rank=None,
            )
            torch.testing.assert_close(z2_input.grad.to_local(), d_z2_expected_dtensor.to_local())
            torch.testing.assert_close(z2_input.grad.full_tensor().cpu(), d_z2_expected_global_host.to(dtype=dtype))

    # Guard against vacuous pass: verify CP sharding is active
    cp_placement = z1_dtensor.placements[1]
    if isinstance(cp_placement, Shard):
        local_size = z1_dtensor.to_local().shape[cp_placement.dim]
        global_size = z1_dtensor.shape[cp_placement.dim]
        assert local_size < global_size, (
            f"CP sharding not active on dim {cp_placement.dim}: " f"local={local_size}, global={global_size}"
        )

    # Guard: output has correct row-slab shape
    N_global = z1_dtensor.shape[1]
    cp_size = manager.device_mesh.shape[1]
    N_local = N_global // cp_size
    assert (
        output_dtensor.to_local().shape[1] == N_local
    ), f"Expected row-slab with N_local={N_local} rows, got {output_dtensor.to_local().shape[1]}"
    assert (
        output_dtensor.to_local().shape[2] == N_global
    ), f"Expected full N={N_global} columns, got {output_dtensor.to_local().shape[2]}"

    # Guard: gradients are non-zero
    assert z1_dtensor.grad.to_local().abs().sum() > 0, "z1 gradients are all zero"
    if not symmetric:
        assert z2_input.grad.to_local().abs().sum() > 0, "z2 gradients are all zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


def _serial_outer_sum(z1, z2):
    """Serial reference: z1[:, :, None, :] + z2[:, None, :, :]."""
    return z1.unsqueeze(2) + z2.unsqueeze(1)


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
        ((1, 4), True, "cuda", "ENV"),
        ((2, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cuda", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
@pytest.mark.parametrize(
    "dtype_and_check_error_hist",
    [
        (torch.float32, False),
        (torch.float32, True),
        (torch.float64, False),
    ],
    ids=lambda x: f"dtype={x[0]}, check_error_hist={x[1]}",
)
@pytest.mark.parametrize(
    "symmetric",
    [False, True],
    ids=lambda x: f"symmetric={x}",
)
def test_outer_sum_1d(setup_env, dtype_and_check_error_hist, symmetric):
    """Test outer_sum_1d parity against serial outer sum."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, check_error_hist = dtype_and_check_error_hist

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    if check_error_hist and grid_group_sizes["dp"] > 1:
        pytest.skip("skip error histogram check for dp > 1 to save test time")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    if check_error_hist or dtype == torch.float64:
        N = cp_size * 64
        C = 128
    else:
        N = cp_size * 4
        C = 16

    seed_by_rank(0, seed=42)

    z1_global_fp64 = torch.empty(B, N, C, dtype=torch.float64, requires_grad=True, device=device_type)
    z2_global_fp64 = torch.empty(B, N, C, dtype=torch.float64, requires_grad=True, device=device_type)
    with torch.no_grad():
        z1_global_fp64.uniform_(-5e-2, 5e-2)
        z2_global_fp64.uniform_(-5e-2, 5e-2)

    # Serial reference in fp64
    z2_for_serial = z1_global_fp64 if symmetric else z2_global_fp64
    output_fp64 = _serial_outer_sum(z1_global_fp64, z2_for_serial)

    d_output_fp64 = torch.rand_like(output_fp64)
    output_fp64.backward(d_output_fp64)
    d_z1_fp64 = z1_global_fp64.grad.detach().clone()
    d_z2_fp64 = z2_global_fp64.grad.detach().clone() if not symmetric else None
    output_fp64 = output_fp64.detach().clone()

    if check_error_hist:
        z1_global_fp32 = z1_global_fp64.detach().clone().to(torch.float32).requires_grad_(True)
        z2_global_fp32 = z2_global_fp64.detach().clone().to(torch.float32).requires_grad_(True)
        z2_for_serial_fp32 = z1_global_fp32 if symmetric else z2_global_fp32
        output_fp32 = _serial_outer_sum(z1_global_fp32, z2_for_serial_fp32)
        d_output_fp32 = d_output_fp64.to(torch.float32)
        output_fp32.backward(d_output_fp32)
        output_global_fp32_host = output_fp32.detach().clone().cpu()
        d_z1_global_fp32_host = z1_global_fp32.grad.detach().clone().cpu()
        d_z2_global_fp32_host = z2_global_fp32.grad.detach().clone().cpu() if not symmetric else None
    else:
        output_global_fp32_host = None
        d_z1_global_fp32_host = None
        d_z2_global_fp32_host = None

    spawn_multiprocessing(
        _worker_outer_sum_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        z1_global_fp64.detach().clone().cpu(),
        z2_global_fp64.detach().clone().cpu(),
        output_fp64.cpu(),
        d_output_fp64.cpu(),
        d_z1_fp64.cpu(),
        d_z2_fp64.cpu() if d_z2_fp64 is not None else None,
        symmetric,
        output_global_fp32_host,
        d_z1_global_fp32_host,
        d_z2_global_fp32_host,
    )
