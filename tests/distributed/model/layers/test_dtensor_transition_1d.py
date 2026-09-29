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

"""Parity tests for Transition1D and SwiGLU1D under 1D CP (2D mesh ``(dp, cp)``)."""

from collections import OrderedDict

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.transition_1d import SwiGLU1D, Transition1D
from boltz.model.layers.transition import Transition
from boltz.testing.utils import (
    assert_all_identical,
    assert_no_percentile_upshift,
    assert_tensors_identical,
    get_param_by_key,
    seed_by_rank,
    spawn_multiprocessing,
)


def _worker_transition_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    dim,
    hidden,
    out_dim,
    layer_state_dict,
    input_global_host,
    placements_input,
    output_expected_global_host,
    d_output_expected_global_host,
    d_input_expected_global_host,
    grad_params_expected_global_host,
    output_global_fp32_host=None,
    d_input_global_fp32_host=None,
    grad_params_fp32_global_host=None,
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

    # Create serial reference module
    module_serial = Transition(dim, hidden, out_dim)
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.to(dtype=dtype, device=manager.device)
    module_serial.train()

    # Create distributed module — uses 2D device_mesh (dp, cp)
    module = Transition1D(module_serial, manager.device_mesh)
    module.train()

    # Distribute input tensor
    input_dtensor = distribute_tensor(
        input_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
    ).requires_grad_(True)

    # Distribute expected outputs
    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
        src_data_rank=None,
    )
    d_output_expected_dtensor = distribute_tensor(
        d_output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
    )
    d_input_expected_dtensor = distribute_tensor(
        d_input_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
        src_data_rank=None,
    )

    input_dtensor_copy = input_dtensor.detach().clone().requires_grad_(True)

    if check_error_hist:
        # CP all-reduce sums cp_size partial gradient contributions in a different
        # accumulation order than serial.  FP32 rounding error scales roughly as
        # O(cp_size * eps_fp32).  For cp=4 the worst-case percentile shift is ~2x
        # the cp=2 baseline, so we widen atol proportionally.  The default
        # assert_close tolerances (atol=1e-05, rtol=1.3e-06) suffice for cp<=2.
        cp_size = grid_group_sizes["cp"]
        if cp_size <= 2:
            perc = None  # default tolerances
        else:
            # Scale atol by cp_size/2 relative to the FP32 default of 1e-05,
            # giving 2e-05 for cp=4, 3e-05 for cp=6, etc.
            scaled_atol = 1e-05 * cp_size / 2
            perc = OrderedDict(
                {
                    0.25: (scaled_atol, None),
                    0.5: (scaled_atol, None),
                    0.75: (scaled_atol, None),
                    0.95: (scaled_atol, None),
                }
            )

        output_dtensor_result = module(input_dtensor)
        output_dtensor_result.backward(d_output_expected_dtensor)

        output_fp32_dtensor = distribute_tensor(
            output_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_input,
            src_data_rank=None,
        )
        d_input_fp32_dtensor = distribute_tensor(
            d_input_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_input,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            output_dtensor_result.to_local(),
            output_expected_dtensor.to_local(),
            output_fp32_dtensor.to_local(),
            perc=perc,
            names_input=("output_cp_fp32", "output_serial_fp64", "output_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_dtensor.grad.to_local(),
            d_input_expected_dtensor.to_local(),
            d_input_fp32_dtensor.to_local(),
            perc=perc,
            names_input=("d_input_cp_fp32", "d_input_serial_fp64", "d_input_serial_fp32"),
        )

        for name, grad_param_expected_global in grad_params_expected_global_host.items():
            grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
            assert_no_percentile_upshift(
                grad_param_result_global,
                grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                grad_params_fp32_global_host[name],
                perc=perc,
                names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
            )
    else:
        # Forward pass
        output_dtensor_result = module(input_dtensor)

        # Verify inputs weren't modified
        assert_tensors_identical(
            input_dtensor_copy.to_local(), input_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )

        # Test forward pass results (local shard comparison — no communication)
        torch.testing.assert_close(output_dtensor_result.to_local(), output_expected_dtensor.to_local())

        # Backward pass
        d_output_expected_dtensor_copy = d_output_expected_dtensor.detach().clone()
        output_dtensor_result.backward(d_output_expected_dtensor)

        # Verify upstream gradient wasn't modified
        assert_tensors_identical(d_output_expected_dtensor_copy.to_local(), d_output_expected_dtensor.to_local())

        # Test input gradients (local shard comparison)
        torch.testing.assert_close(input_dtensor.grad.to_local(), d_input_expected_dtensor.to_local())

        # Verify full tensors match serial reference (requires all-gather)
        input_global_result_host = input_dtensor.full_tensor().cpu()
        output_global_result_host = output_dtensor_result.full_tensor().cpu()
        d_input_global_result_host = input_dtensor.grad.full_tensor().cpu()

        torch.testing.assert_close(input_global_result_host, input_global_host.to(dtype=dtype))
        torch.testing.assert_close(output_global_result_host, output_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_global_result_host, d_input_expected_global_host.to(dtype=dtype))

        # Test parameter gradients
        grad_params_result_dtensors = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                if name not in grad_params_expected_global_host:
                    raise ValueError(f"Parameter {name} has a resulting gradient but it is not in the reference module")
                grad_params_result_dtensors[name] = param.grad

        for name, grad_param_expected_global in grad_params_expected_global_host.items():
            assert name in grad_params_result_dtensors, f"Parameter {name}'s gradient is not found in result gradients"
            grad_params_result = grad_params_result_dtensors[name]
            param_grad_result = grad_params_result.full_tensor()
            torch.testing.assert_close(param_grad_result.cpu(), grad_param_expected_global.to(dtype=dtype))
            assert_all_identical(param_grad_result, manager.group["cp"])

        # Guard against vacuous pass: verify CP sharding is active (mesh dim 1 = cp)
        cp_placement = input_dtensor.placements[1]  # cp axis
        if isinstance(cp_placement, Shard):
            local_size = input_dtensor.to_local().shape[cp_placement.dim]
            global_size = input_dtensor.shape[cp_placement.dim]
            assert (
                local_size < global_size
            ), f"CP sharding not active on dim {cp_placement.dim}: local={local_size}, global={global_size}"

        # Guard against vacuous pass: verify gradients are non-zero
        assert input_dtensor.grad.to_local().abs().sum() > 0, "Input gradients are all zero"
        for name, param in module.named_parameters():
            if param.grad is not None:
                assert param.grad.to_local().abs().sum() > 0, f"Gradient for {name} is all zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


def _worker_swiglu_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    placements_input,
    input_global_host,
    output_expected_global_host,
    d_output_expected_global_host,
    d_input_expected_global_host,
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

    module = SwiGLU1D()

    input_dtensor = distribute_tensor(
        input_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
    ).requires_grad_(True)

    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
        src_data_rank=None,
    )
    d_output_expected_dtensor = distribute_tensor(
        d_output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
    )
    d_input_expected_dtensor = distribute_tensor(
        d_input_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_input,
        src_data_rank=None,
    )

    output_dtensor_result = module(input_dtensor)

    # Forward parity
    torch.testing.assert_close(output_dtensor_result.to_local(), output_expected_dtensor.to_local())

    # Backward parity
    output_dtensor_result.backward(d_output_expected_dtensor)
    torch.testing.assert_close(input_dtensor.grad.to_local(), d_input_expected_dtensor.to_local())

    # Full tensor parity against serial
    torch.testing.assert_close(output_dtensor_result.full_tensor().cpu(), output_expected_global_host.to(dtype=dtype))
    torch.testing.assert_close(input_dtensor.grad.full_tensor().cpu(), d_input_expected_global_host.to(dtype=dtype))

    # Guard: CP sharding is active (mesh dim 1 = cp)
    cp_placement = input_dtensor.placements[1]  # cp axis
    if isinstance(cp_placement, Shard):
        assert input_dtensor.to_local().shape[cp_placement.dim] < input_dtensor.shape[cp_placement.dim]

    # Guard: gradients are non-zero
    assert input_dtensor.grad.to_local().abs().sum() > 0

    DistributedManager.cleanup()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# Transition1D tests
# ---------------------------------------------------------------------------


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
    "repr_kind",
    ["single", "pair"],
    ids=lambda x: f"repr={x}",
)
def test_transition_1d(setup_env, dtype_and_check_error_hist, repr_kind):
    """Test Transition1D parity against serial Transition for single/pair reprs."""
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
        N = cp_size * 128
        dim = 128
        hidden = 512
        out_dim = 128
    else:
        N = cp_size * 4
        dim = 16
        hidden = 64
        out_dim = 16

    seed = 42
    seed_by_rank(0, seed=seed)

    # Build input shape and placements based on repr kind
    if repr_kind == "single":
        # s [B, N, C_s] -> (Shard(0), Shard(1))
        input_shape = (B, N, dim)
        placements_input = (Shard(0), Shard(1))
    else:
        # z [B, N, N, C_z] -> (Shard(0), Shard(1))
        input_shape = (B, N, N, dim)
        placements_input = (Shard(0), Shard(1))

    input_global_fp64 = torch.empty(input_shape, dtype=torch.float64, requires_grad=True, device=device_type)

    reference_module = Transition(dim, hidden, out_dim)
    with torch.no_grad():
        input_global_fp64.uniform_(-1e-2, 1e-2)
        for name, param in reference_module.named_parameters():
            param.uniform_(-1e-2, 1e-2)

    layer_state_dict_fp64 = reference_module.state_dict()
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).train()

    output_expected_global_fp64 = reference_module(input_global_fp64)
    d_output_expected_global_fp64 = torch.rand_like(output_expected_global_fp64)
    output_expected_global_fp64.backward(d_output_expected_global_fp64)

    grad_params_fp64_expected_global_host = {
        name: param.grad.detach().clone().cpu() for name, param in reference_module.named_parameters()
    }

    if check_error_hist:
        input_global_fp32 = input_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        reference_module_fp32 = Transition(dim, hidden, out_dim)
        reference_module_fp32.load_state_dict(layer_state_dict_fp64)
        reference_module_fp32 = reference_module_fp32.to(dtype=torch.float32, device=device_type).train()

        output_global_fp32 = reference_module_fp32(input_global_fp32)
        d_output_expected_global_fp32 = d_output_expected_global_fp64.to(dtype=torch.float32)
        output_global_fp32.backward(d_output_expected_global_fp32)

        output_global_fp32_host = output_global_fp32.detach().clone().cpu()
        d_input_global_fp32_host = input_global_fp32.grad.detach().clone().cpu()
        grad_params_fp32_global_host = {
            name: param.grad.detach().clone().cpu() for name, param in reference_module_fp32.named_parameters()
        }
    else:
        output_global_fp32_host = None
        d_input_global_fp32_host = None
        grad_params_fp32_global_host = None

    spawn_multiprocessing(
        _worker_transition_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        dim,
        hidden,
        out_dim,
        layer_state_dict_fp64,
        input_global_fp64.detach().clone().cpu(),
        placements_input,
        output_expected_global_fp64.detach().clone().cpu(),
        d_output_expected_global_fp64.detach().clone().cpu(),
        input_global_fp64.grad.detach().clone().cpu(),
        grad_params_fp64_expected_global_host,
        output_global_fp32_host,
        d_input_global_fp32_host,
        grad_params_fp32_global_host,
    )


# ---------------------------------------------------------------------------
# SwiGLU1D tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
        ((1, 4), True, "cuda", "ENV"),
        ((1, 3), True, "cuda", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=lambda x: f"dtype={x}",
)
def test_swiglu_1d(setup_env, dtype):
    """Test SwiGLU1D parity against serial SwiGLU for single repr."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 4
    C = 32  # must be even for SwiGLU chunk(2, dim=-1)

    seed_by_rank(0, seed=42)

    # Single repr: s [B, N, C] -> (Shard(0), Shard(1))
    input_shape = (B, N, C)
    placements_input = (Shard(0), Shard(1))

    input_global = torch.empty(input_shape, dtype=torch.float64, requires_grad=True, device=device_type)
    with torch.no_grad():
        input_global.uniform_(-5e-2, 5e-2)

    # Serial SwiGLU reference
    import torch.nn.functional as F

    x_ref = input_global.detach().clone().requires_grad_(True)
    x_chunks, gates_chunks = x_ref.chunk(2, dim=-1)
    output_ref = F.silu(gates_chunks) * x_chunks
    d_output_ref = torch.rand_like(output_ref)
    output_ref.backward(d_output_ref)

    # SwiGLU output has half the last dim, so its placements are the same
    # (Shard dims don't touch last dim for single repr)
    spawn_multiprocessing(
        _worker_swiglu_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        placements_input,
        input_global.detach().clone().cpu(),
        output_ref.detach().clone().cpu(),
        d_output_ref.detach().clone().cpu(),
        x_ref.grad.detach().clone().cpu(),
    )
