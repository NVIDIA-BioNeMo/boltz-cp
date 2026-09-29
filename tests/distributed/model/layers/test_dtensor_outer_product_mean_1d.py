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


import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers import utils as _layer_utils
from boltz.distributed.model.layers.outer_product_mean_1d import OuterProductMean1D
from boltz.model.layers.outer_product_mean import OuterProductMean as SerialOuterProductMean
from boltz.testing.utils import (
    assert_all_identical,
    assert_no_percentile_upshift,
    assert_tensors_identical,
    get_param_by_key,
    init_module_params_uniform,
    init_tensors_uniform,
    seed_by_rank,
    spawn_multiprocessing,
)


def parallel_assert_outer_prod_mean_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    C_in,
    C_hidden,
    C_out,
    layer_state_dict,
    input_global_host,
    mask_global_host,
    output_expected_global_host,
    d_output_expected_global_host,
    d_input_expected_global_host,
    grad_params_expected_global_host,
    output_global_fp32_host: torch.Tensor | None = None,
    d_input_global_fp32_host: torch.Tensor | None = None,
    grad_params_fp32_global_host: dict[str, torch.Tensor] | None = None,
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

    # Memory-shape assertion: every ring P2P inside the OPM 1D kernel must
    # ship a chunk whose token-dimension size matches the local N/cp shard.
    # If a regression re-materialises a full-N b-buffer, the first send
    # tensor's dim-1 will jump to N_global and trip this assert.
    cp_size_test = grid_group_sizes["cp"]
    N_global_test = input_global_host.shape[2]
    expected_n_per_chunk = N_global_test // cp_size_test
    _orig_ring = _layer_utils._ring_p2p_send_recv

    def _ring_with_shape_assert(send_tensors, recv_tensors, *args, **kwargs):
        for t in send_tensors:
            assert t.shape[1] == expected_n_per_chunk, (
                f"OPM ring chunk had shape[1]={t.shape[1]}, expected {expected_n_per_chunk} "
                f"(N_global={N_global_test}, cp_size={cp_size_test})"
            )
        return _orig_ring(send_tensors, recv_tensors, *args, **kwargs)

    monkeypatch.setattr(_layer_utils, "_ring_p2p_send_recv", _ring_with_shape_assert)
    # The OPM module imports the helper by name; patch the bound reference too.
    import boltz.distributed.model.layers.outer_product_mean_1d as _opm_mod

    monkeypatch.setattr(_opm_mod, "_ring_p2p_send_recv", _ring_with_shape_assert)

    if torch.finfo(dtype).resolution < torch.finfo(output_expected_global_host.dtype).resolution:
        raise ValueError(
            f"Target dtype {dtype} has higher precision than reference output's dtype {output_expected_global_host.dtype}"
        )

    if ((output_global_fp32_host is None) != (d_input_global_fp32_host is None)) or (
        (output_global_fp32_host is not None) != (grad_params_fp32_global_host is not None)
    ):
        raise ValueError(
            "output_global_fp32_host, d_input_global_fp32_host, and grad_params_fp32_global_host "
            "must be either all None or all not None"
        )

    check_error_hist = output_global_fp32_host is not None

    # 1D CP: 2D mesh (dp, cp), cp_group is the flat "cp" group
    cp_group = manager.group["cp"]

    module_serial = SerialOuterProductMean(C_in, C_hidden, C_out).to(dtype=dtype)
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.to(device=manager.device)
    module = OuterProductMean1D(module_serial, manager.device_mesh, cp_group)
    module.train()

    # 1D CP placements: (Shard(0), Shard(2)) for MSA tensors on 2D mesh (dp, cp)
    placements_msa = (Shard(0), Shard(2))
    # Pair output placements: (Shard(0), Shard(1))
    placements_pair = (Shard(0), Shard(1))

    input_dtensor = distribute_tensor(
        input_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_msa,
    ).requires_grad_(True)
    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_msa,
    )
    d_output_expected_dtensor = distribute_tensor(
        d_output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_pair,
    )
    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_pair,
        src_data_rank=None,
    )
    d_input_expected_dtensor = distribute_tensor(
        d_input_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_msa,
        src_data_rank=None,
    )

    input_dtensor_copy = input_dtensor.detach().clone().requires_grad_(True)
    mask_dtensor_copy = mask_dtensor.detach().clone()

    if check_error_hist:
        output_dtensor_result = module(input_dtensor, mask_dtensor)
        output_dtensor_result.backward(d_output_expected_dtensor)

        output_fp32_dtensor = distribute_tensor(
            output_global_fp32_host.to(manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )

        d_input_fp32_dtensor = distribute_tensor(
            d_input_global_fp32_host.to(manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_msa,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            output_dtensor_result.to_local(),
            output_expected_dtensor.to_local(),
            output_fp32_dtensor.to_local(),
            names_input=("output_cp_fp32", "output_serial_fp64", "output_serial_fp32"),
        )

        assert_no_percentile_upshift(
            input_dtensor.grad.to_local(),
            d_input_expected_dtensor.to_local(),
            d_input_fp32_dtensor.to_local(),
            names_input=("d_input_cp_fp32", "d_input_serial_fp64", "d_input_serial_fp32"),
        )

        for name, grad_param_expected_global in grad_params_expected_global_host.items():
            grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
            assert_no_percentile_upshift(
                grad_param_result_global,
                grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                grad_params_fp32_global_host[name],
                names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
            )
    else:
        output_dtensor_result = module(input_dtensor, mask_dtensor)

        # Verify sharding is active: local N < global N on the sharded dim
        assert (
            output_dtensor_result.to_local().shape[1] < output_dtensor_result.shape[1]
        ), "Sharding is not active: local shape equals global shape on the sharded dim"

        # no modification on the input
        assert_tensors_identical(
            input_dtensor_copy.to_local(), input_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )
        assert_tensors_identical(mask_dtensor_copy.to_local(), mask_dtensor.to_local())

        # test for consistent forward results with the single-device
        assert (
            output_dtensor_result.shape == output_expected_dtensor.shape
        ), f"Output shape mismatch: {output_dtensor_result.shape} != {output_expected_dtensor.shape}"
        assert (
            output_dtensor_result.stride() == output_expected_dtensor.stride()
        ), f"Output stride mismatch: {output_dtensor_result.stride()} != {output_expected_dtensor.stride()}"
        torch.testing.assert_close(output_dtensor_result.to_local(), output_expected_dtensor.to_local())

        # check backward pass
        d_output_expected_dtensor_copy = d_output_expected_dtensor.detach().clone()
        output_dtensor_result.backward(d_output_expected_dtensor)

        # backward pass should not modify the upstream adjoint
        assert_tensors_identical(d_output_expected_dtensor_copy.to_local(), d_output_expected_dtensor.to_local())

        # Verify gradients are non-zero to guard against vacuous passes
        assert input_dtensor.grad.to_local().abs().sum() > 0, "Input gradient is all zeros"

        assert (
            input_dtensor.grad.shape == d_input_expected_dtensor.shape
        ), f"Gradient shape mismatch: {input_dtensor.grad.shape} != {d_input_expected_dtensor.shape}"
        assert (
            input_dtensor.grad.stride() == d_input_expected_dtensor.stride()
        ), f"Gradient stride mismatch: {input_dtensor.grad.stride()} != {d_input_expected_dtensor.stride()}"
        torch.testing.assert_close(input_dtensor.grad.to_local(), d_input_expected_dtensor.to_local())

        # check gradient of the weight
        grad_params_result_dtensors = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                if name not in grad_params_expected_global_host:
                    raise ValueError(f"Parameter {name} has a resulting gradient but it is not in the reference module")
                grad_params_result_dtensors[name] = param.grad

        for name, grad_param_expected_global_host in grad_params_expected_global_host.items():
            assert name in grad_params_result_dtensors, f"Parameter {name}'s gradient is not found in result gradients"
            grad_params_result = grad_params_result_dtensors[name]
            assert (
                grad_params_result.shape == grad_param_expected_global_host.shape
            ), f"Gradient shape mismatch: {grad_params_result.shape} != {grad_param_expected_global_host.shape}"
            assert (
                grad_params_result.stride() == grad_param_expected_global_host.stride()
            ), f"Gradient stride mismatch: {grad_params_result.stride()} != {grad_param_expected_global_host.stride()}"
            grad_params_result_global = grad_params_result.full_tensor()
            torch.testing.assert_close(grad_params_result_global.cpu(), grad_param_expected_global_host.to(dtype=dtype))
            assert_all_identical(grad_params_result_global, manager.group["cp"])

        # check the results with the full tensor to make sure the module's output and
        # gradients can be gathered into consistent results with the single-device
        input_global_result = input_dtensor.full_tensor()
        mask_global_result = mask_dtensor.full_tensor()
        output_global_result = output_dtensor_result.full_tensor()
        d_input_global_result = input_dtensor.grad.full_tensor()

        torch.testing.assert_close(input_global_result.cpu(), input_global_host.to(dtype=dtype))
        torch.testing.assert_close(mask_global_result.cpu(), mask_global_host.to(dtype=dtype))
        torch.testing.assert_close(output_global_result.cpu(), output_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_global_result.cpu(), d_input_expected_global_host.to(dtype=dtype))

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, check_error_hist",
    (
        params_test := [
            # CUDA tests (2 GPUs) - 1D CP with cp=2
            (((1, 2), True, "cuda", "ENV"), torch.float64, False),
            (((1, 2), True, "cuda", "ENV"), torch.float32, True),
            (((1, 2), True, "cuda", "ENV"), torch.float32, False),
            # CUDA test - non-power-of-two CP for GPU CI with 3+ GPUs
            (((1, 3), True, "cuda", "ENV"), torch.float64, False),
            # CPU test - non-power-of-two CP for CPU-only CI
            (((1, 3), True, "cpu", "ENV"), torch.float64, False),
        ]
    ),
    indirect=["setup_env"],
    ids=[
        f"dp:{x[0][0][0]}, cp:{x[0][0][1]}, device_type:{x[0][2]}, " f"dtype:{x[1]}, check_error_hist:{x[2]}"
        for x in params_test
    ],
)
def test_outer_product_mean_1d(setup_env, dtype, check_error_hist):
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    # For error histogram check, use realistic sizes; otherwise use small sizes
    # to keep tests fast and expose logical bugs.
    test_large_model = check_error_hist or dtype == torch.float64

    cp_size = grid_group_sizes["cp"]

    B = 2 * grid_group_sizes["dp"]
    if test_large_model:
        N = cp_size * 64
        S = cp_size * 64
        C_in = 64
        C_hidden = 32
        C_out = 128
        min_val_init = -5e-2
        max_val_init = 5e-2
    else:
        N = cp_size * 4
        S = cp_size * 3
        C_in = 3
        C_hidden = 5
        C_out = 3
        min_val_init = -0.5
        max_val_init = 0.5

    seed = 42
    seed_by_rank(0, seed=seed)

    # Compute reference results with FP64
    input_global_fp64 = torch.empty((B, S, N, C_in), dtype=torch.float64, requires_grad=True, device=device_type)
    mask_global_fp64 = torch.ones((B, S, N), dtype=torch.float64, requires_grad=False, device=device_type)
    # Mask out some rows to exercise non-trivial masking
    mask_global_fp64[0, (S // cp_size) :, :] = 0
    mask_global_fp64[0, :, (N // cp_size) :] = 0
    reference_module = SerialOuterProductMean(C_in, C_hidden, C_out).to(dtype=torch.float64)
    init_tensors_uniform([input_global_fp64], low=min_val_init, high=max_val_init)
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)
    layer_state_dict_fp64 = reference_module.state_dict()
    reference_module = reference_module.to(device=device_type).train()

    output_expected_global_fp64 = reference_module(input_global_fp64, mask_global_fp64)
    d_output_expected_global_fp64 = torch.rand_like(output_expected_global_fp64)
    output_expected_global_fp64.backward(d_output_expected_global_fp64)

    grad_params_fp64_expected_global_host = {
        name: param.grad.detach().clone().cpu() for name, param in reference_module.named_parameters()
    }

    if check_error_hist:
        input_global_fp32 = input_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        mask_global_fp32 = mask_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(False)
        reference_module_fp32 = SerialOuterProductMean(C_in, C_hidden, C_out).to(dtype=torch.float32)
        reference_module_fp32.load_state_dict(layer_state_dict_fp64)
        reference_module_fp32 = reference_module_fp32.to(device=device_type).train()
        output_global_fp32 = reference_module_fp32(input_global_fp32, mask_global_fp32)
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
        parallel_assert_outer_prod_mean_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        C_in,
        C_hidden,
        C_out,
        layer_state_dict_fp64,
        input_global_fp64.detach().clone().cpu(),
        mask_global_fp64.detach().clone().cpu(),
        output_expected_global_fp64.detach().clone().cpu(),
        d_output_expected_global_fp64.detach().clone().cpu(),
        input_global_fp64.grad.detach().clone().cpu(),
        grad_params_fp64_expected_global_host,
        output_global_fp32_host,
        d_input_global_fp32_host,
        grad_params_fp32_global_host,
    )
