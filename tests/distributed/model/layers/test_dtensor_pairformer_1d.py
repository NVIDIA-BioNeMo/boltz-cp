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

"""Parity tests for 1D CP PairformerLayer1D and PairformerModule1D.

Tests both PairformerNoSeqLayer (pair-only) and full PairformerLayer
(pair + sequence), verifying:
- Forward pass matches serial fp64 reference (eval mode, dropout=0)
- Backward pass (input gradients + parameter gradients) matches serial
- DTensor shape/stride metadata is correct
- Sharding is active (local shape < global shape on sharded dim)
- Replicated parameter gradients are identical across CP ranks
- Gradients are non-zero (guards against vacuous pass)
"""

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.pairformer_1d import (
    PairformerLayer1D,
    PairformerModule1D,
    _apply_dropout_1d,
    _get_dropout_mask_1d,
)
from boltz.model.layers.pairformer import (
    PairformerLayer as SerialPairformerLayer,
)
from boltz.model.layers.pairformer import (
    PairformerModule as SerialPairformerModule,
)
from boltz.model.layers.pairformer import (
    PairformerNoSeqLayer as SerialPairformerNoSeqLayer,
)
from boltz.testing.utils import (
    assert_all_identical,
    assert_no_percentile_upshift,
    assert_tensors_identical,
    get_param_by_key,
    init_module_params_uniform,
    init_tensors_uniform,
    seed_by_rank,
    set_dtype_specific_inf_values,
    spawn_multiprocessing,
)


def _parallel_assert_pairformer_noseq_layer_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    token_z,
    pairwise_head_width,
    pairwise_num_heads,
    layer_state_dict,
    input_z_global_host,
    pair_mask_global_host,
    output_z_expected_global_host,
    d_output_z_expected_global_host,
    d_input_z_expected_global_host,
    expected_param_grads_global_host_dict,
    output_z_global_fp32_host=None,
    d_input_z_global_fp32_host=None,
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

    cp_group = manager.group["cp"]
    device_mesh = manager.device_mesh  # 2D mesh (dp, cp)

    # 1D CP placements
    placements_z = (Shard(0), Shard(1))

    # Create serial module, load weights, convert to target dtype
    module_serial = SerialPairformerNoSeqLayer(
        token_z=token_z,
        dropout=0.0,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        post_layer_norm=False,
    )
    module_serial.to(dtype=dtype)
    module_serial.load_state_dict({k: v.to(dtype=dtype) for k, v in layer_state_dict.items()})
    set_dtype_specific_inf_values(module_serial, dtype)
    module_serial = module_serial.to(device=manager.device)

    # Create distributed module
    module = PairformerLayer1D(module_serial, manager)
    module.eval()  # Eval mode to disable dropout

    # Distribute inputs
    input_z_dtensor = distribute_tensor(
        input_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_z,
    ).requires_grad_(True)

    pair_mask_dtensor = distribute_tensor(
        pair_mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_z,
    )

    d_output_z_expected_dtensor = distribute_tensor(
        d_output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_z,
    )
    output_z_expected_dtensor = distribute_tensor(
        output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_z,
        src_data_rank=None,
    )
    d_input_z_expected_dtensor = distribute_tensor(
        d_input_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_z,
        src_data_rank=None,
    )

    check_error_hist = output_z_global_fp32_host is not None

    # Verify sharding is active
    assert input_z_dtensor.to_local().shape[1] < input_z_dtensor.shape[1], (
        f"Sharding not active: local dim1 {input_z_dtensor.to_local().shape[1]} "
        f"should be < global dim1 {input_z_dtensor.shape[1]}"
    )

    if check_error_hist:
        # Forward + backward
        output_z_dtensor = module(z=input_z_dtensor, pair_mask=pair_mask_dtensor)
        output_z_dtensor.backward(d_output_z_expected_dtensor)

        output_z_fp32_dtensor = distribute_tensor(
            output_z_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
            src_data_rank=None,
        )
        d_input_z_fp32_dtensor = distribute_tensor(
            d_input_z_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            output_z_dtensor.to_local(),
            output_z_expected_dtensor.to_local(),
            output_z_fp32_dtensor.to_local(),
            names_input=("output_z_cp_fp32", "output_z_serial_fp64", "output_z_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_z_dtensor.grad.to_local(),
            d_input_z_expected_dtensor.to_local(),
            d_input_z_fp32_dtensor.to_local(),
            names_input=("d_input_z_cp_fp32", "d_input_z_serial_fp64", "d_input_z_serial_fp32"),
        )

        for name, grad_param_expected_global in expected_param_grads_global_host_dict.items():
            grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
            assert_no_percentile_upshift(
                grad_param_result_global,
                grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                grad_params_fp32_global_host[name],
                names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
            )
    else:
        # Forward pass
        output_z_dtensor = module(z=input_z_dtensor, pair_mask=pair_mask_dtensor)

        # Check output metadata
        assert output_z_dtensor.shape == output_z_expected_dtensor.shape
        assert output_z_dtensor.stride() == output_z_expected_dtensor.stride()

        # Forward parity
        torch.testing.assert_close(output_z_dtensor.to_local(), output_z_expected_dtensor.to_local())
        torch.testing.assert_close(
            output_z_dtensor.full_tensor().cpu(),
            output_z_expected_global_host.to(dtype=dtype),
        )

        # Backward pass
        output_z_dtensor.backward(d_output_z_expected_dtensor)

        # Input gradient parity
        torch.testing.assert_close(input_z_dtensor.grad.to_local(), d_input_z_expected_dtensor.to_local())
        torch.testing.assert_close(
            input_z_dtensor.grad.full_tensor().cpu(),
            d_input_z_expected_global_host.to(dtype=dtype),
        )

        # Verify gradients are non-zero
        assert input_z_dtensor.grad.to_local().abs().max() > 0, "Input gradient is all zeros"
        assert output_z_dtensor.to_local().abs().max() > 0, "Output is all zeros"

        # Parameter gradient parity
        for name, param in module.named_parameters():
            if param.grad is not None:
                assert (
                    name in expected_param_grads_global_host_dict
                ), f"Parameter {name} has a gradient but is not in the reference"
                grad_global = param.grad.full_tensor()
                torch.testing.assert_close(
                    grad_global.cpu(),
                    expected_param_grads_global_host_dict[name].to(dtype=dtype),
                )
                # Replicated param grads must be identical across CP ranks
                assert_all_identical(grad_global, cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype_and_check_error_hist",
    (
        params_noseq := [
            (((1, 2), True, "cuda", "ENV"), (torch.float32, True)),
            (((1, 2), True, "cuda", "ENV"), (torch.float64, False)),
            (((1, 4), True, "cuda", "ENV"), (torch.float64, False)),
            (((2, 2), True, "cuda", "ENV"), (torch.float64, False)),
            (((1, 3), True, "cuda", "ENV"), (torch.float64, False)),
            (((1, 3), True, "cpu", "ENV"), (torch.float64, False)),
        ]
    ),
    indirect=["setup_env"],
)
def test_pairformer_noseq_layer_1d(setup_env, dtype_and_check_error_hist):
    """Test PairformerLayer1D wrapping a serial PairformerNoSeqLayer."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, check_error_hist = dtype_and_check_error_hist

    if device_type == "cpu" and grid_group_sizes["cp"] > 1:
        pytest.skip("DAP ending-node requires NCCL; Gloo lacks all_to_all")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    if check_error_hist or dtype == torch.float64:
        N = cp_size * 64
        token_z = 128
        pairwise_head_width = 32
        pairwise_num_heads = 4
        min_val_init = -0.03
        max_val_init = 0.03
    else:
        N = cp_size * 4
        token_z = 8
        pairwise_head_width = 4
        pairwise_num_heads = 2
        min_val_init = -0.5
        max_val_init = 0.5

    seed = 42
    seed_by_rank(0, seed=seed)

    # Serial reference in fp64
    input_z_global = torch.empty((B, N, N, token_z), dtype=torch.float64, requires_grad=True, device=device_type)
    pair_mask_global = torch.randint(0, 2, (B, N, N), dtype=torch.float64, requires_grad=False, device=device_type)
    # Emulate blocks of pure padding
    pair_mask_global[0, N // cp_size :, :] = 0
    pair_mask_global[0, :, N // cp_size :] = 0

    reference_module = SerialPairformerNoSeqLayer(
        token_z=token_z,
        dropout=0.0,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        post_layer_norm=False,
    )
    init_tensors_uniform([input_z_global], low=min_val_init, high=max_val_init)
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)
    set_dtype_specific_inf_values(reference_module, torch.float64)
    layer_state_dict = reference_module.state_dict()
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).eval()

    output_z_expected = reference_module(input_z_global, pair_mask_global)
    d_output_z = torch.rand_like(output_z_expected)
    output_z_expected.backward(d_output_z)

    grad_params_expected = {
        name: param.grad.detach().clone().cpu() for name, param in reference_module.named_parameters()
    }

    if check_error_hist:
        input_z_global_fp32 = input_z_global.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        pair_mask_global_fp32 = pair_mask_global.detach().to(dtype=torch.float32, copy=True)
        reference_module_fp32 = SerialPairformerNoSeqLayer(
            token_z=token_z,
            dropout=0.0,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=False,
        )
        reference_module_fp32.load_state_dict(layer_state_dict)
        reference_module_fp32 = reference_module_fp32.to(dtype=torch.float32, device=device_type).eval()
        set_dtype_specific_inf_values(reference_module_fp32, torch.float32)

        output_z_fp32 = reference_module_fp32(input_z_global_fp32, pair_mask_global_fp32)
        d_output_z_fp32 = d_output_z.to(dtype=torch.float32)
        output_z_fp32.backward(d_output_z_fp32)

        output_z_global_fp32_host = output_z_fp32.detach().to(device="cpu", copy=True)
        d_input_z_global_fp32_host = input_z_global_fp32.grad.detach().to(device="cpu", copy=True)
        grad_params_fp32_global_host = {
            name: param.grad.detach().to(device="cpu", copy=True)
            for name, param in reference_module_fp32.named_parameters()
        }
    else:
        output_z_global_fp32_host = None
        d_input_z_global_fp32_host = None
        grad_params_fp32_global_host = None

    spawn_multiprocessing(
        _parallel_assert_pairformer_noseq_layer_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        token_z,
        pairwise_head_width,
        pairwise_num_heads,
        layer_state_dict,
        input_z_global.detach().clone().cpu(),
        pair_mask_global.detach().clone().cpu(),
        output_z_expected.detach().clone().cpu(),
        d_output_z.detach().clone().cpu(),
        input_z_global.grad.detach().clone().cpu(),
        grad_params_expected,
        output_z_global_fp32_host,
        d_input_z_global_fp32_host,
        grad_params_fp32_global_host,
    )


def _parallel_assert_pairformer_layer_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    token_s,
    token_z,
    num_heads,
    pairwise_head_width,
    pairwise_num_heads,
    layer_state_dict,
    input_s_global_host,
    input_z_global_host,
    mask_global_host,
    pair_mask_global_host,
    output_s_expected_global_host,
    output_z_expected_global_host,
    d_output_s_expected_global_host,
    d_output_z_expected_global_host,
    d_input_s_expected_global_host,
    d_input_z_expected_global_host,
    expected_param_grads_global_host_dict,
    output_s_global_fp32_host=None,
    output_z_global_fp32_host=None,
    d_input_s_global_fp32_host=None,
    d_input_z_global_fp32_host=None,
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

    cp_group = manager.group["cp"]
    device_mesh = manager.device_mesh

    check_error_hist = output_s_global_fp32_host is not None

    placements_pair = (Shard(0), Shard(1))
    placements_single = (Shard(0), Shard(1))
    placements_mask = (Shard(0), Shard(1))

    # Create serial module and load weights
    module_serial = SerialPairformerLayer(
        token_s=token_s,
        token_z=token_z,
        num_heads=num_heads,
        dropout=0.0,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        post_layer_norm=False,
        v2=True,
    )
    module_serial.to(dtype=dtype)
    module_serial.load_state_dict({k: v.to(dtype=dtype) for k, v in layer_state_dict.items()})
    set_dtype_specific_inf_values(module_serial, dtype)
    module_serial = module_serial.to(device=manager.device)

    module = PairformerLayer1D(module_serial, manager)
    module.eval()

    # Distribute inputs
    input_s_dtensor = distribute_tensor(
        input_s_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
    ).requires_grad_(True)
    input_z_dtensor = distribute_tensor(
        input_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
    ).requires_grad_(True)
    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_mask,
    )
    pair_mask_dtensor = distribute_tensor(
        pair_mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
    )

    # Expected outputs
    output_s_expected_dt = distribute_tensor(
        output_s_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
        src_data_rank=None,
    )
    output_z_expected_dt = distribute_tensor(
        output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
        src_data_rank=None,
    )
    d_output_s_dt = distribute_tensor(
        d_output_s_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
    )
    d_output_z_dt = distribute_tensor(
        d_output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
    )
    d_input_s_expected_dt = distribute_tensor(
        d_input_s_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
        src_data_rank=None,
    )
    d_input_z_expected_dt = distribute_tensor(
        d_input_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
        src_data_rank=None,
    )

    if check_error_hist:
        # Forward + backward
        s_out, z_out = module(s=input_s_dtensor, z=input_z_dtensor, mask=mask_dtensor, pair_mask=pair_mask_dtensor)
        torch.autograd.backward([s_out, z_out], [d_output_s_dt, d_output_z_dt])

        output_s_fp32_dt = distribute_tensor(
            output_s_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        output_z_fp32_dt = distribute_tensor(
            output_z_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )
        d_input_s_fp32_dt = distribute_tensor(
            d_input_s_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        d_input_z_fp32_dt = distribute_tensor(
            d_input_z_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            s_out.to_local(),
            output_s_expected_dt.to_local(),
            output_s_fp32_dt.to_local(),
            names_input=("output_s_cp_fp32", "output_s_serial_fp64", "output_s_serial_fp32"),
        )
        assert_no_percentile_upshift(
            z_out.to_local(),
            output_z_expected_dt.to_local(),
            output_z_fp32_dt.to_local(),
            names_input=("output_z_cp_fp32", "output_z_serial_fp64", "output_z_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_s_dtensor.grad.to_local(),
            d_input_s_expected_dt.to_local(),
            d_input_s_fp32_dt.to_local(),
            names_input=("d_input_s_cp_fp32", "d_input_s_serial_fp64", "d_input_s_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_z_dtensor.grad.to_local(),
            d_input_z_expected_dt.to_local(),
            d_input_z_fp32_dt.to_local(),
            names_input=("d_input_z_cp_fp32", "d_input_z_serial_fp64", "d_input_z_serial_fp32"),
        )

        for name, grad_param_expected_global in expected_param_grads_global_host_dict.items():
            grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
            assert_no_percentile_upshift(
                grad_param_result_global,
                grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                grad_params_fp32_global_host[name],
                names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
            )
    else:
        # Forward
        s_out, z_out = module(s=input_s_dtensor, z=input_z_dtensor, mask=mask_dtensor, pair_mask=pair_mask_dtensor)

        # Forward parity
        torch.testing.assert_close(s_out.to_local(), output_s_expected_dt.to_local())
        torch.testing.assert_close(z_out.to_local(), output_z_expected_dt.to_local())

        # Backward
        torch.autograd.backward([s_out, z_out], [d_output_s_dt, d_output_z_dt])

        # Input gradient parity
        torch.testing.assert_close(input_s_dtensor.grad.to_local(), d_input_s_expected_dt.to_local())
        torch.testing.assert_close(input_z_dtensor.grad.to_local(), d_input_z_expected_dt.to_local())

        # Non-zero gradients
        assert input_s_dtensor.grad.to_local().abs().max() > 0, "s gradient is all zeros"
        assert input_z_dtensor.grad.to_local().abs().max() > 0, "z gradient is all zeros"

        # Parameter gradient parity
        for name, param in module.named_parameters():
            if param.grad is not None:
                assert (
                    name in expected_param_grads_global_host_dict
                ), f"Parameter {name} has a gradient but is not in the reference"
                grad_global = param.grad.full_tensor()
                torch.testing.assert_close(
                    grad_global.cpu(),
                    expected_param_grads_global_host_dict[name].to(dtype=dtype),
                )
                assert_all_identical(grad_global, cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype_and_check_error_hist",
    (
        params_full := [
            (((1, 2), True, "cuda", "ENV"), (torch.float32, True)),
            (((1, 2), True, "cuda", "ENV"), (torch.float64, False)),
            (((1, 4), True, "cuda", "ENV"), (torch.float64, False)),
            (((2, 2), True, "cuda", "ENV"), (torch.float64, False)),
            (((1, 3), True, "cuda", "ENV"), (torch.float64, False)),
            (((1, 3), True, "cpu", "ENV"), (torch.float64, False)),
        ]
    ),
    indirect=["setup_env"],
)
def test_pairformer_layer_1d(setup_env, dtype_and_check_error_hist):
    """Test PairformerLayer1D wrapping a serial PairformerLayer (full, with sequence track)."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, check_error_hist = dtype_and_check_error_hist

    if device_type == "cpu" and grid_group_sizes["cp"] > 1:
        pytest.skip("DAP ending-node requires NCCL; Gloo lacks all_to_all")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    if check_error_hist or dtype == torch.float64:
        N = cp_size * 64
        token_s = 32
        token_z = 128
        num_heads = 16
        pairwise_head_width = 32
        pairwise_num_heads = 4
        min_val_init = -0.03
        max_val_init = 0.03
    else:
        N = cp_size * 4
        token_s = 12
        token_z = 8
        num_heads = 4
        pairwise_head_width = 4
        pairwise_num_heads = 2
        min_val_init = -0.5
        max_val_init = 0.5

    seed = 42
    seed_by_rank(0, seed=seed)

    # Serial reference
    input_s_global = torch.empty((B, N, token_s), dtype=torch.float64, requires_grad=True, device=device_type)
    input_z_global = torch.empty((B, N, N, token_z), dtype=torch.float64, requires_grad=True, device=device_type)
    mask_global = torch.randint(0, 2, (B, N), dtype=torch.float64, requires_grad=False, device=device_type)
    pair_mask_global = torch.randint(0, 2, (B, N, N), dtype=torch.float64, requires_grad=False, device=device_type)
    pair_mask_global[0, N // cp_size :, :] = 0
    pair_mask_global[0, :, N // cp_size :] = 0

    reference_module = SerialPairformerLayer(
        token_s=token_s,
        token_z=token_z,
        num_heads=num_heads,
        dropout=0.0,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
        post_layer_norm=False,
        v2=True,
    )
    init_tensors_uniform([input_s_global, input_z_global], low=min_val_init, high=max_val_init)
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)
    set_dtype_specific_inf_values(reference_module, torch.float64)
    layer_state_dict = reference_module.state_dict()
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).eval()

    s_out_expected, z_out_expected = reference_module(input_s_global, input_z_global, mask_global, pair_mask_global)
    d_s_out = torch.rand_like(s_out_expected)
    d_z_out = torch.rand_like(z_out_expected)
    torch.autograd.backward([s_out_expected, z_out_expected], [d_s_out, d_z_out])

    grad_params_expected = {
        name: param.grad.detach().clone().cpu() for name, param in reference_module.named_parameters()
    }

    if check_error_hist:
        input_s_global_fp32 = input_s_global.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        input_z_global_fp32 = input_z_global.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        mask_global_fp32 = mask_global.detach().to(dtype=torch.float32, copy=True)
        pair_mask_global_fp32 = pair_mask_global.detach().to(dtype=torch.float32, copy=True)

        reference_module_fp32 = SerialPairformerLayer(
            token_s=token_s,
            token_z=token_z,
            num_heads=num_heads,
            dropout=0.0,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
            post_layer_norm=False,
            v2=True,
        )
        reference_module_fp32.load_state_dict(layer_state_dict)
        reference_module_fp32 = reference_module_fp32.to(dtype=torch.float32, device=device_type).eval()
        set_dtype_specific_inf_values(reference_module_fp32, torch.float32)

        s_out_fp32, z_out_fp32 = reference_module_fp32(
            input_s_global_fp32, input_z_global_fp32, mask_global_fp32, pair_mask_global_fp32
        )
        d_s_out_fp32 = d_s_out.to(dtype=torch.float32)
        d_z_out_fp32 = d_z_out.to(dtype=torch.float32)
        torch.autograd.backward([s_out_fp32, z_out_fp32], [d_s_out_fp32, d_z_out_fp32])

        output_s_global_fp32_host = s_out_fp32.detach().to(device="cpu", copy=True)
        output_z_global_fp32_host = z_out_fp32.detach().to(device="cpu", copy=True)
        d_input_s_global_fp32_host = input_s_global_fp32.grad.detach().to(device="cpu", copy=True)
        d_input_z_global_fp32_host = input_z_global_fp32.grad.detach().to(device="cpu", copy=True)
        grad_params_fp32_global_host = {
            name: param.grad.detach().to(device="cpu", copy=True)
            for name, param in reference_module_fp32.named_parameters()
        }
    else:
        output_s_global_fp32_host = None
        output_z_global_fp32_host = None
        d_input_s_global_fp32_host = None
        d_input_z_global_fp32_host = None
        grad_params_fp32_global_host = None

    spawn_multiprocessing(
        _parallel_assert_pairformer_layer_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        token_s,
        token_z,
        num_heads,
        pairwise_head_width,
        pairwise_num_heads,
        layer_state_dict,
        input_s_global.detach().clone().cpu(),
        input_z_global.detach().clone().cpu(),
        mask_global.detach().clone().cpu(),
        pair_mask_global.detach().clone().cpu(),
        s_out_expected.detach().clone().cpu(),
        z_out_expected.detach().clone().cpu(),
        d_s_out.detach().clone().cpu(),
        d_z_out.detach().clone().cpu(),
        input_s_global.grad.detach().clone().cpu(),
        input_z_global.grad.detach().clone().cpu(),
        grad_params_expected,
        output_s_global_fp32_host,
        output_z_global_fp32_host,
        d_input_s_global_fp32_host,
        d_input_z_global_fp32_host,
        grad_params_fp32_global_host,
    )


# ---------------------------------------------------------------------------
# PairformerModule1D module-level tests (multi-block, activation checkpointing)
# ---------------------------------------------------------------------------


def _parallel_assert_pairformer_module_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    pairformer_params,
    module_state_dict,
    input_s_global_host,
    input_z_global_host,
    mask_global_host,
    pair_mask_global_host,
    output_s_expected_global_host,
    output_z_expected_global_host,
    d_output_s_expected_global_host,
    d_output_z_expected_global_host,
    d_input_s_expected_global_host,
    d_input_z_expected_global_host,
    expected_param_grads_global_host_dict,
    output_s_global_fp32_host=None,
    output_z_global_fp32_host=None,
    d_input_s_global_fp32_host=None,
    d_input_z_global_fp32_host=None,
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

    cp_group = manager.group["cp"]
    device_mesh = manager.device_mesh

    check_error_hist = output_s_global_fp32_host is not None

    placements_pair = (Shard(0), Shard(1))
    placements_single = (Shard(0), Shard(1))
    placements_mask = (Shard(0), Shard(1))

    # Create serial module, load weights, convert to target dtype
    module_serial = SerialPairformerModule(**pairformer_params)
    module_serial.to(dtype=dtype)
    module_serial.load_state_dict({k: v.to(dtype=dtype) for k, v in module_state_dict.items()})
    set_dtype_specific_inf_values(module_serial, dtype)
    module_serial = module_serial.to(device=manager.device)

    # Create distributed module
    module = PairformerModule1D(module_serial, manager, cpu_offloading=False)
    module.train()

    # Distribute inputs
    input_s_dtensor = distribute_tensor(
        input_s_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
    ).requires_grad_(True)
    input_z_dtensor = distribute_tensor(
        input_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
    ).requires_grad_(True)
    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_mask,
    )
    pair_mask_dtensor = distribute_tensor(
        pair_mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
    )

    # Expected outputs
    output_s_expected_dt = distribute_tensor(
        output_s_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
        src_data_rank=None,
    )
    output_z_expected_dt = distribute_tensor(
        output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
        src_data_rank=None,
    )
    d_output_s_dt = distribute_tensor(
        d_output_s_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
    )
    d_output_z_dt = distribute_tensor(
        d_output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
    )
    d_input_s_expected_dt = distribute_tensor(
        d_input_s_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_single,
        src_data_rank=None,
    )
    d_input_z_expected_dt = distribute_tensor(
        d_input_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_pair,
        src_data_rank=None,
    )

    # Verify sharding is active
    assert input_z_dtensor.to_local().shape[1] < input_z_dtensor.shape[1], (
        f"Sharding not active: local dim1 {input_z_dtensor.to_local().shape[1]} "
        f"should be < global dim1 {input_z_dtensor.shape[1]}"
    )

    input_s_dtensor_copy = input_s_dtensor.detach().clone().requires_grad_(True)
    input_z_dtensor_copy = input_z_dtensor.detach().clone().requires_grad_(True)

    if check_error_hist:
        # Forward + backward
        s_out, z_out = module(s=input_s_dtensor, z=input_z_dtensor, mask=mask_dtensor, pair_mask=pair_mask_dtensor)
        torch.autograd.backward([s_out, z_out], [d_output_s_dt, d_output_z_dt])

        output_s_fp32_dt = distribute_tensor(
            output_s_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        output_z_fp32_dt = distribute_tensor(
            output_z_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )
        d_input_s_fp32_dt = distribute_tensor(
            d_input_s_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        d_input_z_fp32_dt = distribute_tensor(
            d_input_z_global_fp32_host.to(device=manager.device),
            device_mesh=device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            s_out.to_local(),
            output_s_expected_dt.to_local(),
            output_s_fp32_dt.to_local(),
            names_input=("output_s_cp_fp32", "output_s_serial_fp64", "output_s_serial_fp32"),
        )
        assert_no_percentile_upshift(
            z_out.to_local(),
            output_z_expected_dt.to_local(),
            output_z_fp32_dt.to_local(),
            names_input=("output_z_cp_fp32", "output_z_serial_fp64", "output_z_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_s_dtensor.grad.to_local(),
            d_input_s_expected_dt.to_local(),
            d_input_s_fp32_dt.to_local(),
            names_input=("d_input_s_cp_fp32", "d_input_s_serial_fp64", "d_input_s_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_z_dtensor.grad.to_local(),
            d_input_z_expected_dt.to_local(),
            d_input_z_fp32_dt.to_local(),
            names_input=("d_input_z_cp_fp32", "d_input_z_serial_fp64", "d_input_z_serial_fp32"),
        )

        for name, grad_param_expected_global in expected_param_grads_global_host_dict.items():
            grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
            assert_no_percentile_upshift(
                grad_param_result_global,
                grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                grad_params_fp32_global_host[name],
                names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
            )
    else:
        # Forward pass
        s_out, z_out = module(s=input_s_dtensor, z=input_z_dtensor, mask=mask_dtensor, pair_mask=pair_mask_dtensor)

        # Verify inputs weren't modified
        assert_tensors_identical(
            input_s_dtensor_copy.to_local(), input_s_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )
        assert_tensors_identical(
            input_z_dtensor_copy.to_local(), input_z_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )

        # Forward parity
        torch.testing.assert_close(s_out.to_local(), output_s_expected_dt.to_local())
        torch.testing.assert_close(z_out.to_local(), output_z_expected_dt.to_local())

        # Backward
        d_output_s_dt_copy = d_output_s_dt.detach().clone()
        d_output_z_dt_copy = d_output_z_dt.detach().clone()
        torch.autograd.backward([s_out, z_out], [d_output_s_dt, d_output_z_dt])

        # Verify upstream gradients weren't modified
        assert_tensors_identical(d_output_s_dt_copy.to_local(), d_output_s_dt.to_local())
        assert_tensors_identical(d_output_z_dt_copy.to_local(), d_output_z_dt.to_local())

        # Input gradient parity
        torch.testing.assert_close(input_s_dtensor.grad.to_local(), d_input_s_expected_dt.to_local())
        torch.testing.assert_close(input_z_dtensor.grad.to_local(), d_input_z_expected_dt.to_local())

        # Full tensor parity against serial reference
        output_s_global_result = s_out.full_tensor().cpu()
        output_z_global_result = z_out.full_tensor().cpu()
        d_input_s_global_result = input_s_dtensor.grad.full_tensor().cpu()
        d_input_z_global_result = input_z_dtensor.grad.full_tensor().cpu()
        torch.testing.assert_close(output_s_global_result, output_s_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(output_z_global_result, output_z_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_s_global_result, d_input_s_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_z_global_result, d_input_z_expected_global_host.to(dtype=dtype))

        # Non-zero gradients
        assert input_s_dtensor.grad.to_local().abs().max() > 0, "s gradient is all zeros"
        assert input_z_dtensor.grad.to_local().abs().max() > 0, "z gradient is all zeros"

        # Parameter gradient parity
        result_param_grads = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                assert (
                    name in expected_param_grads_global_host_dict
                ), f"Parameter {name} has a gradient but is not in the reference"
                result_param_grads[name] = param.grad
        for name, expected_grad_global_host in expected_param_grads_global_host_dict.items():
            assert name in result_param_grads, f"Parameter {name}'s gradient is not found in result gradients"
            result_grad_global = result_param_grads[name].full_tensor()
            torch.testing.assert_close(result_grad_global.cpu(), expected_grad_global_host.to(dtype=dtype))
            assert_all_identical(result_grad_global, cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, check_error_hist, activation_checkpointing",
    (
        params_module := [
            (((1, 2), True, "cuda", "ENV"), torch.float32, True, False),
            (((1, 2), True, "cuda", "ENV"), torch.float64, False, False),
            (((2, 2), True, "cuda", "ENV"), torch.float64, False, False),
            (((1, 2), True, "cuda", "ENV"), torch.float64, False, True),
            (((1, 3), True, "cpu", "ENV"), torch.float32, False, False),
        ]
    ),
    indirect=["setup_env"],
)
def test_pairformer_module_1d(setup_env, dtype, check_error_hist, activation_checkpointing):
    """Test PairformerModule1D: multi-block forward/backward parity, activation checkpointing, fp32 error hist."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cpu" and grid_group_sizes["cp"] > 1:
        pytest.skip("DAP ending-node requires NCCL; Gloo lacks all_to_all")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    test_large_model = check_error_hist or dtype == torch.float64

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    if test_large_model:
        N = cp_size * 64
        token_s = 32
        token_z = 128
        num_blocks = 4
        num_heads = 16
        pairwise_head_width = 32
        pairwise_num_heads = 4
        min_val_init = -0.03
        max_val_init = 0.03
    else:
        N = cp_size * 4
        token_s = 8
        token_z = 12
        num_blocks = 2
        num_heads = 4
        pairwise_head_width = 4
        pairwise_num_heads = 2
        min_val_init = -0.5
        max_val_init = 0.5
    dropout = 0.0
    post_layer_norm = False

    pairformer_params = {
        "token_s": token_s,
        "token_z": token_z,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "dropout": dropout,
        "pairwise_head_width": pairwise_head_width,
        "pairwise_num_heads": pairwise_num_heads,
        "post_layer_norm": post_layer_norm,
        "activation_checkpointing": activation_checkpointing,
        "v2": True,
    }

    seed = 42
    seed_by_rank(0, seed=seed)

    # Compute reference results with fp64
    input_s_global_fp64 = torch.empty((B, N, token_s), dtype=torch.float64, requires_grad=True, device=device_type)
    input_z_global_fp64 = torch.empty((B, N, N, token_z), dtype=torch.float64, requires_grad=True, device=device_type)
    mask_global_fp64 = torch.ones((B, N), dtype=torch.float64, requires_grad=False, device=device_type)
    mask_global_fp64[0, N // cp_size :] = 0
    pair_mask_global_fp64 = torch.randint(0, 2, (B, N, N), dtype=torch.float64, requires_grad=False, device=device_type)
    pair_mask_global_fp64[0, N // cp_size :, :] = 0
    pair_mask_global_fp64[0, :, N // cp_size :] = 0

    reference_module = SerialPairformerModule(**pairformer_params)
    init_tensors_uniform([input_s_global_fp64, input_z_global_fp64], low=min_val_init, high=max_val_init)
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)
    set_dtype_specific_inf_values(reference_module, torch.float64)

    reference_module = reference_module.to(dtype=torch.float64, device=device_type).train()
    module_state_dict_fp64 = reference_module.state_dict()

    output_s_expected_fp64, output_z_expected_fp64 = reference_module(
        s=input_s_global_fp64,
        z=input_z_global_fp64,
        mask=mask_global_fp64,
        pair_mask=pair_mask_global_fp64,
    )
    d_output_s_fp64 = torch.rand_like(output_s_expected_fp64)
    d_output_z_fp64 = torch.rand_like(output_z_expected_fp64)
    torch.autograd.backward(
        [output_s_expected_fp64, output_z_expected_fp64],
        [d_output_s_fp64, d_output_z_fp64],
    )

    grad_params_fp64_expected_global_host = {
        name: param.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        for name, param in reference_module.named_parameters()
    }

    del reference_module
    if device_type == "cuda":
        torch.cuda.empty_cache()

    if check_error_hist:
        input_s_global_fp32 = input_s_global_fp64.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        input_z_global_fp32 = input_z_global_fp64.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        mask_global_fp32 = mask_global_fp64.detach().to(dtype=torch.float32, copy=True)
        pair_mask_global_fp32 = pair_mask_global_fp64.detach().to(dtype=torch.float32, copy=True)
        reference_module_fp32 = SerialPairformerModule(**pairformer_params)
        reference_module_fp32.load_state_dict(module_state_dict_fp64)
        reference_module_fp32 = reference_module_fp32.to(dtype=torch.float32, device=device_type).train()
        set_dtype_specific_inf_values(reference_module_fp32, torch.float32)
        output_s_fp32, output_z_fp32 = reference_module_fp32(
            s=input_s_global_fp32,
            z=input_z_global_fp32,
            mask=mask_global_fp32,
            pair_mask=pair_mask_global_fp32,
        )
        d_output_s_fp32 = d_output_s_fp64.to(dtype=torch.float32)
        d_output_z_fp32 = d_output_z_fp64.to(dtype=torch.float32)
        torch.autograd.backward(
            [output_s_fp32, output_z_fp32],
            [d_output_s_fp32, d_output_z_fp32],
        )
        output_s_global_fp32_host = output_s_fp32.detach().to(device="cpu", copy=True)
        output_z_global_fp32_host = output_z_fp32.detach().to(device="cpu", copy=True)
        d_input_s_global_fp32_host = input_s_global_fp32.grad.detach().to(device="cpu", copy=True)
        d_input_z_global_fp32_host = input_z_global_fp32.grad.detach().to(device="cpu", copy=True)
        grad_params_fp32_global_host = {
            name: param.grad.detach().to(device="cpu", copy=True)
            for name, param in reference_module_fp32.named_parameters()
        }
    else:
        output_s_global_fp32_host = None
        output_z_global_fp32_host = None
        d_input_s_global_fp32_host = None
        d_input_z_global_fp32_host = None
        grad_params_fp32_global_host = None

    spawn_multiprocessing(
        _parallel_assert_pairformer_module_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        pairformer_params,
        module_state_dict_fp64,
        input_s_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        input_z_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        mask_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        pair_mask_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        output_s_expected_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        output_z_expected_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        d_output_s_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        d_output_z_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        input_s_global_fp64.grad.detach().to(dtype=dtype, device="cpu", copy=True),
        input_z_global_fp64.grad.detach().to(dtype=dtype, device="cpu", copy=True),
        grad_params_fp64_expected_global_host,
        output_s_global_fp32_host,
        output_z_global_fp32_host,
        d_input_s_global_fp32_host,
        d_input_z_global_fp32_host,
        grad_params_fp32_global_host,
    )


# ---------------------------------------------------------------------------
# Dropout gradient and column-broadcast tests
# ---------------------------------------------------------------------------


def _parallel_assert_dropout_1d_gradients(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
):
    """Verify _apply_dropout_1d passes gradients and column masks are identical across ranks."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    cp_group = manager.group["cp"]
    device_mesh = manager.device_mesh

    from torch.distributed.tensor import DTensor

    from boltz.distributed.utils import update_exhaustive_strides

    B, N_local, N_full, C = 2, 4, 8, 6
    dtype = torch.float64

    # Create a DTensor input with requires_grad
    x_local = torch.randn(B, N_local, N_full, C, dtype=dtype, device=manager.device)
    x_global_shape = torch.Size([B * grid_group_sizes["dp"], N_local * grid_group_sizes["cp"], N_full, C])
    x_stride = update_exhaustive_strides(x_local.shape, x_local.stride(), x_global_shape)
    placements = (Shard(0), Shard(1))
    x_dt = DTensor.from_local(
        x_local,
        device_mesh,
        placements,
        shape=x_global_shape,
        stride=x_stride,
    ).requires_grad_(True)

    dropout_rate = 0.5
    grad_output_local = torch.randn_like(x_local)
    grad_output_stride = update_exhaustive_strides(grad_output_local.shape, grad_output_local.stride(), x_global_shape)
    grad_output_dt = DTensor.from_local(
        grad_output_local,
        device_mesh,
        placements,
        shape=x_global_shape,
        stride=grad_output_stride,
    )

    # --- Test 1: rowwise dropout gradient flow ---
    out_row = _apply_dropout_1d(x_dt, dropout_rate, True, False, cp_group)
    out_row.backward(grad_output_dt)
    assert x_dt.grad is not None, "Gradient did not flow through rowwise _apply_dropout_1d"
    # With 50% dropout, ~50% of grads should be non-zero (scaled by 2x)
    # and ~50% should be zero. Check that not all are zero and not all are non-zero.
    grad_local = x_dt.grad.to_local()
    nonzero_frac = (grad_local.abs() > 0).float().mean().item()
    assert (
        0.1 < nonzero_frac < 0.9
    ), f"Rowwise dropout gradient nonzero fraction {nonzero_frac:.2f} is outside expected range"

    # --- Test 1b: verify exact gradient values against serial reference ---
    # The forward computed out_local = x_local * mask / (1 - dropout_rate), so
    # d_x_local = grad_output_local * mask / (1 - dropout_rate).
    # Reconstruct the effective mask from the output: where x_local != 0,
    # mask_eff = out_local / x_local; where x_local == 0, mask_eff is ambiguous,
    # but grad should be grad_output * mask_eff regardless.
    out_row_local = out_row.to_local().detach()
    x_local_detached = x_local.detach()
    # Recover the effective scaling mask from the forward pass
    eff_mask_row = torch.where(
        x_local_detached.abs() > 1e-30,
        out_row_local / x_local_detached,
        torch.zeros_like(x_local_detached),
    )
    expected_grad_row = grad_output_local * eff_mask_row
    torch.testing.assert_close(grad_local, expected_grad_row)

    # Reset grad
    x_dt2 = DTensor.from_local(
        x_local.clone(),
        device_mesh,
        placements,
        shape=x_global_shape,
        stride=x_stride,
    ).requires_grad_(True)

    # --- Test 2: columnwise dropout gradient flow ---
    out_col = _apply_dropout_1d(x_dt2, dropout_rate, True, True, cp_group)
    out_col.backward(grad_output_dt)
    assert x_dt2.grad is not None, "Gradient did not flow through columnwise _apply_dropout_1d"
    grad_local_col = x_dt2.grad.to_local()
    nonzero_frac_col = (grad_local_col.abs() > 0).float().mean().item()
    assert (
        0.1 < nonzero_frac_col < 0.9
    ), f"Columnwise dropout gradient nonzero fraction {nonzero_frac_col:.2f} is outside expected range"

    # --- Test 2b: verify exact gradient values against serial reference ---
    out_col_local = out_col.to_local().detach()
    x_local2_detached = x_local.detach()
    eff_mask_col = torch.where(
        x_local2_detached.abs() > 1e-30,
        out_col_local / x_local2_detached,
        torch.zeros_like(x_local2_detached),
    )
    expected_grad_col = grad_output_local * eff_mask_col
    torch.testing.assert_close(grad_local_col, expected_grad_col)

    # --- Test 3: columnwise mask is identical across CP ranks ---
    # Generate column dropout mask and verify all ranks get the same mask
    z_local = torch.randn(B, N_local, N_full, C, dtype=dtype, device=manager.device)
    col_mask = _get_dropout_mask_1d(dropout_rate, z_local, True, True, cp_group)
    # col_mask shape: [B, 1, N_full, 1] — should be identical across CP ranks
    assert_all_identical(col_mask, cp_group)

    # --- Test 4: rowwise mask differs across CP ranks (different rows) ---
    row_mask = _get_dropout_mask_1d(dropout_rate, z_local, True, False, cp_group)
    # row_mask shape: [B, N_local, 1, 1] — may differ across ranks
    # (with high probability for non-trivial N_local and dropout rate)
    # We don't assert they differ (could be coincidence), but we do assert shape is correct
    assert row_mask.shape == (B, N_local, 1, 1), f"Row mask shape {row_mask.shape} unexpected"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    (
        params_dropout := [
            ((1, 2), True, "cuda", "ENV"),
            ((1, 3), True, "cuda", "ENV"),
            ((1, 3), True, "cpu", "ENV"),
        ]
    ),
    indirect=["setup_env"],
    ids=[f"dp:{x[0][0]}, cp:{x[0][1]}, device_type:{x[2]}" for x in params_dropout],
)
def test_dropout_1d_gradient_and_broadcast(setup_env):
    """Verify _apply_dropout_1d passes gradients and column masks are broadcast across CP ranks."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        _parallel_assert_dropout_1d_gradients,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
    )
