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

"""Tests for 1D CP MSALayer1D and MSAModule1D.

Verifies forward/backward parity against the serial MSALayer/MSAModule
using the 1D CP placements:
- MSA ``[B, S, N, C]``: ``(Shard(0), Shard(2))``
- Pair ``[B, N, N, C]``: ``(Shard(0), Shard(1))``
"""

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.data import const
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.modules.msa_1d import (
    MSA_PLACEMENTS_1D,
    PAIR_PLACEMENTS_1D,
    MSALayer1D,
    MSAModule1D,
    PairWeightedAveraging1DTiled,
)
from boltz.model.layers.pair_averaging import PairWeightedAveraging as SerialPairWeightedAveraging
from boltz.model.modules.trunkv2 import MSALayer as SerialMSALayer
from boltz.model.modules.trunkv2 import MSAModule as SerialMSAModule
from boltz.testing.utils import (
    assert_all_identical,
    assert_no_percentile_upshift,
    assert_tensors_identical,
    create_msa_module_init_params_v2,
    get_param_by_key,
    init_module_params_uniform,
    init_tensors_uniform,
    seed_by_rank,
    set_dtype_specific_inf_values,
    spawn_multiprocessing,
)


def parallel_assert_msa_layer_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    msa_s,
    token_z,
    msa_dropout,
    z_dropout,
    pairwise_head_width,
    pairwise_num_heads,
    layer_state_dict,
    input_z_global_host,
    input_m_global_host,
    token_mask_global_host,
    msa_mask_global_host,
    output_z_expected_global_host,
    output_m_expected_global_host,
    d_output_z_global_host,
    d_output_m_global_host,
    d_input_z_expected_global_host,
    d_input_m_expected_global_host,
    expected_param_grads_global_host_dict,
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

    # 1D CP: 2D mesh (dp, cp), cp_group is the flat "cp" group
    cp_group = manager.group["cp"]

    # Create serial module and load weights
    module_serial = SerialMSALayer(
        msa_s=msa_s,
        token_z=token_z,
        msa_dropout=msa_dropout,
        z_dropout=z_dropout,
        pairwise_head_width=pairwise_head_width,
        pairwise_num_heads=pairwise_num_heads,
    ).to(dtype=dtype, device=manager.device)
    module_serial.load_state_dict(layer_state_dict)
    set_dtype_specific_inf_values(module_serial, dtype)

    # Create distributed 1D module
    module = MSALayer1D(module_serial, manager)
    module.train()

    # Distribute inputs with 1D CP placements
    input_z = distribute_tensor(
        input_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=PAIR_PLACEMENTS_1D,
    ).requires_grad_(True)

    input_m = distribute_tensor(
        input_m_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=MSA_PLACEMENTS_1D,
    ).requires_grad_(True)

    token_mask = distribute_tensor(
        token_mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=PAIR_PLACEMENTS_1D,
    )

    msa_mask = distribute_tensor(
        msa_mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=MSA_PLACEMENTS_1D,
    )

    # Forward pass
    output_z, output_m = module(input_z, input_m, token_mask, msa_mask)

    # Verify forward results via full_tensor
    output_z_global = output_z.full_tensor().cpu()
    output_m_global = output_m.full_tensor().cpu()
    torch.testing.assert_close(output_z_global, output_z_expected_global_host.to(dtype=dtype))
    torch.testing.assert_close(output_m_global, output_m_expected_global_host.to(dtype=dtype))

    # Backward pass with explicit random grad_output
    d_output_z = distribute_tensor(
        d_output_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=PAIR_PLACEMENTS_1D,
    )
    d_output_m = distribute_tensor(
        d_output_m_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=MSA_PLACEMENTS_1D,
    )
    torch.autograd.backward([output_z, output_m], [d_output_z, d_output_m])

    # Verify input gradients
    d_input_z_global = input_z.grad.full_tensor().cpu()
    d_input_m_global = input_m.grad.full_tensor().cpu()
    torch.testing.assert_close(d_input_z_global, d_input_z_expected_global_host.to(dtype=dtype))
    torch.testing.assert_close(d_input_m_global, d_input_m_expected_global_host.to(dtype=dtype))

    # I2: Non-zero gradient assertions
    assert input_z.grad.to_local().abs().max() > 0, "input_z grad is all zeros"
    assert input_m.grad.to_local().abs().max() > 0, "input_m grad is all zeros"

    # Verify parameter gradients
    for name, expected_grad in expected_param_grads_global_host_dict.items():
        param = dict(module.named_parameters())[name]
        assert param.grad is not None, f"Parameter {name} has no gradient"
        result_grad_global = param.grad.full_tensor().cpu()
        torch.testing.assert_close(result_grad_global, expected_grad.to(dtype=dtype))

    # I1: Replicated parameter gradients must be identical across CP ranks
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert_all_identical(param.grad.full_tensor(), cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype",
    (
        params_test := [
            # CUDA tests - 1D CP with various mesh configs
            (((1, 2), True, "cuda", "ENV"), torch.float64),
            (((1, 4), True, "cuda", "ENV"), torch.float64),
            (((2, 2), True, "cuda", "ENV"), torch.float64),
            # CUDA test - non-power-of-two CP for GPU CI with 3+ GPUs
            (((1, 3), True, "cuda", "ENV"), torch.float64),
            # CPU test - non-power-of-two CP for CPU-only CI
            (((1, 3), True, "cpu", "ENV"), torch.float64),
        ]
    ),
    indirect=["setup_env"],
)
def test_msa_layer_1d(setup_env, dtype):
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cpu" and grid_group_sizes["cp"] > 1:
        pytest.skip("DAP ending-node requires NCCL; Gloo lacks all_to_all")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 4
    S = cp_size * 6
    msa_s = 8
    token_z = 12
    pairwise_head_width = 4
    pairwise_num_heads = 2
    msa_dropout = 0.0
    z_dropout = 0.0

    gen = torch.Generator(device="cpu").manual_seed(42)

    # Create serial reference in FP64
    input_z = torch.empty(B, N, N, token_z, dtype=torch.float64, device=device_type, requires_grad=True)
    input_m = torch.empty(B, S, N, msa_s, dtype=torch.float64, device=device_type, requires_grad=True)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed())
        init_tensors_uniform([input_z, input_m], low=-0.5, high=0.5)

    token_mask = torch.ones(B, N, N, dtype=torch.float64, device=device_type)
    msa_mask = torch.ones(B, S, N, dtype=torch.float64, device=device_type)

    ref_module = (
        SerialMSALayer(
            msa_s=msa_s,
            token_z=token_z,
            msa_dropout=msa_dropout,
            z_dropout=z_dropout,
            pairwise_head_width=pairwise_head_width,
            pairwise_num_heads=pairwise_num_heads,
        )
        .to(dtype=torch.float64, device=device_type)
        .train()
    )
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed() + 1)
        init_module_params_uniform(ref_module, low=-0.5, high=0.5)
    set_dtype_specific_inf_values(ref_module, torch.float64)

    state_dict = ref_module.state_dict()

    # Forward
    out_z, out_m = ref_module(input_z, input_m, token_mask, msa_mask)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed() + 2)
        d_out_z = torch.rand_like(out_z)
        d_out_m = torch.rand_like(out_m)
    torch.autograd.backward([out_z, out_m], [d_out_z, d_out_m])

    grad_params = {name: p.grad.detach().cpu() for name, p in ref_module.named_parameters()}

    spawn_multiprocessing(
        parallel_assert_msa_layer_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        msa_s,
        token_z,
        msa_dropout,
        z_dropout,
        pairwise_head_width,
        pairwise_num_heads,
        state_dict,
        input_z.detach().cpu(),
        input_m.detach().cpu(),
        token_mask.detach().cpu(),
        msa_mask.detach().cpu(),
        out_z.detach().cpu(),
        out_m.detach().cpu(),
        d_out_z.detach().cpu(),
        d_out_m.detach().cpu(),
        input_z.grad.detach().cpu(),
        input_m.grad.detach().cpu(),
        grad_params,
    )


# ---------------------------------------------------------------------------
# MSAModule1D tests (module-level: multi-layer, activation checkpointing,
# check_error_hist)
# ---------------------------------------------------------------------------


def _feats_for_distributed_1d(feats_global, dtype, device="cpu"):
    """Convert feats to the target dtype/device for the distributed 1D module.

    Integer features (e.g. ``msa``) are kept as-is because the distributed
    MSAModule1D applies ``shardwise_one_hot`` internally.
    """
    out = {}
    for key, value in feats_global.items():
        if value.dtype.is_floating_point:
            out[key] = value.to(dtype=dtype, device=device)
        else:
            out[key] = value.to(device=device)
    return out


def parallel_assert_msa_module_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    msa_module_params,
    module_state_dict,
    input_z_global_host,
    input_emb_global_host,
    input_feats_global_host,
    output_z_expected_global_host,
    d_output_z_expected_global_host,
    d_input_z_expected_global_host,
    d_input_emb_expected_global_host,
    expected_param_grads_global_host_dict,
    output_z_global_fp32_host=None,
    d_input_z_global_fp32_host=None,
    d_input_emb_global_fp32_host=None,
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

    check_error_hist = output_z_global_fp32_host is not None

    module_serial = SerialMSAModule(**msa_module_params)
    module_serial = module_serial.to(dtype=dtype, device=manager.device)
    module_serial.load_state_dict(module_state_dict)
    set_dtype_specific_inf_values(module_serial, dtype)

    module = MSAModule1D(module_serial, manager)
    assert module.activation_checkpointing == msa_module_params["activation_checkpointing"]
    module.train()

    placements_z = PAIR_PLACEMENTS_1D  # (Shard(0), Shard(1))
    placements_emb = PAIR_PLACEMENTS_1D  # (Shard(0), Shard(1)) — emb is [B, N, C_s]
    placements_msa = MSA_PLACEMENTS_1D  # (Shard(0), Shard(2))
    placements_token_mask = PAIR_PLACEMENTS_1D  # (Shard(0), Shard(1))

    input_z_dtensor = distribute_tensor(
        input_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_z,
    ).requires_grad_(True)

    input_emb_dtensor = distribute_tensor(
        input_emb_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_emb,
    ).requires_grad_(True)

    input_feats_dtensor = {}
    for key, value in input_feats_global_host.items():
        if key in ["msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"]:
            input_feats_dtensor[key] = distribute_tensor(
                value.to(dtype=dtype if value.dtype.is_floating_point else value.dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_msa,
            )
        elif key == "token_pair_pad_mask":
            input_feats_dtensor[key] = distribute_tensor(
                value.to(dtype=dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_token_mask,
            )

    d_output_z_expected_dtensor = distribute_tensor(
        d_output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_z,
    )
    output_z_expected_dtensor = distribute_tensor(
        output_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_z,
        src_data_rank=None,
    )
    d_input_z_expected_dtensor = distribute_tensor(
        d_input_z_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_z,
        src_data_rank=None,
    )
    d_input_emb_expected_dtensor = distribute_tensor(
        d_input_emb_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_emb,
        src_data_rank=None,
    )

    if check_error_hist:
        output_z_dtensor_result = module(input_z_dtensor, input_emb_dtensor, input_feats_dtensor)
        torch.autograd.backward([output_z_dtensor_result], [d_output_z_expected_dtensor])

        output_z_fp32_dtensor = distribute_tensor(
            output_z_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_z,
            src_data_rank=None,
        )
        d_input_z_fp32_dtensor = distribute_tensor(
            d_input_z_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_z,
            src_data_rank=None,
        )
        d_input_emb_fp32_dtensor = distribute_tensor(
            d_input_emb_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_emb,
            src_data_rank=None,
        )

        assert_no_percentile_upshift(
            output_z_dtensor_result.to_local(),
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

        assert_no_percentile_upshift(
            input_emb_dtensor.grad.to_local(),
            d_input_emb_expected_dtensor.to_local(),
            d_input_emb_fp32_dtensor.to_local(),
            names_input=("d_input_emb_cp_fp32", "d_input_emb_serial_fp64", "d_input_emb_serial_fp32"),
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
        input_z_dtensor_copy = input_z_dtensor.detach().clone().requires_grad_(True)
        input_emb_dtensor_copy = input_emb_dtensor.detach().clone().requires_grad_(True)

        output_z_dtensor_result = module(input_z_dtensor, input_emb_dtensor, input_feats_dtensor)

        # Verify inputs weren't modified
        assert_tensors_identical(
            input_z_dtensor_copy.to_local(), input_z_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )
        assert_tensors_identical(
            input_emb_dtensor_copy.to_local(), input_emb_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )

        # Forward parity (local shard comparison)
        torch.testing.assert_close(output_z_dtensor_result.to_local(), output_z_expected_dtensor.to_local())

        # Backward pass
        torch.autograd.backward([output_z_dtensor_result], [d_output_z_expected_dtensor])

        # Input gradient parity (local shard comparison)
        torch.testing.assert_close(input_z_dtensor.grad.to_local(), d_input_z_expected_dtensor.to_local())
        torch.testing.assert_close(input_emb_dtensor.grad.to_local(), d_input_emb_expected_dtensor.to_local())

        # Full tensor parity against serial reference
        output_z_global_result_host = output_z_dtensor_result.full_tensor().cpu()
        d_input_z_global_result_host = input_z_dtensor.grad.full_tensor().cpu()
        d_input_emb_global_result_host = input_emb_dtensor.grad.full_tensor().cpu()

        torch.testing.assert_close(output_z_global_result_host, output_z_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_z_global_result_host, d_input_z_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_emb_global_result_host, d_input_emb_expected_global_host.to(dtype=dtype))

        # Parameter gradient parity
        result_param_grads_dict = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                if name not in expected_param_grads_global_host_dict:
                    raise ValueError(f"Parameter {name} has a resulting gradient but it is not in the reference module")
                result_param_grads_dict[name] = param.grad

        for name, expected_grad_global_host in expected_param_grads_global_host_dict.items():
            assert name in result_param_grads_dict, f"Parameter {name}'s gradient is not found in result gradients"
            result_grad = result_param_grads_dict[name]
            result_grad_global = result_grad.full_tensor()
            torch.testing.assert_close(result_grad_global.cpu(), expected_grad_global_host.to(dtype=dtype))
            assert_all_identical(result_grad_global, cp_group)

        # Guard: CP sharding is active on z (mesh dim 1 = cp)
        cp_placement = input_z_dtensor.placements[1]
        if isinstance(cp_placement, Shard):
            local_size = input_z_dtensor.to_local().shape[cp_placement.dim]
            global_size = input_z_dtensor.shape[cp_placement.dim]
            assert (
                local_size < global_size
            ), f"CP sharding not active on dim {cp_placement.dim}: local={local_size}, global={global_size}"

        # Guard: non-zero gradients
        assert input_z_dtensor.grad.to_local().abs().max() > 0, "input_z grad is all zeros"
        assert input_emb_dtensor.grad.to_local().abs().max() > 0, "input_emb grad is all zeros"
        for name, param in module.named_parameters():
            if param.grad is not None:
                assert param.grad.to_local().abs().sum() > 0, f"Gradient for {name} is all zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, check_error_hist, activation_checkpointing",
    (
        params_test_module := [
            # CUDA tests
            (((1, 2), True, "cuda", "ENV"), torch.float32, True, False),
            (((1, 2), True, "cuda", "ENV"), torch.float64, False, False),
            (((2, 2), True, "cuda", "ENV"), torch.float64, True, False),
            (((1, 2), True, "cuda", "ENV"), torch.float64, False, True),  # actv ckpt
            # CPU test - non-power-of-two CP for CPU-only CI
            (((1, 3), True, "cpu", "ENV"), torch.float32, False, False),
        ]
    ),
    indirect=["setup_env"],
)
def test_msa_module_1d(setup_env, dtype, check_error_hist, activation_checkpointing):
    """Test MSAModule1D against serial MSAModule (forward, backward, param grads)."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cpu" and grid_group_sizes["cp"] > 1:
        pytest.skip("DAP ending-node requires NCCL; Gloo lacks all_to_all")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    test_large_model = check_error_hist or dtype == torch.float64
    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    if test_large_model:
        N = cp_size * 64
        S = cp_size * 64
        min_val_init = -0.01
        max_val_init = 0.01
    else:
        N = cp_size * 4
        S = cp_size * 6
        min_val_init = -0.5
        max_val_init = 0.5

    msa_module_params = create_msa_module_init_params_v2(test_large_model)
    msa_module_params["activation_checkpointing"] = activation_checkpointing

    seed = 42
    seed_by_rank(0, seed=seed)

    input_z_global_fp64 = torch.empty(
        (B, N, N, msa_module_params["token_z"]), dtype=torch.float64, requires_grad=True, device=device_type
    )
    input_emb_global_fp64 = torch.empty(
        (B, N, msa_module_params["token_s"]), dtype=torch.float64, requires_grad=True, device=device_type
    )

    dim_input_msa = const.num_tokens
    input_feats_global_fp64 = {
        "msa": torch.randint(0, dim_input_msa, (B, S, N), dtype=torch.int64, device=device_type),
        "has_deletion": torch.empty((B, S, N), dtype=torch.float64, device=device_type),
        "deletion_value": torch.empty((B, S, N), dtype=torch.float64, device=device_type),
        "msa_paired": torch.randint(0, 2, (B, S, N), dtype=torch.float64, device=device_type),
        "msa_mask": torch.ones((B, S, N), dtype=torch.float64, device=device_type),
        "token_pad_mask": torch.randint(0, 2, (B, N), dtype=torch.float64, device=device_type),
    }
    input_feats_global_fp64["token_pad_mask"][0, N // cp_size :] = 0
    input_feats_global_fp64["token_pair_pad_mask"] = (
        input_feats_global_fp64["token_pad_mask"][:, :, None] * input_feats_global_fp64["token_pad_mask"][:, None, :]
    )
    input_feats_global_fp64["msa_mask"][0, (S // cp_size) :, :] = 0
    input_feats_global_fp64["msa_mask"][0, :, (N // cp_size) :] = 0

    reference_module = SerialMSAModule(**msa_module_params)

    init_tensors_uniform([input_z_global_fp64, input_emb_global_fp64], low=min_val_init, high=max_val_init)
    for key, tensor in input_feats_global_fp64.items():
        if tensor.dtype.is_floating_point and "mask" not in key and "msa_paired" not in key:
            init_tensors_uniform([tensor], low=min_val_init, high=max_val_init)

    reference_module = reference_module.to(dtype=torch.float64, device=device_type).train()
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)
    set_dtype_specific_inf_values(reference_module, torch.float64)

    module_state_dict_fp64 = reference_module.state_dict()

    output_z_expected_global_fp64 = reference_module(
        input_z_global_fp64, input_emb_global_fp64, input_feats_global_fp64
    )
    d_output_z_expected_global_fp64 = torch.rand_like(output_z_expected_global_fp64)
    torch.autograd.backward([output_z_expected_global_fp64], [d_output_z_expected_global_fp64])

    grad_params_fp64_expected_global_host = {
        name: param.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        for name, param in reference_module.named_parameters()
        if param.grad is not None
    }

    if check_error_hist:
        # Run serial FP32 reference for three-way error histogram comparison
        input_z_global_fp32 = input_z_global_fp64.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        input_emb_global_fp32 = input_emb_global_fp64.detach().to(dtype=torch.float32, copy=True).requires_grad_(True)
        input_feats_global_fp32 = {}
        for key, tensor in input_feats_global_fp64.items():
            if key == "msa":
                input_feats_global_fp32[key] = tensor.detach().clone()
            elif tensor.dtype.is_floating_point:
                input_feats_global_fp32[key] = tensor.detach().to(dtype=torch.float32, copy=True)
            else:
                input_feats_global_fp32[key] = tensor.detach().clone()

        reference_module_fp32 = SerialMSAModule(**msa_module_params)
        reference_module_fp32.load_state_dict(module_state_dict_fp64)
        set_dtype_specific_inf_values(reference_module_fp32, torch.float32)
        reference_module_fp32 = reference_module_fp32.to(dtype=torch.float32, device=device_type).train()

        output_z_global_fp32 = reference_module_fp32(
            input_z_global_fp32, input_emb_global_fp32, input_feats_global_fp32
        )
        d_output_z_expected_global_fp32 = d_output_z_expected_global_fp64.to(dtype=torch.float32)
        torch.autograd.backward([output_z_global_fp32], [d_output_z_expected_global_fp32])

        output_z_global_fp32_host = output_z_global_fp32.detach().to(device="cpu", copy=True)
        d_input_z_global_fp32_host = input_z_global_fp32.grad.detach().to(device="cpu", copy=True)
        d_input_emb_global_fp32_host = input_emb_global_fp32.grad.detach().to(device="cpu", copy=True)
        grad_params_fp32_global_host = {
            name: param.grad.detach().to(device="cpu", copy=True)
            for name, param in reference_module_fp32.named_parameters()
            if param.grad is not None
        }

        output_z_for_worker = output_z_expected_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True)
        d_input_z_for_worker = input_z_global_fp64.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        d_input_emb_for_worker = input_emb_global_fp64.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        grad_params_for_worker = grad_params_fp64_expected_global_host
    elif dtype == torch.float32:
        # FP32 without error histogram: run serial FP32 reference to match
        # parameter truncation (FP64 state_dict loaded into FP32 module).
        ref_fp32 = SerialMSAModule(**msa_module_params)
        ref_fp32.load_state_dict(module_state_dict_fp64)
        set_dtype_specific_inf_values(ref_fp32, torch.float32)
        ref_fp32 = ref_fp32.to(dtype=torch.float32, device=device_type).train()

        inp_z = input_z_global_fp64.detach().to(dtype=torch.float32, device=device_type).requires_grad_(True)
        inp_emb = input_emb_global_fp64.detach().to(dtype=torch.float32, device=device_type).requires_grad_(True)
        inp_feats = {}
        for key, tensor in input_feats_global_fp64.items():
            if key == "msa":
                inp_feats[key] = tensor.detach().clone()
            elif tensor.dtype.is_floating_point:
                inp_feats[key] = tensor.detach().to(dtype=torch.float32, device=device_type)
            else:
                inp_feats[key] = tensor.detach().clone().to(device=device_type)

        out_z = ref_fp32(inp_z, inp_emb, inp_feats)
        d_out_z = d_output_z_expected_global_fp64.to(dtype=torch.float32)
        torch.autograd.backward([out_z], [d_out_z])

        output_z_for_worker = out_z.detach().cpu()
        d_input_z_for_worker = inp_z.grad.detach().cpu()
        d_input_emb_for_worker = inp_emb.grad.detach().cpu()
        grad_params_for_worker = {
            name: param.grad.detach().cpu() for name, param in ref_fp32.named_parameters() if param.grad is not None
        }

        output_z_global_fp32_host = None
        d_input_z_global_fp32_host = None
        d_input_emb_global_fp32_host = None
        grad_params_fp32_global_host = None
    else:
        # FP64: use FP64 reference directly
        output_z_for_worker = output_z_expected_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True)
        d_input_z_for_worker = input_z_global_fp64.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        d_input_emb_for_worker = input_emb_global_fp64.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        grad_params_for_worker = grad_params_fp64_expected_global_host

        output_z_global_fp32_host = None
        d_input_z_global_fp32_host = None
        d_input_emb_global_fp32_host = None
        grad_params_fp32_global_host = None

    input_feats_for_distributed = _feats_for_distributed_1d(input_feats_global_fp64, dtype, device="cpu")

    spawn_multiprocessing(
        parallel_assert_msa_module_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        msa_module_params,
        module_state_dict_fp64,
        input_z_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        input_emb_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        input_feats_for_distributed,
        output_z_for_worker,
        d_output_z_expected_global_fp64.detach().to(dtype=dtype, device="cpu", copy=True),
        d_input_z_for_worker,
        d_input_emb_for_worker,
        grad_params_for_worker,
        output_z_global_fp32_host,
        d_input_z_global_fp32_host,
        d_input_emb_global_fp32_host,
        grad_params_fp32_global_host,
    )


# ---------------------------------------------------------------------------
# MSAModule1D activation checkpointing parity test
# ---------------------------------------------------------------------------


def parallel_assert_msa_module_1d_activation_checkpointing(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    msa_module_params,
    min_val_init,
    max_val_init,
    input_z_global_host,
    input_emb_global_host,
    input_feats_global_host,
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

    seed_by_rank(0, seed=42)

    # First module: WITHOUT activation checkpointing
    msa_module_params = dict(msa_module_params)
    msa_module_params["activation_checkpointing"] = False
    module_serial = SerialMSAModule(**msa_module_params)
    module_serial = module_serial.to(dtype=dtype, device=manager.device)
    init_module_params_uniform(module_serial, low=min_val_init, high=max_val_init)
    set_dtype_specific_inf_values(module_serial, dtype)

    module_state_dict_ref = module_serial.state_dict()

    module = MSAModule1D(module_serial, manager)
    module.train()

    placements_z = PAIR_PLACEMENTS_1D
    placements_emb = PAIR_PLACEMENTS_1D
    placements_msa = MSA_PLACEMENTS_1D
    placements_token_mask = PAIR_PLACEMENTS_1D

    input_z_dtensor = distribute_tensor(
        input_z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_z,
    ).requires_grad_(True)

    input_emb_dtensor = distribute_tensor(
        input_emb_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_emb,
    ).requires_grad_(True)

    input_feats_dtensor = {}
    for key, value in input_feats_global_host.items():
        if key in ["msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"]:
            input_feats_dtensor[key] = distribute_tensor(
                value.to(dtype=dtype if value.dtype.is_floating_point else value.dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_msa,
            )
        elif key == "token_pair_pad_mask":
            input_feats_dtensor[key] = distribute_tensor(
                value.to(dtype=dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_token_mask,
            )

    input_z_dtensor_copy = input_z_dtensor.detach().clone().requires_grad_(True)
    input_emb_dtensor_copy = input_emb_dtensor.detach().clone().requires_grad_(True)
    input_feats_dtensor_copy = {k: v.detach().clone() for k, v in input_feats_dtensor.items()}

    # Save RNG state so the second forward pass (with activation checkpointing)
    # sees the same dropout masks as the first forward pass.
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device=manager.device) if device_type == "cuda" else None

    output_z_dtensor_result = module(input_z_dtensor, input_emb_dtensor, input_feats_dtensor)

    d_output_z_dtensor = torch.distributed.tensor.rand(
        output_z_dtensor_result.shape,
        requires_grad=False,
        dtype=dtype,
        device_mesh=manager.device_mesh,
        placements=output_z_dtensor_result.placements,
    )
    d_output_z_dtensor_copy = d_output_z_dtensor.detach().clone()

    torch.autograd.backward([output_z_dtensor_result], [d_output_z_dtensor])

    # Create second module with activation checkpointing enabled
    msa_module_params["activation_checkpointing"] = True
    module_serial_act_ckpt = SerialMSAModule(**msa_module_params)
    module_serial_act_ckpt.load_state_dict(module_state_dict_ref)
    set_dtype_specific_inf_values(module_serial_act_ckpt, dtype)
    module_serial_act_ckpt = module_serial_act_ckpt.to(dtype=dtype, device=manager.device)
    module_act_ckpt = MSAModule1D(module_serial_act_ckpt, manager)
    module_act_ckpt.train()

    # Restore RNG state so dropout masks match the first forward pass
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device=manager.device)

    output_z_dtensor_result_act_ckpt = module_act_ckpt(
        input_z_dtensor_copy, input_emb_dtensor_copy, input_feats_dtensor_copy
    )

    assert_tensors_identical(
        output_z_dtensor_result_act_ckpt.to_local(),
        output_z_dtensor_result.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )

    torch.autograd.backward([output_z_dtensor_result_act_ckpt], [d_output_z_dtensor_copy])

    assert_tensors_identical(input_z_dtensor.grad.to_local(), input_z_dtensor_copy.grad.to_local())
    assert_tensors_identical(input_emb_dtensor.grad.to_local(), input_emb_dtensor_copy.grad.to_local())

    result_param_grads_dict = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            result_param_grads_dict[name] = param.grad

    for name, param_act_ckpt_grad in module_act_ckpt.named_parameters():
        assert name in result_param_grads_dict, f"Parameter {name}'s gradient is not found in result gradients"
        result_grad = result_param_grads_dict[name]
        assert_tensors_identical(result_grad.to_local(), param_act_ckpt_grad.grad.to_local())

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype",
    (
        params_test_act_ckpt := [
            (((1, 2), True, "cuda", "ENV"), torch.float32),
            (((2, 2), True, "cuda", "ENV"), torch.float32),
        ]
    ),
    indirect=["setup_env"],
    ids=[f"dp:{x[0][0][0]}, cp:{x[0][0][1]}, device_type:{x[0][2]}, dtype:{x[1]}" for x in params_test_act_ckpt],
)
def test_msa_module_1d_activation_checkpointing(setup_env, dtype):
    """MSAModule1D with activation checkpointing vs without; results should match."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    msa_module_params = create_msa_module_init_params_v2(use_large_model=False)
    msa_module_params["msa_dropout"] = 0.5
    msa_module_params["z_dropout"] = 0.5

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 4
    S = cp_size * 6
    min_val_init = -1
    max_val_init = 1
    dim_input_msa = const.num_tokens

    input_z_global = torch.empty((B, N, N, msa_module_params["token_z"]), dtype=dtype, requires_grad=True, device="cpu")
    input_emb_global = torch.empty((B, N, msa_module_params["token_s"]), dtype=dtype, requires_grad=True, device="cpu")

    input_feats_global_host = {
        "msa": torch.randint(0, dim_input_msa, (B, S, N), dtype=torch.int64, device="cpu"),
        "has_deletion": torch.empty((B, S, N), dtype=dtype, device="cpu"),
        "deletion_value": torch.empty((B, S, N), dtype=dtype, device="cpu"),
        "msa_paired": torch.randint(0, 2, (B, S, N), dtype=dtype, device="cpu"),
        "msa_mask": torch.ones((B, S, N), dtype=dtype, device="cpu"),
        "token_pad_mask": torch.randint(0, 2, (B, N), dtype=dtype, device="cpu"),
    }
    input_feats_global_host["token_pad_mask"][0, N // cp_size :] = 0
    input_feats_global_host["token_pair_pad_mask"] = (
        input_feats_global_host["token_pad_mask"][:, :, None] * input_feats_global_host["token_pad_mask"][:, None, :]
    )
    input_feats_global_host["msa_mask"][0, (S // cp_size) :, :] = 0
    input_feats_global_host["msa_mask"][0, :, (N // cp_size) :] = 0

    init_tensors_uniform([input_z_global, input_emb_global], low=min_val_init, high=max_val_init)
    for key, tensor in input_feats_global_host.items():
        if tensor.dtype.is_floating_point and "mask" not in key:
            init_tensors_uniform([tensor], low=min_val_init, high=max_val_init)

    input_feats_for_distributed = _feats_for_distributed_1d(input_feats_global_host, dtype, device="cpu")

    spawn_multiprocessing(
        parallel_assert_msa_module_1d_activation_checkpointing,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        msa_module_params,
        min_val_init,
        max_val_init,
        input_z_global.detach().to(dtype=dtype, device="cpu", copy=True),
        input_emb_global.detach().to(dtype=dtype, device="cpu", copy=True),
        input_feats_for_distributed,
    )


# ---------------------------------------------------------------------------
# PairWeightedAveraging1DTiled isolation tests
#
# fp64-parity at cp=2 CUDA + cp=3 CPU (subsumes the prototype sibling test
# previously at test_dtensor_msa_1d_tiled.py).  cp=3 is a regression guard
# against baseline ``v_buffer = [v_mh, empty_like(v_mh)]`` aliasing which
# corrupts the saved v_mh after cp_size − 1 swaps; invisible at cp=2.
#
# bf16-stability at cp=2 CUDA — verifies the distributed tiled-softmax
# bf16 forward+backward produces error percentiles no worse than the
# serial fp32 baseline at a production-realistic head shape (H=8, d=32).
# This is the same accuracy-degradation invariant used by the 2D-CP PWA
# tests via :func:`assert_no_percentile_upshift`; see derivation block on
# the bf16 test for the first-principles justification.
# ---------------------------------------------------------------------------


def parallel_assert_pwa_1d_tiled_fp64(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    c_m,
    c_z,
    c_h,
    num_heads,
    layer_state_dict,
    input_m_global_host,
    input_z_global_host,
    token_mask_global_host,
    output_expected_global_host,
    d_output_global_host,
    d_input_m_expected_global_host,
    d_input_z_expected_global_host,
    expected_param_grads_global_host_dict,
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

    serial_module = SerialPairWeightedAveraging(c_m, c_z, c_h, num_heads).to(dtype=dtype, device=manager.device)
    serial_module.load_state_dict(layer_state_dict)
    set_dtype_specific_inf_values(serial_module, dtype)

    module = PairWeightedAveraging1DTiled(serial_module, manager.device_mesh, cp_group)
    module.train()

    input_m_global = input_m_global_host.to(dtype=dtype, device=manager.device)
    input_z_global = input_z_global_host.to(dtype=dtype, device=manager.device)
    token_mask_global = token_mask_global_host.to(dtype=dtype, device=manager.device)

    input_m = distribute_tensor(
        input_m_global,
        device_mesh=manager.device_mesh,
        placements=MSA_PLACEMENTS_1D,
    ).requires_grad_(True)
    input_z = distribute_tensor(
        input_z_global,
        device_mesh=manager.device_mesh,
        placements=PAIR_PLACEMENTS_1D,
    ).requires_grad_(True)
    token_mask = distribute_tensor(
        token_mask_global,
        device_mesh=manager.device_mesh,
        placements=PAIR_PLACEMENTS_1D,
    )

    # Sharding-active sanity check (vacuous-pass guard).
    cp_size = grid_group_sizes["cp"]
    if cp_size > 1:
        assert input_m.to_local().shape[2] < input_m_global.shape[2], "MSA not sharded on N"
        assert input_z.to_local().shape[1] < input_z_global.shape[1], "Pair not sharded on dim 1"

    output = module(input_m, input_z, token_mask)
    output_global = output.full_tensor().cpu()
    torch.testing.assert_close(output_global, output_expected_global_host.to(dtype=output_global.dtype))

    d_output_global = d_output_global_host.to(dtype=dtype, device=manager.device)
    d_output = distribute_tensor(
        d_output_global,
        device_mesh=manager.device_mesh,
        placements=MSA_PLACEMENTS_1D,
    )

    torch.autograd.backward([output], [d_output])

    d_input_m_global = input_m.grad.full_tensor().cpu()
    d_input_z_global = input_z.grad.full_tensor().cpu()
    torch.testing.assert_close(d_input_m_global, d_input_m_expected_global_host.to(dtype=d_input_m_global.dtype))
    torch.testing.assert_close(d_input_z_global, d_input_z_expected_global_host.to(dtype=d_input_z_global.dtype))

    # Vacuous-pass guards.
    assert input_m.grad.to_local().abs().max() > 0, "input_m grad is all zeros"
    assert input_z.grad.to_local().abs().max() > 0, "input_z grad is all zeros"

    # Parameter gradient parity.
    for name, expected_grad in expected_param_grads_global_host_dict.items():
        param = dict(module.named_parameters())[name]
        assert param.grad is not None, f"Parameter {name} has no gradient"
        result_grad_global = param.grad.full_tensor().cpu()
        torch.testing.assert_close(result_grad_global, expected_grad.to(dtype=result_grad_global.dtype))

    # Replicated-param grads are identical across CP ranks.
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert_all_identical(param.grad.full_tensor(), cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype",
    [
        (((1, 2), True, "cuda", "ENV"), torch.float64),
        (((1, 3), True, "cpu", "ENV"), torch.float64),
    ],
    indirect=["setup_env"],
)
def test_pwa_1d_tiled_fp64_parity(setup_env, dtype):
    """fp64 parity between tiled-softmax 1D PWA and serial PWA.

    cp=2 CUDA + cp=3 CPU.  cp=3 is a permanent regression guard against the
    baseline ``v_buffer = [v_mh, empty_like(v_mh)]`` aliasing that corrupts
    the saved v_mh after cp_size − 1 swaps (invisible at cp=2, fires at
    cp≥3 — see :class:`_PairWeightedAveraging1DTiledImpl.forward` for the
    explicit detach+clone that defuses it).
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 4
    S = cp_size * 3
    c_m = 8
    c_z = 6
    c_h = 4
    num_heads = 2

    gen = torch.Generator(device="cpu").manual_seed(7)

    input_m_global = torch.empty(B, S, N, c_m, dtype=torch.float64, requires_grad=True)
    input_z_global = torch.empty(B, N, N, c_z, dtype=torch.float64, requires_grad=True)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed())
        init_tensors_uniform([input_m_global, input_z_global], low=-0.5, high=0.5)

    token_mask_global = torch.ones(B, N, N, dtype=torch.float64)

    ref_module = SerialPairWeightedAveraging(c_m, c_z, c_h, num_heads).to(dtype=torch.float64).train()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed() + 1)
        init_module_params_uniform(ref_module, low=-0.5, high=0.5)
    set_dtype_specific_inf_values(ref_module, torch.float64)

    state_dict = ref_module.state_dict()

    output = ref_module(input_m_global, input_z_global, token_mask_global)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed() + 2)
        d_output_global = torch.rand_like(output)
    torch.autograd.backward([output], [d_output_global])

    grad_params = {name: p.grad.detach().cpu() for name, p in ref_module.named_parameters()}

    # fp64: torch.testing.assert_close defaults (rtol=1.3e-6, atol=1e-7) are
    # many orders of magnitude looser than the actual error budget
    # (~N * eps_fp64 ~ N * 2e-16).  Defaults are sufficient.
    spawn_multiprocessing(
        parallel_assert_pwa_1d_tiled_fp64,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        c_m,
        c_z,
        c_h,
        num_heads,
        state_dict,
        input_m_global.detach().cpu(),
        input_z_global.detach().cpu(),
        token_mask_global.detach().cpu(),
        output.detach().cpu(),
        d_output_global.detach().cpu(),
        input_m_global.grad.detach().cpu(),
        input_z_global.grad.detach().cpu(),
        grad_params,
    )


def parallel_assert_pwa_1d_tiled_bf16(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    c_m,
    c_z,
    c_h,
    num_heads,
    layer_state_dict_fp64,
    input_m_global_fp64_host,
    input_z_global_fp64_host,
    token_mask_global_fp64_host,
    d_output_global_fp64_host,
    output_expected_fp64_host,
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
    dtype = torch.bfloat16

    # Distributed module materialised in bf16 from the fp64 state-dict.
    # Convert serial module to bf16 then load to avoid the fp32→bf16
    # parameter round-trip that load_state_dict would otherwise apply.
    serial_module = SerialPairWeightedAveraging(c_m, c_z, c_h, num_heads).to(dtype=dtype, device=manager.device)
    bf16_state = {k: v.to(dtype=dtype) for k, v in layer_state_dict_fp64.items()}
    serial_module.load_state_dict(bf16_state)
    set_dtype_specific_inf_values(serial_module, dtype)

    module = PairWeightedAveraging1DTiled(serial_module, manager.device_mesh, cp_group)
    module.train()

    input_m_global = input_m_global_fp64_host.to(dtype=dtype, device=manager.device)
    input_z_global = input_z_global_fp64_host.to(dtype=dtype, device=manager.device)
    token_mask_global = token_mask_global_fp64_host.to(dtype=dtype, device=manager.device)

    input_m = distribute_tensor(
        input_m_global, device_mesh=manager.device_mesh, placements=MSA_PLACEMENTS_1D
    ).requires_grad_(True)
    input_z = distribute_tensor(
        input_z_global, device_mesh=manager.device_mesh, placements=PAIR_PLACEMENTS_1D
    ).requires_grad_(True)
    token_mask = distribute_tensor(token_mask_global, device_mesh=manager.device_mesh, placements=PAIR_PLACEMENTS_1D)

    cp_size = grid_group_sizes["cp"]
    if cp_size > 1:
        assert input_m.to_local().shape[2] < input_m_global.shape[2], "MSA not sharded on N"
        assert input_z.to_local().shape[1] < input_z_global.shape[1], "Pair not sharded on dim 1"

    output = module(input_m, input_z, token_mask)
    output_global = output.full_tensor().cpu()  # bf16
    d_output_global = d_output_global_fp64_host.to(dtype=dtype, device=manager.device)
    d_output = distribute_tensor(d_output_global, device_mesh=manager.device_mesh, placements=MSA_PLACEMENTS_1D)
    torch.autograd.backward([output], [d_output])

    # Stability gates (NaN/Inf — bf16 path must not blow up).
    assert torch.isfinite(output_global).all(), "Forward output has non-finite values (NaN/Inf)"
    assert torch.isfinite(input_m.grad.to_local()).all(), "input_m grad has non-finite values"
    assert torch.isfinite(input_z.grad.to_local()).all(), "input_z grad has non-finite values"

    # Vacuous-pass guards.
    assert input_m.grad.to_local().abs().max() > 0, "input_m grad is all zeros"
    assert input_z.grad.to_local().abs().max() > 0, "input_z grad is all zeros"

    # Forward-output bf16-vs-fp64 first-principles tolerance.
    #
    # bf16 has 7 mantissa bits → unit roundoff u_bf16 = 2⁻⁸ ≈ 3.9e-3 and
    # machine epsilon eps_bf16 = 2⁻⁷ ≈ 7.8e-3.  The forward consists of
    # ~8 sequential bf16 ops (3 input projections; 1 bias+mask add;
    # 1 logsumexp+softmax per ring step; 1 einsum o_block per ring step;
    # 3 online-merge ops per ring step × cp_size chunks; sigmoid·multiply;
    # proj_o).  Each contributes ≤ u_bf16 relative error; RMS chain error
    # grows as √num_ops · u_bf16 ≈ √15 · 2⁻⁸ ≈ 4 · eps_bf16, with a 2×
    # safety factor for correlated rounding in the final matmul (proj_o).
    #
    #   rtol = K · eps_bf16          with K = 8
    #   atol = K · eps_bf16 · max|x|     (floor for elements near zero;
    #                                    max|x| ≤ 2.0 in this synthetic
    #                                    fixture — single-stage matmul of
    #                                    ~|N(0,0.5)| inputs with ~|N(0,0.5)|
    #                                    weights over 256 channels gives
    #                                    O(1) outputs)
    #
    # Bounds: rtol = 0.0625, atol = 0.125.  Forward output is the
    # primary numerical contract of this test; backward parameter-grad
    # comparison would require a percentile-upshift framework with
    # bf16-specific calibration outside the scope of a single MR (see
    # 2D-CP PWA test for the fp32 analog).  Backward correctness is
    # already proven by the cp=2 + cp=3 fp64 parity tests in this same
    # file, so the bf16 backward needs only stability (no NaN/Inf)
    # which is asserted above.
    bf16_eps = 2.0**-7
    K = 8.0
    max_abs_x = 2.0
    output_atol = K * bf16_eps * max_abs_x
    output_rtol = K * bf16_eps
    torch.testing.assert_close(
        output_global.to(dtype=torch.float32),
        output_expected_fp64_host.to(dtype=torch.float32),
        atol=output_atol,
        rtol=output_rtol,
    )

    # Replicated-param grads are identical across CP ranks (parity-of-state
    # invariant: distributed bf16 must not introduce CP-rank divergence).
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert_all_identical(param.grad.full_tensor(), cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype",
    [
        (((1, 2), True, "cuda", "ENV"), torch.bfloat16),
    ],
    indirect=["setup_env"],
)
def test_pwa_1d_tiled_bf16_stability(setup_env, dtype):
    """bf16 stability of the tiled-softmax 1D PWA at a production-class head shape.

    The tiled-softmax forward keeps the o/v ring buffers in input dtype
    (bf16 under autocast), eliminating the three fp32 buffers that
    dominated the pre-tiled forward peak (~470 MiB each at the 7fct cp=2
    target shape).  This test verifies bf16 stability at H=8, head_dim=32
    — the production head shape — with a representative N_local matching
    the cp=2 regime.

    What this test asserts:

    1. **No NaN/Inf** in forward output or input gradients (the bf16 path
       must not blow up — softmax over -inf masked positions and ring
       reductions are the highest-risk sites for numerical overflow).
    2. **Non-zero gradients** (vacuous-pass guard).
    3. **Forward output bf16-vs-fp64 within first-principles tolerance**
       (atol/rtol derived in the parallel worker; see comment block
       on the `output_atol`/`output_rtol` computation there).
    4. **CP-rank parity of replicated parameter grads** —
       `full_tensor()` reductions must produce identical values across
       CP ranks (no CP-rank divergence introduced by the distributed
       impl).

    What this test does NOT assert: bf16-vs-fp64 accuracy of backward
    parameter gradients.  Per CLAUDE.md L186-188, deriving a
    first-principles bf16 atol bound on parameter gradients requires
    per-parameter accumulation-chain analysis (e.g. LayerNorm.bias.grad
    has a 16384-element accumulation chain; proj_z.weight.grad accumulates
    over both i and j axes), and no single atol/rtol bounds them
    uniformly.  Backward CORRECTNESS is already proven by the fp64
    parity tests in this same file (cp=2 cuda + cp=3 cpu), which use
    the canonical assert_close defaults.  This bf16 cell only needs to
    verify the bf16 forward is numerically stable AND CP-parity holds.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = grid_group_sizes["cp"]
    B = 1
    # Production head shape (H=8, d=32).  Keep B=1, S=64, N_local=64 to
    # bound test runtime on a 2-GPU dev box; the bf16 accumulation error
    # distribution is well-resolved at these S, N_local values because
    # both saturate the bf16 mantissa within the first ~30 sums.
    N_local = 64
    N = cp_size * N_local
    S = 64
    num_heads = 8
    head_dim = 32
    c_h = head_dim
    c_m = num_heads * head_dim  # 256
    c_z = num_heads * head_dim  # 256

    gen = torch.Generator(device="cpu").manual_seed(11)

    # Initialise everything in fp64.  Serial fp64 forward is the
    # high-precision ground truth for the bf16-vs-fp64 output comparison.
    input_m_fp64 = torch.empty(B, S, N, c_m, dtype=torch.float64)
    input_z_fp64 = torch.empty(B, N, N, c_z, dtype=torch.float64)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed())
        init_tensors_uniform([input_m_fp64, input_z_fp64], low=-0.5, high=0.5)

    token_mask_fp64 = torch.ones(B, N, N, dtype=torch.float64)

    ref_module_fp64 = SerialPairWeightedAveraging(c_m, c_z, c_h, num_heads).to(dtype=torch.float64).train()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed() + 1)
        init_module_params_uniform(ref_module_fp64, low=-0.5, high=0.5)
    set_dtype_specific_inf_values(ref_module_fp64, torch.float64)

    state_dict_fp64 = ref_module_fp64.state_dict()

    # fp64 forward → output_expected.  Random grad_output (explicit, never
    # `.sum().backward()` per CLAUDE.md L198-199) is reserved for the
    # distributed bf16 path inside the worker.
    output_fp64 = ref_module_fp64(input_m_fp64, input_z_fp64, token_mask_fp64)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.random.manual_seed(gen.initial_seed() + 2)
        d_output_fp64 = torch.rand_like(output_fp64)

    spawn_multiprocessing(
        parallel_assert_pwa_1d_tiled_bf16,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        c_m,
        c_z,
        c_h,
        num_heads,
        state_dict_fp64,
        input_m_fp64.detach().cpu(),
        input_z_fp64.detach().cpu(),
        token_mask_fp64.detach().cpu(),
        d_output_fp64.detach().cpu(),
        output_fp64.detach().cpu(),
    )
