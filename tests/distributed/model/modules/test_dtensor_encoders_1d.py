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

"""Tests for 1D CP encoder wrappers (FourierEmbedding1D, SingleConditioning1D, PairwiseConditioning1D).

Verifies forward and backward parity against the serial Boltz-2 encoder modules
on a 2D mesh ``(dp, cp)`` with 2-element placements.

Placements under test:

- Single ``s``: ``(Shard(0), Shard(1))``
- Pair ``z``: ``(Shard(0), Shard(1))`` (row-slab)
- Times: ``(Shard(0), Replicate())``

These are thin wrappers — the tests focus on placement correctness, delegation
to serial logic, and placement validation rejection.
"""

import pytest
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.modules.encoders_1d import (
    FourierEmbedding1D,
    PairwiseConditioning1D,
    RelativePositionEncoder1D,
    SingleConditioning1D,
)
from boltz.model.modules.encodersv2 import (
    FourierEmbedding as SerialFourierEmbedding,
)
from boltz.model.modules.encodersv2 import (
    PairwiseConditioning as SerialPairwiseConditioning,
)
from boltz.model.modules.encodersv2 import (
    RelativePositionEncoder as SerialRelativePositionEncoder,
)
from boltz.model.modules.encodersv2 import (
    SingleConditioning as SerialSingleConditioning,
)
from boltz.testing.utils import (
    assert_all_identical,
    assert_tensors_identical,
    init_module_params_uniform,
    init_tensors_uniform,
    make_divergent_cyclic_rel_pos_feats,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)

_INIT_LOW, _INIT_HIGH = -0.08, 0.08

# Shared parametrize values for all encoder 1D tests.
_ENCODER_1D_PARAMS = [
    ((1, 2), True, "cuda", "ENV"),
    ((1, 3), True, "cuda", "ENV"),
    ((1, 2), True, "cpu", "ENV"),
    ((1, 3), True, "cpu", "ENV"),
]

# ---------------------------------------------------------------------------
# FourierEmbedding1D
# ---------------------------------------------------------------------------


def _worker_fourier_embedding_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dim,
    state_dict,
    times_global_host,
    output_expected_global_host,
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

    # Create serial module from state dict
    serial = SerialFourierEmbedding(dim)
    serial = serial.to(device=manager.device)
    serial.load_state_dict(state_dict)

    # Create 1D CP module
    module = FourierEmbedding1D(serial, manager.device_mesh)

    placements_times = (Shard(0), Replicate())

    times_dt = distribute_tensor(
        times_global_host.to(device=manager.device),
        manager.device_mesh,
        placements_times,
    ).requires_grad_(False)

    times_dt_copy = times_dt.detach().clone()

    # Forward
    output_dt = module(times_dt)

    # Input immutability
    assert_tensors_identical(times_dt_copy.to_local(), times_dt.to_local(), check_grad=False, check_grad_fn=False)

    # Forward parity
    torch.testing.assert_close(output_dt.full_tensor().cpu(), output_expected_global_host)

    # Verify output placements
    assert output_dt.placements == (Shard(0), Replicate()), f"Bad output placements: {output_dt.placements}"

    # CP ranks produce identical output (frozen params, replicated computation on cp axis)
    assert_all_identical(output_dt.to_local().detach(), manager.group["cp"])

    # All parameters are frozen
    for name, param in module.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"

    # Placement validation: wrong placements should be rejected
    bad_times = distribute_tensor(
        times_global_host.to(device=manager.device),
        manager.device_mesh,
        (Replicate(), Replicate()),
    )
    with pytest.raises(ValueError, match="incorrect placements"):
        module(bad_times)

    DistributedManager.cleanup()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# SingleConditioning1D
# ---------------------------------------------------------------------------


def _worker_single_conditioning_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    state_dict,
    module_kwargs,
    times_global_host,
    s_trunk_global_host,
    s_inputs_global_host,
    s_expected_global_host,
    normed_fourier_expected_global_host,
    d_s_global_host,
    d_normed_fourier_global_host,
    d_s_trunk_expected_global_host,
    d_s_inputs_expected_global_host,
    expected_param_grads_host,
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

    # Recreate serial module from state dict
    serial = SerialSingleConditioning(**module_kwargs)
    serial = serial.to(dtype=dtype, device=manager.device)
    serial.load_state_dict(state_dict)
    serial.train()

    # Create 1D CP module
    module = SingleConditioning1D(serial, manager.device_mesh).train()

    placements_times = (Shard(0), Replicate())
    placements_s = (Shard(0), Shard(1))

    times_dt = distribute_tensor(
        times_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_times,
    ).requires_grad_(False)
    s_trunk_dt = distribute_tensor(
        s_trunk_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_s,
    ).requires_grad_(True)
    s_inputs_dt = distribute_tensor(
        s_inputs_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_s,
    ).requires_grad_(True)

    # Clone for immutability check
    times_dt_copy = times_dt.detach().clone()
    s_trunk_dt_copy = s_trunk_dt.detach().clone().requires_grad_(s_trunk_dt.requires_grad)
    s_inputs_dt_copy = s_inputs_dt.detach().clone().requires_grad_(s_inputs_dt.requires_grad)

    # Forward
    s_dt, normed_fourier_dt = module(times_dt, s_trunk_dt, s_inputs_dt)

    # Input immutability
    assert_tensors_identical(times_dt_copy.to_local(), times_dt.to_local(), check_grad=False, check_grad_fn=False)
    assert_tensors_identical(s_trunk_dt_copy.to_local(), s_trunk_dt.to_local(), check_grad=False, check_grad_fn=False)
    assert_tensors_identical(s_inputs_dt_copy.to_local(), s_inputs_dt.to_local(), check_grad=False, check_grad_fn=False)

    # Forward parity
    torch.testing.assert_close(s_dt.full_tensor().cpu(), s_expected_global_host)

    # Verify output placements
    assert s_dt.placements == placements_s, f"Bad s placements: {s_dt.placements}"

    # Sharding is active
    assert s_dt.to_local().shape[1] < s_dt.shape[1], "Sharding not active on s dim 1"

    # Vacuousness: output must differ from zero
    assert s_dt.to_local().abs().max() > 0, "s output is all zeros"

    if normed_fourier_expected_global_host is not None:
        assert normed_fourier_dt is not None
        torch.testing.assert_close(normed_fourier_dt.full_tensor().cpu(), normed_fourier_expected_global_host)
        assert normed_fourier_dt.placements == placements_times
    else:
        assert normed_fourier_dt is None

    # Backward
    outputs = [s_dt]
    grad_outputs = [
        distribute_tensor(
            d_s_global_host.to(device=manager.device, dtype=dtype),
            manager.device_mesh,
            placements_s,
        )
    ]
    if normed_fourier_dt is not None and d_normed_fourier_global_host is not None:
        outputs.append(normed_fourier_dt)
        grad_outputs.append(
            distribute_tensor(
                d_normed_fourier_global_host.to(device=manager.device, dtype=dtype),
                manager.device_mesh,
                placements_times,
            )
        )
    torch.autograd.backward(outputs, grad_outputs)

    # Input gradient parity
    torch.testing.assert_close(s_trunk_dt.grad.full_tensor().cpu(), d_s_trunk_expected_global_host)
    torch.testing.assert_close(s_inputs_dt.grad.full_tensor().cpu(), d_s_inputs_expected_global_host)

    # Non-zero gradients
    assert s_trunk_dt.grad.to_local().abs().max() > 0, "s_trunk grad is all zeros"
    assert s_inputs_dt.grad.to_local().abs().max() > 0, "s_inputs grad is all zeros"

    # Parameter gradient parity (skip frozen FourierEmbedding.proj)
    for name, param in module.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, f"Frozen parameter {name} should have no gradient"
            continue
        assert param.grad is not None, f"Parameter {name} has no gradient"
        expected_grad = expected_param_grads_host[name]
        torch.testing.assert_close(
            param.grad.full_tensor().cpu(),
            expected_grad,
            msg=lambda m, n=name: f"Parameter gradient mismatch for {n}: {m}",
        )
        # Replicated param grads identical across CP ranks
        assert_all_identical(param.grad.full_tensor().detach(), manager.group["cp"])

    # Placement validation: wrong placements for s_trunk should be rejected
    bad_s_trunk = distribute_tensor(
        s_trunk_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        (Replicate(), Replicate()),
    )
    with pytest.raises(ValueError, match="s_trunk placements"):
        module(times_dt, bad_s_trunk, s_inputs_dt)

    DistributedManager.cleanup()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# PairwiseConditioning1D
# ---------------------------------------------------------------------------


def _worker_pairwise_conditioning_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    state_dict,
    module_kwargs,
    z_trunk_global_host,
    token_rel_pos_feats_global_host,
    z_expected_global_host,
    d_z_global_host,
    d_z_trunk_expected_global_host,
    d_token_rel_pos_feats_expected_global_host,
    expected_param_grads_host,
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

    # Recreate serial module from state dict
    serial = SerialPairwiseConditioning(**module_kwargs)
    serial = serial.to(dtype=dtype, device=manager.device)
    serial.load_state_dict(state_dict)
    serial.train()

    # Create 1D CP module
    module = PairwiseConditioning1D(serial, manager.device_mesh).train()

    placements_pair = (Shard(0), Shard(1))

    z_trunk_dt = distribute_tensor(
        z_trunk_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_pair,
    ).requires_grad_(True)
    token_rel_pos_feats_dt = distribute_tensor(
        token_rel_pos_feats_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_pair,
    ).requires_grad_(True)

    # Clone for immutability check
    z_trunk_dt_copy = z_trunk_dt.detach().clone().requires_grad_(z_trunk_dt.requires_grad)
    token_rel_pos_feats_dt_copy = (
        token_rel_pos_feats_dt.detach().clone().requires_grad_(token_rel_pos_feats_dt.requires_grad)
    )

    # Forward
    z_dt = module(z_trunk_dt, token_rel_pos_feats_dt)

    # Input immutability
    assert_tensors_identical(z_trunk_dt_copy.to_local(), z_trunk_dt.to_local(), check_grad=False, check_grad_fn=False)
    assert_tensors_identical(
        token_rel_pos_feats_dt_copy.to_local(),
        token_rel_pos_feats_dt.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )

    # Forward parity
    torch.testing.assert_close(z_dt.full_tensor().cpu(), z_expected_global_host)

    # Verify output placements
    assert z_dt.placements == placements_pair, f"Bad z placements: {z_dt.placements}"

    # Sharding is active (row-slab: dim 1 is sharded)
    assert z_dt.to_local().shape[1] < z_dt.shape[1], "Sharding not active on z dim 1"

    # Vacuousness: output must differ from zero
    assert z_dt.to_local().abs().max() > 0, "z output is all zeros"

    # Backward
    d_z_dt = distribute_tensor(
        d_z_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_pair,
    )
    z_dt.backward(d_z_dt)

    # Input gradient parity
    torch.testing.assert_close(z_trunk_dt.grad.full_tensor().cpu(), d_z_trunk_expected_global_host)
    torch.testing.assert_close(
        token_rel_pos_feats_dt.grad.full_tensor().cpu(), d_token_rel_pos_feats_expected_global_host
    )

    # Non-zero gradients
    assert z_trunk_dt.grad.to_local().abs().max() > 0, "z_trunk grad is all zeros"
    assert token_rel_pos_feats_dt.grad.to_local().abs().max() > 0, "token_rel_pos_feats grad is all zeros"

    # Parameter gradient parity
    for name, param in module.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, f"Frozen param {name} should have no grad"
            continue
        assert param.grad is not None, f"Parameter {name} has no gradient"
        expected_grad = expected_param_grads_host[name]
        torch.testing.assert_close(
            param.grad.full_tensor().cpu(),
            expected_grad,
            msg=lambda m, n=name: f"Parameter gradient mismatch for {n}: {m}",
        )
        # Replicated param grads identical across CP ranks
        assert isinstance(param.grad, DTensor), f"Param grad '{name}' should be DTensor, got {type(param.grad)}"
        assert_all_identical(param.grad.full_tensor().detach(), manager.group["cp"])

    # Placement validation: wrong placements for z_trunk should be rejected
    bad_z_trunk = distribute_tensor(
        z_trunk_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        (Replicate(), Replicate()),
    )
    with pytest.raises(ValueError, match="z_trunk placements"):
        module(bad_z_trunk, token_rel_pos_feats_dt)

    DistributedManager.cleanup()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# Test: FourierEmbedding1D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    _ENCODER_1D_PARAMS,
    indirect=["setup_env"],
    ids=[f"dp:{x[0][0]}, cp:{x[0][1]}, device:{x[2]}" for x in _ENCODER_1D_PARAMS],
)
def test_fourier_embedding_1d(setup_env):
    """FourierEmbedding1D: forward parity vs serial, placement validation, frozen params."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type=device_type, world_size=world_size)

    B = 4 * grid_group_sizes["dp"]
    dim = 32

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        # Create serial module
        serial = SerialFourierEmbedding(dim)
        serial = serial.to(device=device_type)
        state_dict = serial.state_dict()

        # Create input on CPU then move to device (avoids generator device mismatch)
        times_global = torch.rand(B).to(device=device_type)

        # Serial forward
        output_expected = serial(times_global)

    spawn_multiprocessing(
        _worker_fourier_embedding_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dim,
        state_dict,
        times_global.detach().cpu(),
        output_expected.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Test: SingleConditioning1D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    _ENCODER_1D_PARAMS,
    indirect=["setup_env"],
    ids=[f"dp:{x[0][0]}, cp:{x[0][1]}, device:{x[2]}" for x in _ENCODER_1D_PARAMS],
)
@pytest.mark.parametrize("disable_times", [False, True], ids=["times:on", "times:off"])
def test_single_conditioning_1d(setup_env, disable_times):
    """SingleConditioning1D: forward/backward parity, placement validation."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type=device_type, world_size=world_size)

    dtype = torch.float64
    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 8
    token_s = 16
    dim_fourier = 32
    num_transitions = 2
    sigma_data = 1.0

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        module_kwargs = {
            "sigma_data": sigma_data,
            "token_s": token_s,
            "dim_fourier": dim_fourier,
            "num_transitions": num_transitions,
            "disable_times": disable_times,
        }

        # Create serial module — cast to dtype BEFORE param init
        serial = SerialSingleConditioning(**module_kwargs)
        serial = serial.to(dtype=dtype).train()
        init_module_params_uniform(serial, low=_INIT_LOW, high=_INIT_HIGH)
        state_dict = serial.state_dict()

        # Create inputs
        times_global = torch.empty(B, dtype=dtype)
        s_trunk_global = torch.empty(B, N, token_s, dtype=dtype, requires_grad=True)
        s_inputs_global = torch.empty(B, N, token_s, dtype=dtype, requires_grad=True)
        init_tensors_uniform([times_global, s_trunk_global, s_inputs_global], low=_INIT_LOW, high=_INIT_HIGH)

        # Serial forward
        s_serial, normed_fourier_serial = serial(times_global, s_trunk_global, s_inputs_global)

        # Create upstream gradients
        d_s = torch.empty_like(s_serial)
        init_tensors_uniform([d_s], low=_INIT_LOW, high=_INIT_HIGH)

        d_normed_fourier = None
        if normed_fourier_serial is not None:
            d_normed_fourier = torch.empty_like(normed_fourier_serial)
            init_tensors_uniform([d_normed_fourier], low=_INIT_LOW, high=_INIT_HIGH)

        # Serial backward
        outputs = [s_serial]
        grad_outputs = [d_s]
        if normed_fourier_serial is not None and d_normed_fourier is not None:
            outputs.append(normed_fourier_serial)
            grad_outputs.append(d_normed_fourier)
        torch.autograd.backward(outputs, grad_outputs)

        # Collect expected results
        s_expected = s_serial.detach().cpu()
        normed_fourier_expected = normed_fourier_serial.detach().cpu() if normed_fourier_serial is not None else None
        d_s_trunk_expected = s_trunk_global.grad.detach().cpu()
        d_s_inputs_expected = s_inputs_global.grad.detach().cpu()

        expected_param_grads = {}
        for name, param in serial.named_parameters():
            if param.requires_grad and param.grad is not None:
                expected_param_grads[name] = param.grad.detach().cpu()

    spawn_multiprocessing(
        _worker_single_conditioning_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        state_dict,
        module_kwargs,
        times_global.detach().cpu(),
        s_trunk_global.detach().cpu(),
        s_inputs_global.detach().cpu(),
        s_expected,
        normed_fourier_expected,
        d_s.detach().cpu(),
        d_normed_fourier.detach().cpu() if d_normed_fourier is not None else None,
        d_s_trunk_expected,
        d_s_inputs_expected,
        expected_param_grads,
    )


# ---------------------------------------------------------------------------
# Test: PairwiseConditioning1D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    _ENCODER_1D_PARAMS,
    indirect=["setup_env"],
    ids=[f"dp:{x[0][0]}, cp:{x[0][1]}, device:{x[2]}" for x in _ENCODER_1D_PARAMS],
)
def test_pairwise_conditioning_1d(setup_env):
    """PairwiseConditioning1D: forward/backward parity, placement validation."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type=device_type, world_size=world_size)

    dtype = torch.float64
    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 4
    token_z = 16
    dim_token_rel_pos_feats = 8
    num_transitions = 2

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        module_kwargs = {
            "token_z": token_z,
            "dim_token_rel_pos_feats": dim_token_rel_pos_feats,
            "num_transitions": num_transitions,
        }

        # Create serial module — cast to dtype BEFORE param init
        serial = SerialPairwiseConditioning(**module_kwargs)
        serial = serial.to(dtype=dtype).train()
        init_module_params_uniform(serial, low=_INIT_LOW, high=_INIT_HIGH)
        state_dict = serial.state_dict()

        # Create inputs
        z_trunk_global = torch.empty(B, N, N, token_z, dtype=dtype, requires_grad=True)
        token_rel_pos_feats_global = torch.empty(B, N, N, dim_token_rel_pos_feats, dtype=dtype, requires_grad=True)
        init_tensors_uniform([z_trunk_global, token_rel_pos_feats_global], low=_INIT_LOW, high=_INIT_HIGH)

        # Serial forward
        z_serial = serial(z_trunk_global, token_rel_pos_feats_global)

        # Create upstream gradient
        d_z = torch.empty_like(z_serial)
        init_tensors_uniform([d_z], low=_INIT_LOW, high=_INIT_HIGH)

        # Serial backward
        z_serial.backward(d_z)

        # Collect expected results
        z_expected = z_serial.detach().cpu()
        d_z_trunk_expected = z_trunk_global.grad.detach().cpu()
        d_token_rel_pos_feats_expected = token_rel_pos_feats_global.grad.detach().cpu()

        expected_param_grads = {}
        for name, param in serial.named_parameters():
            if param.grad is not None:
                expected_param_grads[name] = param.grad.detach().cpu()

    spawn_multiprocessing(
        _worker_pairwise_conditioning_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        state_dict,
        module_kwargs,
        z_trunk_global.detach().cpu(),
        token_rel_pos_feats_global.detach().cpu(),
        z_expected,
        d_z.detach().cpu(),
        d_z_trunk_expected,
        d_token_rel_pos_feats_expected,
        expected_param_grads,
    )


# ---------------------------------------------------------------------------
# RelativePositionEncoder1D
# ---------------------------------------------------------------------------


def _worker_relative_position_encoder_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    state_dict,
    module_kwargs,
    feats_global_host,
    p_expected_global_host,
    d_p_global_host,
    expected_param_grads_host,
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

    # Recreate serial module from state dict (random-initialised weights).
    serial = SerialRelativePositionEncoder(**module_kwargs)
    serial = serial.to(dtype=dtype, device=manager.device)
    serial.load_state_dict(state_dict)
    serial.train()

    # 1D CP module — shares the cp group for the column all-gathers.
    module = RelativePositionEncoder1D(serial, manager.device_mesh, manager.group["cp"]).train()

    placements_single = (Shard(0), Shard(1))
    placements_pair = (Shard(0), Shard(1))

    # Integer structural features are non-differentiable.
    feats_dt = {
        key: distribute_tensor(
            value.to(device=manager.device),
            manager.device_mesh,
            placements_single,
        ).requires_grad_(False)
        for key, value in feats_global_host.items()
    }

    # Sanity: the cyclic layout is actually rank-divergent on this mesh, so the
    # test genuinely exercises the lockstep all_reduce (and would deadlock
    # without it).
    cyclic_local = feats_dt["cyclic_period"].to_local()
    has_cyclic_local = int(torch.any(cyclic_local > 0))
    has_cyclic_all = [None] * grid_group_sizes["cp"]
    torch.distributed.all_gather_object(has_cyclic_all, has_cyclic_local, group=manager.group["cp"])
    assert (
        min(has_cyclic_all) == 0 and max(has_cyclic_all) == 1
    ), f"Test setup is not rank-divergent on cp={grid_group_sizes['cp']}: {has_cyclic_all}"

    # Forward
    p_dt = module(feats_dt)

    # Forward parity vs serial single-device reference.
    torch.testing.assert_close(p_dt.full_tensor().cpu(), p_expected_global_host)

    # Output placements + active sharding (row-slab: dim 1 sharded along cp).
    assert p_dt.placements == placements_pair, f"Bad p placements: {p_dt.placements}"
    assert p_dt.to_local().shape[1] < p_dt.shape[1], "Sharding not active on p dim 1"
    assert p_dt.to_local().abs().max() > 0, "p output is all zeros"

    # Backward with an explicit random grad_output (NOT .sum().backward()).
    d_p_dt = distribute_tensor(
        d_p_global_host.to(device=manager.device, dtype=dtype),
        manager.device_mesh,
        placements_pair,
    )
    p_dt.backward(d_p_dt)

    # Parameter gradient parity (only linear_layer is differentiable).
    saw_grad = False
    for name, param in module.named_parameters():
        assert param.requires_grad, f"Parameter {name} unexpectedly frozen"
        assert param.grad is not None, f"Parameter {name} has no gradient"
        expected_grad = expected_param_grads_host[name]
        torch.testing.assert_close(
            param.grad.full_tensor().cpu(),
            expected_grad,
            msg=lambda m, n=name: f"Parameter gradient mismatch for {n}: {m}",
        )
        assert isinstance(param.grad, DTensor), f"Param grad '{name}' should be DTensor, got {type(param.grad)}"
        # Replicated param grads identical across CP ranks.
        assert_all_identical(param.grad.full_tensor().detach(), manager.group["cp"])
        assert param.grad.full_tensor().abs().max() > 0, f"Param grad {name} is all zeros"
        saw_grad = True
    assert saw_grad, "No differentiable parameters were exercised"

    # Placement validation: a wrong input placement is rejected (mirrors the
    # sibling 1D encoders).  Re-extract feats — the forward consumed the graph.
    bad_feats = {
        key: distribute_tensor(
            value.to(device=manager.device),
            manager.device_mesh,
            placements_single,
        ).requires_grad_(False)
        for key, value in feats_global_host.items()
    }
    bad_feats["asym_id"] = distribute_tensor(
        feats_global_host["asym_id"].to(device=manager.device),
        manager.device_mesh,
        (Replicate(), Replicate()),
    )
    with pytest.raises(ValueError, match="placements"):
        module(bad_feats)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    _ENCODER_1D_PARAMS,
    indirect=["setup_env"],
    ids=[f"dp:{x[0][0]}, cp:{x[0][1]}, device:{x[2]}" for x in _ENCODER_1D_PARAMS],
)
def test_relative_position_encoder_1d(setup_env):
    """RelativePositionEncoder1D: forward/backward parity vs serial under random
    init, with a rank-divergent cyclic layout that exercises the lockstep
    all_reduce (deadlocks without it)."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type=device_type, world_size=world_size)

    dtype = torch.float64
    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 4
    token_z = 16
    r_max = 8
    s_max = 2
    cyclic_period_val = 3

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        module_kwargs = {
            "token_z": token_z,
            "r_max": r_max,
            "s_max": s_max,
            "fix_sym_check": True,
            "cyclic_pos_enc": True,
        }

        # Serial module — random init (cast to dtype BEFORE param init).
        serial = SerialRelativePositionEncoder(**module_kwargs)
        serial = serial.to(dtype=dtype).train()
        init_module_params_uniform(serial, low=_INIT_LOW, high=_INIT_HIGH)
        state_dict = serial.state_dict()

        # Randomize the non-trigger features (contained generator) while keeping
        # the rank-divergent cyclic_period backbone — exercises a realistic
        # feature distribution instead of all-zero/sequential inputs.
        feats_rng = torch.Generator().manual_seed(1234)
        feats_global = make_divergent_cyclic_rel_pos_feats(B, N, cyclic_period_val, rng=feats_rng)

        # Serial forward
        p_serial = serial(feats_global)

        # Non-vacuity for the cyclic MATH (not just the deadlock): the cyclic
        # correction must materially change the output, else the parity test
        # could pass even if the cyclic branch were silently broken.  Re-run
        # serial with cyclic_period zeroed (correction skipped) and confirm the
        # outputs differ — requires cyclic_period_val small vs the residue-index
        # span so round(d / period) != 0 for some same-chain pairs.
        with torch.no_grad():
            feats_no_cyclic = dict(feats_global)
            feats_no_cyclic["cyclic_period"] = torch.zeros_like(feats_global["cyclic_period"])
            p_no_cyclic = serial(feats_no_cyclic)
        assert not torch.allclose(p_serial.detach(), p_no_cyclic), (
            "cyclic correction is a no-op in this setup — pick cyclic_period_val small "
            "vs the residue-index span so round(d / period) != 0 for some pairs"
        )

        # Explicit random upstream gradient (no .sum().backward()).
        d_p = torch.empty_like(p_serial)
        init_tensors_uniform([d_p], low=_INIT_LOW, high=_INIT_HIGH)

        # Serial backward
        p_serial.backward(d_p)

        p_expected = p_serial.detach().cpu()
        expected_param_grads = {
            name: param.grad.detach().cpu() for name, param in serial.named_parameters() if param.grad is not None
        }

    spawn_multiprocessing(
        _worker_relative_position_encoder_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        state_dict,
        module_kwargs,
        feats_global,
        p_expected,
        d_p.detach().cpu(),
        expected_param_grads,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
