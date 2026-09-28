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

"""Checkpoint save/load round-trip test for 1D CP.

Tests that model state_dicts with DTensor parameters on a 1D CP mesh
``(dp, cp)`` survive a round-trip through:

1. ``convert_dtensors_to_tensors`` — saves DTensors as plain tensors
2. ``convert_serial_checkpoint_to_distributed_state_dict`` — loads plain
   tensors back into a DTensor template

This validates the core checkpoint portability guarantee of
:class:`~boltz.distributed.lightning_strategy.BoltzContextParallelStrategy`
without going through the full ``train()`` entrypoint (which is already
tested by ``test_dtensor_boltz2_1d_train.py``).
"""

import os

import pytest
import torch
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from boltz.distributed.lightning_strategy import _redistribute_optimizer_state_to_params
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.modules.utils import (
    convert_dtensors_to_tensors,
    convert_serial_checkpoint_to_distributed_state_dict,
    has_dtensors,
)
from boltz.testing.utils import (
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)


class _TinyModel(torch.nn.Module):
    """Minimal model with a few linear layers for checkpoint round-trip tests."""

    def __init__(self, hidden: int = 8, out: int = 4) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden, hidden, bias=True)
        self.linear2 = torch.nn.Linear(hidden, out, bias=False)


def _wrap_as_dtensor(
    model: _TinyModel,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
) -> None:
    """Replace model parameters with Replicate DTensors on the given mesh.

    This mirrors what the Boltz2_1D model wrapper does: all parameters are
    replicated across the CP dimension so every rank holds the same weights.
    """
    placements = (Replicate(), Replicate())
    for name, param in list(model.named_parameters()):
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        dtensor_param = torch.nn.Parameter(
            distribute_tensor(param.data, device_mesh=device_mesh, placements=placements),
            requires_grad=param.requires_grad,
        )
        setattr(parent, parts[-1], dtensor_param)


def _parallel_checkpoint_round_trip(rank: int, payload: tuple) -> None:
    """Multi-rank worker: test checkpoint save/load round-trip on 1D CP mesh.

    Steps:
    1. Create a tiny model and wrap parameters as DTensors on a 2D mesh (dp, cp).
    2. Save state_dict via convert_dtensors_to_tensors (simulating strategy save).
    3. Assert saved checkpoint contains only plain tensors.
    4. Persist to disk and reload (simulates real file I/O).
    5. Load back via convert_serial_checkpoint_to_distributed_state_dict.
    6. Assert all parameters match after round-trip.
    7. Verify optimizer state redistribution works with synthetic state.
    """
    env_per_rank, device_type, backend, grid_group_sizes, ckpt_dir = payload

    monkeypatch = pytest.MonkeyPatch()
    for key, value in env_per_rank.items():
        monkeypatch.setenv(key, f"{rank}" if value == "<INPUT_RANK>" else value)
    DistributedManager._state = {}

    try:
        DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
        dm = DistributedManager()
        device_mesh = dm.device_mesh

        # -- 1. Create model with DTensor parameters --
        model = _TinyModel(hidden=8, out=4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for p in model.parameters():
                p.data = torch.randn(p.shape)
        model.to(device_type)
        _wrap_as_dtensor(model, device_mesh)

        # Guard against vacuous pass: verify parameters are DTensors
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor), f"Parameter '{name}' should be DTensor, got {type(param)}"

        # Capture pre-save state for comparison (full_tensor for Replicate is no-op)
        pre_save_state = {k: v.full_tensor().detach().clone().cpu() for k, v in model.state_dict().items()}
        # Verify we have actual parameter data (guard against vacuous pass)
        assert len(pre_save_state) > 0, "Model has no parameters"
        assert all(v.numel() > 0 for v in pre_save_state.values()), "Some parameters are empty"

        # -- 2. Save: convert DTensors to plain tensors --
        raw_state_dict = model.state_dict()
        saved_checkpoint = convert_dtensors_to_tensors({"state_dict": raw_state_dict})

        # -- 3. Assert no DTensors in saved checkpoint --
        assert not has_dtensors(saved_checkpoint), "Saved checkpoint should contain no DTensors"
        for key, value in saved_checkpoint["state_dict"].items():
            assert isinstance(value, torch.Tensor), f"Key '{key}': expected Tensor, got {type(value)}"
            assert not isinstance(value, DTensor), f"Key '{key}': should not be DTensor after conversion"

        # Verify shapes are preserved
        for key in pre_save_state:
            assert saved_checkpoint["state_dict"][key].shape == pre_save_state[key].shape, (
                f"Key '{key}': shape mismatch after conversion — "
                f"pre_save={pre_save_state[key].shape}, saved={saved_checkpoint['state_dict'][key].shape}"
            )

        # -- 4. Persist to disk and reload --
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_rank{rank}.pt")
        torch.save(saved_checkpoint, ckpt_path)
        loaded_checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert not has_dtensors(loaded_checkpoint), "Loaded checkpoint should contain no DTensors"

        # -- 5. Load back into a fresh DTensor model --
        model_fresh = _TinyModel(hidden=8, out=4)
        model_fresh.to(device_type)
        _wrap_as_dtensor(model_fresh, device_mesh)

        # Verify fresh model has different weights (guard against vacuous pass)
        fresh_pre_load = {k: v.full_tensor().detach().clone().cpu() for k, v in model_fresh.state_dict().items()}
        weights_differ_before_load = any(not torch.equal(pre_save_state[k], fresh_pre_load[k]) for k in pre_save_state)
        assert weights_differ_before_load, "Fresh model should have different weights from saved model before load"

        # Use the strategy's conversion: serial checkpoint -> distributed state_dict
        template = model_fresh.state_dict()
        distributed_state = convert_serial_checkpoint_to_distributed_state_dict(
            checkpoint=loaded_checkpoint,
            strict=True,
            state_dict_template=template,
        )
        model_fresh.load_state_dict(distributed_state)

        # -- 6. Assert parameter identity after round-trip --
        post_load_state = {k: v.full_tensor().detach().clone().cpu() for k, v in model_fresh.state_dict().items()}
        assert pre_save_state.keys() == post_load_state.keys(), (
            f"Key mismatch: pre_save={sorted(pre_save_state.keys())}, " f"post_load={sorted(post_load_state.keys())}"
        )
        for key in pre_save_state:
            torch.testing.assert_close(
                pre_save_state[key],
                post_load_state[key],
                msg=lambda msg, k=key: (f"Round-trip mismatch for '{k}' on rank {rank}\n{msg}"),
            )

        # Verify loaded parameters are DTensors (not accidentally plain tensors)
        for name, param in model_fresh.named_parameters():
            assert isinstance(param, DTensor), f"After load, parameter '{name}' should be DTensor, got {type(param)}"

        # -- 7. Verify optimizer state redistribution --
        # Create optimizer and populate synthetic state (plain tensors).
        # This simulates what happens when loading optimizer state from a
        # checkpoint: the tensors are plain, but the parameters are DTensors.
        optimizer = torch.optim.Adam(model_fresh.parameters(), lr=1e-3)
        for param in model_fresh.parameters():
            if not isinstance(param, DTensor):
                continue
            local_data = param.to_local()
            optimizer.state[param] = {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(local_data),
                "exp_avg_sq": torch.zeros_like(local_data),
            }

        # Redistribute: plain tensors should become DTensors matching params
        _redistribute_optimizer_state_to_params(optimizer)
        for param in model_fresh.parameters():
            if not isinstance(param, DTensor):
                continue
            p_state = optimizer.state.get(param)
            assert p_state is not None, "Optimizer state missing for DTensor param"
            for buf_key in ("exp_avg", "exp_avg_sq"):
                buf_val = p_state[buf_key]
                assert isinstance(buf_val, DTensor), (
                    f"After redistribution, optimizer buffer '{buf_key}' " f"should be DTensor, got {type(buf_val)}"
                )
                assert buf_val.placements == param.placements, (
                    f"Optimizer buffer '{buf_key}' placements {buf_val.placements} "
                    f"do not match param placements {param.placements}"
                )

    finally:
        DistributedManager.cleanup()
        DistributedManager._state = {}
        monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cpu", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp1-cp2", "cpu-dp1-cp3", "cuda-dp1-cp2", "cuda-dp1-cp3"],
)
def test_checkpoint_round_trip_1d(setup_env, tmp_path):
    """Checkpoint save/load round-trip for 1D CP on a 2D mesh (dp=1, cp=2).

    Verifies:
    - DTensor state_dict is saved as plain tensors (no DTensors in checkpoint)
    - Plain-tensor checkpoint loads back into a DTensor model with correct values
    - Parameter identity is preserved across save -> disk -> load cycle
    - Optimizer state redistribution works for Replicate parameters
    - Guards against vacuous pass: parameters are verified as DTensors before
      save and after load, fresh model has different weights before load,
      and actual tensor values are compared
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    payload = (env_per_rank, device_type, backend, grid_group_sizes, ckpt_dir)
    spawn_multiprocessing(_parallel_checkpoint_round_trip, world_size, payload)
