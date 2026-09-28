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

"""1D CP stop/go (checkpoint resume) parity tests.

Ports all four stop/go sub-tests from ``test_dtensor_stop_and_go.py`` to the
1D CP topology.

The three topology-agnostic worker functions
(``_parallel_assert_optimizer_param_ordering``,
``_parallel_assert_dtensor_stop_and_go_ema``,
``_parallel_assert_cross_mode_stop_and_go``) are reused via sibling import from
the 2D module.  The two topology-specific helpers
(``_write_distogram_config``, ``_parallel_assert_boltz2_optimizer_param_ordering``)
have local 1D copies that hard-code ``cp_topology="1d"`` and
``Boltz2_1D`` respectively, so the 2D file stays byte-identical to dev-v2.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.tensor import DTensor, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.models.boltz2 import _PlaceholderModule
from boltz.distributed.model.models.boltz2_1d import Boltz2_1D as Boltz2Distributed1D
from boltz.model.models.boltz2 import Boltz2 as SerialBoltz2
from boltz.testing.utils import spawn_multiprocessing
from tests.distributed.model.models.test_dtensor_boltz2 import _prepare_serial_model
from tests.distributed.test_dtensor_stop_and_go import (
    _parallel_assert_cross_mode_stop_and_go,
    _parallel_assert_dtensor_stop_and_go_ema,
    _parallel_assert_optimizer_param_ordering,
)


def _write_distogram_config_1d(
    *,
    config_path: Path,
    output_dir: Path,
    size_dp: int,
    size_cp: int,
    accelerator: str = "cpu",
    max_epochs: int = 2,
    limit_train_batches: int = 2,
    resume: str | None = None,
    weights_seed: int = 37,
    data_seed: int = 53,
    token_z: int = 16,
    num_bins: int = 8,
    num_distograms: int = 2,
    num_conformers: int = 2,
    seq_len: int = 12,
    num_samples: int = 2,
    learning_rate: float = 1e-2,
    ema_decay: float = 0.999,
) -> None:
    """1D-CP variant of ``_write_distogram_config``.

    Writes the same YAML config layout as the 2D helper but injects
    ``cp_topology: "1d"`` at the top level so the train entrypoint dispatches
    on the 1D code path.
    """
    config: dict[str, Any] = {
        "data": {
            "seq_len": seq_len,
            "token_z": token_z,
            "num_bins": num_bins,
            "num_conformers": num_conformers,
            "num_samples": num_samples,
            "seed": data_seed,
        },
        "model": {
            "token_z": token_z,
            "num_bins": num_bins,
            "num_distograms": num_distograms,
            "num_conformers": num_conformers,
            "weights_seed": weights_seed,
            "learning_rate": learning_rate,
            "ema_decay": ema_decay,
        },
        "output": str(output_dir),
        "trainer": {
            "accelerator": accelerator,
            "devices": 1,
            "max_epochs": max_epochs,
            "limit_train_batches": limit_train_batches,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
        "parallel_size": {"size_dp": size_dp, "size_cp": size_cp},
        "precision": "FP32",
        "find_unused_parameters": False,
        "save_top_k": -1,
        "disable_checkpoint": False,
        "debug": False,
        "validation_only": False,
        "seed": 11,
        "checkpoint": {
            "monitor": None,
            "save_last": True,
            "every_n_epochs": 1,
        },
        "cp_topology": "1d",
    }
    if resume is not None:
        config["resume"] = resume
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), config_path)


def _parallel_assert_boltz2_optimizer_param_ordering_1d(rank: int, payload: tuple[Any, ...]) -> None:
    """1D-CP variant of ``_parallel_assert_boltz2_optimizer_param_ordering``.

    Mirrors the 2D worker but wraps the serial model with ``Boltz2_1D``
    instead of ``Boltz2`` so the optimizer-ordering / FQN-key invariants are
    validated on the 1D topology code path.
    """
    grid_group_sizes, device_type, backend, env_per_rank, serial_state_dict, serial_hparams = payload

    monkeypatch = pytest.MonkeyPatch()
    for key, value in env_per_rank.items():
        monkeypatch.setenv(key, f"{rank}" if value == "<INPUT_RANK>" else value)

    try:
        DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
        manager = DistributedManager()

        serial_model = SerialBoltz2(**serial_hparams)
        serial_model.load_state_dict(serial_state_dict, strict=True)
        serial_model = serial_model.to(manager.device)

        cp_model = Boltz2Distributed1D(serial_model, manager)
        cp_model = cp_model.to(manager.device)

        serial_model2 = SerialBoltz2(**serial_hparams)
        serial_model2.load_state_dict(serial_state_dict, strict=True)

        cp_names = [name for name, _ in cp_model.named_parameters()]
        serial_names = [name for name, _ in serial_model2.named_parameters()]

        cp_names_canon_set = {n.replace("._serial.", ".") for n in cp_names}
        serial_names_set = set(serial_names)

        missing = serial_names_set - cp_names_canon_set
        extra = cp_names_canon_set - serial_names_set
        assert not missing and not extra, (
            f"Parameter name set mismatch between serial and distributed Boltz2_1D.\n"
            f"  Missing from distributed: {sorted(missing)[:5]}\n"
            f"  Extra in distributed:     {sorted(extra)[:5]}"
        )

        cp_result = cp_model.configure_optimizers()
        serial_result = serial_model2.configure_optimizers()

        cp_opt = cp_result[0][0] if isinstance(cp_result, tuple) else cp_result
        serial_opt = serial_result[0][0] if isinstance(serial_result, tuple) else serial_result

        serial_trainable_names = {n for n, p in serial_model2.named_parameters() if p.requires_grad}
        serial_opt_total = sum(len(g["params"]) for g in serial_opt.param_groups)
        assert len(serial_trainable_names) == serial_opt_total, (
            f"Serial optimizer param count ({serial_opt_total}) doesn't match "
            f"serial trainable named_parameters count ({len(serial_trainable_names)})"
        )

        placeholder_prefixes = tuple(
            f"{name}." for name, mod in cp_model.named_modules() if isinstance(mod, _PlaceholderModule)
        )
        serial_placeholder_names = {n for n in serial_trainable_names if n.startswith(placeholder_prefixes)}
        serial_non_placeholder_names = serial_trainable_names - serial_placeholder_names

        cp_opt_count = sum(len(g["params"]) for g in cp_opt.param_groups)
        cp_trainable_names = {n for n, p in cp_model.named_parameters() if p.requires_grad}
        assert cp_opt_count == len(cp_trainable_names), (
            f"Distributed optimizer param count ({cp_opt_count}) doesn't match "
            f"distributed trainable param count ({len(cp_trainable_names)})"
        )
        assert cp_opt_count > 0, "Distributed optimizer has zero params"

        assert cp_opt_count == len(serial_non_placeholder_names), (
            f"Distributed optimizer has {cp_opt_count} params but serial model "
            f"has {len(serial_non_placeholder_names)} non-placeholder trainable params "
            f"(total serial trainable: {len(serial_trainable_names)}, "
            f"placeholder: {len(serial_placeholder_names)})"
        )

        cp_trainable_canon = {n.replace("._serial.", ".") for n in cp_trainable_names}
        assert cp_trainable_canon == serial_non_placeholder_names, (
            f"Trainable distributed params don't match serial non-placeholder params.\n"
            f"  Only in distributed: {sorted(cp_trainable_canon - serial_non_placeholder_names)[:5]}\n"
            f"  Only in serial:      {sorted(serial_non_placeholder_names - cp_trainable_canon)[:5]}"
        )

        serial_shapes = {n: p.shape for n, p in serial_model2.named_parameters() if p.requires_grad}
        shape_mismatches = []
        for n, p in cp_model.named_parameters():
            if not p.requires_grad:
                continue
            canon = n.replace("._serial.", ".")
            serial_shape = serial_shapes.get(canon)
            if serial_shape is None:
                continue
            cp_shape = p.shape
            if cp_shape != serial_shape:
                shape_mismatches.append((canon, cp_shape, serial_shape))
        assert not shape_mismatches, "Parameter shape mismatches (distributed vs serial):\n" + "\n".join(
            f"  {n}: {cs} vs {ss}" for n, cs, ss in shape_mismatches[:10]
        )

        for p in cp_model.parameters():
            p.grad = torch.randn_like(p.to_local() if isinstance(p, DTensor) else p)
            if isinstance(p, DTensor):
                p.grad = distribute_tensor(p.grad, device_mesh=p.device_mesh, placements=p.placements)
        cp_opt.step()

        fqn_sd = get_optimizer_state_dict(cp_model, cp_opt)
        fqn_state_keys = sorted(fqn_sd["state"].keys())

        assert all(isinstance(k, str) for k in fqn_state_keys), (
            f"get_optimizer_state_dict should return FQN string keys, "
            f"got types {[type(k).__name__ for k in fqn_state_keys[:3]]}"
        )
        cp_trainable_names = sorted(n for n, p in cp_model.named_parameters() if p.requires_grad)
        assert fqn_state_keys == cp_trainable_names, (
            f"FQN optimizer state keys don't match trainable named_parameters().\n"
            f"  FQN keys (first 5):          {fqn_state_keys[:5]}\n"
            f"  trainable params (first 5):  {cp_trainable_names[:5]}"
        )

        for p in serial_model2.parameters():
            p.grad = torch.randn_like(p)
        serial_opt.step()

        serial_fqn_sd = get_optimizer_state_dict(serial_model2, serial_opt)
        serial_fqn_keys = set(serial_fqn_sd["state"].keys())

        cp_fqn_canon = {k.replace("._serial.", ".") for k in fqn_state_keys}
        not_in_serial = cp_fqn_canon - serial_fqn_keys
        assert not not_in_serial, (
            f"Distributed optimizer FQN keys not found in serial optimizer:\n" f"  {sorted(not_in_serial)[:5]}"
        )
    finally:
        DistributedManager.cleanup()
        DistributedManager._state = {}
        monkeypatch.undo()


# ---------------------------------------------------------------------------
# 1. test_stop_and_go_via_train_entrypoint_1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    [
        ((2, 4), True, "cpu", "ENV"),
        ((2, 1), True, "cuda", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp2-cp4", "cuda-dp2-cp1", "cuda-dp1-cp2"],
)
def test_stop_and_go_via_train_entrypoint_1d(setup_env, tmp_path):
    """Goals: 1D CP checkpoint resume parity through the real ``train()`` entrypoint.

    - Continuous 2-epoch run matches stop-at-epoch-1 + resume-to-epoch-2
    - Model weights, optimizer state, EMA shadow weights all match exactly
    - Epoch and global_step counters match
    - Validates ``BoltzContextParallelStrategy`` checkpoint conversion roundtrip
    - Validates ``cfg.resume`` auto-resume path in ``train.py`` under 1D topology
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    ema_decay = 0.999
    size_dp = int(grid_group_sizes["dp"])
    cp_group = grid_group_sizes["cp"]
    size_cp = int(cp_group[0] * cp_group[1]) if isinstance(cp_group, tuple) else int(cp_group)
    accelerator = "gpu" if device_type == "cuda" else "cpu"

    continuous_dir = tmp_path / "continuous"
    stopgo_dir = tmp_path / "stopgo"

    common_kwargs: dict[str, Any] = {
        "size_dp": size_dp,
        "size_cp": size_cp,
        "accelerator": accelerator,
        "ema_decay": ema_decay,
    }

    continuous_config = continuous_dir / "config.yaml"
    _write_distogram_config_1d(config_path=continuous_config, output_dir=continuous_dir, max_epochs=2, **common_kwargs)

    stage1_config = stopgo_dir / "config_stage1.yaml"
    _write_distogram_config_1d(config_path=stage1_config, output_dir=stopgo_dir, max_epochs=1, **common_kwargs)

    stage2_config = stopgo_dir / "config_stage2.yaml"
    _write_distogram_config_1d(
        config_path=stage2_config,
        output_dir=stopgo_dir,
        max_epochs=2,
        resume=str(stopgo_dir / "last.ckpt"),
        **common_kwargs,
    )

    payload = (
        env_per_rank,
        str(continuous_config),
        str(stage1_config),
        str(stage2_config),
        str(continuous_dir),
        str(stopgo_dir),
    )
    spawn_multiprocessing(_parallel_assert_dtensor_stop_and_go_ema, world_size, payload)


# ---------------------------------------------------------------------------
# 2. test_optimizer_param_ordering_serial_vs_distributed_1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    [
        ((2, 4), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp2-cp4"],
)
def test_optimizer_param_ordering_serial_vs_distributed_1d(setup_env):
    """Goals: 1D CP optimizer parameter ordering and FQN key consistency.

    - named_parameters() yields identical key lists for both model types
    - Optimizers see the same number of parameters
    - get_optimizer_state_dict produces FQN string keys matching named_parameters()
    - FQN keys are identical between serial and distributed models
    - Guards against silent optimizer state misalignment on cross-topology resume
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    spawn_multiprocessing(
        _parallel_assert_optimizer_param_ordering,
        world_size,
        (grid_group_sizes, device_type, backend, env_per_rank),
    )


# ---------------------------------------------------------------------------
# 3. test_optimizer_param_ordering_boltz2_1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    [
        ((2, 4), True, "cpu", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp2-cp4", "cuda-dp1-cp2"],
)
def test_optimizer_param_ordering_boltz2_1d(setup_env):
    """Goals: Boltz-2 1D CP optimizer parameter ordering and FQN key consistency.

    Extends tiny-distogram harness coverage to the full Boltz-2 wrapper under
    1D topology, catching registration-order bugs that the smaller harness
    cannot exercise. The 1D-local worker
    ``_parallel_assert_boltz2_optimizer_param_ordering_1d`` wraps the serial
    model with ``Boltz2_1D``.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    serial_state_dict, serial_hparams = _prepare_serial_model(ema=False)

    spawn_multiprocessing(
        _parallel_assert_boltz2_optimizer_param_ordering_1d,
        world_size,
        (grid_group_sizes, device_type, backend, env_per_rank, serial_state_dict, serial_hparams),
    )


# ---------------------------------------------------------------------------
# 4. test_cross_mode_stop_and_go_1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "setup_env",
    [
        ((2, 4), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp2-cp4"],
)
def test_cross_mode_stop_and_go_1d(setup_env, tmp_path):
    """Goals: 1D CP cross-mode checkpoint interop in both directions.

    - Serial→distributed: 1-epoch serial + resume as 1D distributed matches
      2-epoch continuous 1D distributed baseline
    - Distributed→serial: 1-epoch 1D distributed + resume as serial matches
      2-epoch continuous serial baseline
    - Validates BoltzContextParallelStrategy strips DTensor metadata for
      portable checkpoints under 1D topology
    - Cross-mode interop inherited from BoltzContextParallelStrategy because
      boltz2_1d.py has zero state_dict overrides
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    output_dir = tmp_path / "cross_mode_output"
    payload = (grid_group_sizes, device_type, backend, env_per_rank, str(output_dir))
    spawn_multiprocessing(_parallel_assert_cross_mode_stop_and_go, world_size, payload)
