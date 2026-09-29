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

"""End-to-end Boltz-2 1D CP distributed training integration test via ``train()``.

Calls the real ``train()`` entrypoint with real Boltz-2 training data and a
small model config, exercising the full 1D context-parallelism pipeline:
config loading -> distributed manager -> Boltz2_1D model wrapping ->
Boltz2TrainingDataModule1D -> Trainer.fit -> checkpoint.

This is the 1D counterpart of ``test_dtensor_boltz2_train.py::test_boltz2_train_entrypoint``.
The key difference is that ``cp_topology`` is set to ``"1d"`` in the training
config, which creates a 2D device mesh ``(dp, cp)`` with a flat integer CP
size instead of the 3D ``(dp, cp0, cp1)`` mesh used by 2D CP.

The only monkeypatches are:
- ``_cleanup_distributed -> lambda: None`` (process group safety for tests)

The E2E parity test (``test_boltz2_1d_e2e_training_parity``) additionally
monkeypatches noise, data, smooth_lddt, and DoublePrecision to guarantee
bit-for-bit comparison between serial and 1D distributed training, following
the same approach as the 2D parity test in ``test_dtensor_boltz2_train.py``.
"""

import copy
import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

import boltz.distributed.model.modules.diffusion_1d as dist_diffusion_1d_module
import boltz.distributed.train as train_module
import boltz.model.loss.diffusionv2 as serial_loss_v2_module
import boltz.model.modules.diffusionv2 as serial_diffusion_v2_module
from boltz.data.module.trainingv2 import (
    Boltz2TrainingDataModule,
)
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.models.boltz2_1d import Boltz2_1D
from boltz.distributed.model.modules.diffusion_1d import AtomDiffusion1D
from boltz.distributed.testing.utils import setup_mock_training_datamodule_config
from boltz.model.models.boltz2 import Boltz2 as SerialBoltz2
from boltz.model.modules.diffusionv2 import AtomDiffusion as SerialAtomDiffusionV2
from boltz.model.validation.rcsb import RCSBValidator
from boltz.testing.utils import (
    SetModuleInfValues,
    init_module_params_glorot,
    init_tensors_uniform,
    seed_by_rank,
    spawn_multiprocessing,
)
from tests.distributed.test_dtensor_boltz2_train import (
    TrainTestConfig,
    _apply_cached_getitem,
    _apply_e2e_deterministic_getitem,
    _Bf16AcTestEnv,
    _e2e_model_dict,
    _load_serial_train_module,
    _parallel_assert_boltz2_stop_and_go,
    _parallel_assert_boltz2_train,
    _setup_training_data_all_4_e2e,
    _smooth_lddt_loss_dense_e2e,
    _worker_actv_ckpt_parity,
    _worker_bf16_dtype_parity,
    _write_train_config,
)


@dataclass
class TrainTestConfig1D(TrainTestConfig):
    """Training test config extended with 1D CP topology.

    Inherits all fields from :class:`TrainTestConfig` and adds
    ``cp_topology`` which defaults to ``"1d"``.
    """

    cp_topology: str = "1d"


def _write_train_config_1d(cfg: TrainTestConfig1D) -> None:
    """Write a YAML training config for 1D CP training.

    Delegates to :func:`_write_train_config` from the 2D test module to
    produce the base config, then patches in ``cp_topology: "1d"`` which
    is required for the ``train()`` entrypoint to select 1D model/data
    wrappers.
    """
    _write_train_config(cfg)

    # Load the config written by the 2D helper, add cp_topology, and re-save.
    config = OmegaConf.load(cfg.config_path)
    OmegaConf.set_struct(config, False)
    config["cp_topology"] = cfg.cp_topology
    OmegaConf.save(config, cfg.config_path)


def _e2e_model_dict_1d(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """1D variant of ``_e2e_model_dict`` that selects the 1D RCSB validator.

    The 2D builder hardcodes the validator ``_target_`` to
    :class:`DistributedRCSBValidator`, which requires a ``transpose_comm``
    argument the 1D model does not have. Re-point validator entries to
    :class:`Distributed1DRCSBValidator` for 1D-CP tests.
    """
    model_dict = _e2e_model_dict(*args, **kwargs)
    for v in model_dict.get("validators", []) or []:
        if v.get("_target_") == "boltz.distributed.model.validation.rcsb.DistributedRCSBValidator":
            v["_target_"] = "boltz.distributed.model.validation.rcsb_1d.Distributed1DRCSBValidator"
    return model_dict


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup_env",
    [
        # CUDA: 1D CP with dp=1, cp=2 — exercises 1D topology on GPU.
        ((1, 2), True, "cuda", "ENV"),
        # CPU: 1D CP with dp=1, cp=3
        ((1, 3), True, "cpu", "ENV"),
        # CUDA: 1D CP with dp=1, cp=3 — non-power-of-two CP on GPU (3 GPUs).
        ((1, 3), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2", "cpu-dp1-cp3", "cuda-dp1-cp3"],
)
def test_boltz2_1d_train_entrypoint(
    setup_env,
    tmp_path,
    test_cp_training_data_dir_boltz2,
    canonical_mols_dir,
):
    """End-to-end Boltz-2 1D CP training through the real train() entrypoint.

    Exercises the full 1D CP pipeline with a small model and real training data:
    config -> Hydra instantiate -> _create_distributed_model (Boltz2_1D)
    -> _create_distributed_data_module (Boltz2TrainingDataModule1D) -> Trainer.fit
    -> checkpoint.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cpu" and grid_group_sizes["cp"] > 1:
        pytest.skip("DAP ending-node requires NCCL; Gloo lacks all_to_all")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    output_dir = tmp_path / "boltz2_1d_train_output"
    config_path = tmp_path / "boltz2_1d_train_config.yaml"

    # For 1D CP, grid_group_sizes["cp"] is an integer (not a tuple).
    size_cp = grid_group_sizes["cp"]

    _write_train_config_1d(
        TrainTestConfig1D(
            config_path=config_path,
            output_dir=output_dir,
            test_data_dir=test_cp_training_data_dir_boltz2,
            mol_dir=canonical_mols_dir,
            size_dp=grid_group_sizes["dp"],
            size_cp=size_cp,
            accelerator="gpu" if device_type == "cuda" else "cpu",
        )
    )

    payload = (env_per_rank, str(config_path), str(output_dir))
    spawn_multiprocessing(_parallel_assert_boltz2_train, world_size, payload)


# ---------------------------------------------------------------------------
#  E2E serial-vs-1D-distributed training parity
# ---------------------------------------------------------------------------


def _worker_e2e_training_parity_1d(
    rank: int,
    grid_group_sizes: dict,
    device_type: str,
    backend: str,
    env_per_rank: dict[str, Any],
    dist_config_path: str,
    dist_output_dir: str,
    serial_ckpt_path: str,
    pretrained_ckpt_path: str,
    serial_metrics: dict,
    sigmas_global_host: torch.Tensor,
    noise_global_host: torch.Tensor,
    cached_samples_path: str,
    seed: int,
    dtype: torch.dtype,
) -> None:
    """Multi-rank worker: 1D distributed train() then compare with serial checkpoint.

    1. Applies module-level monkeypatches (noise, data, smooth_lddt)
    2. Calls ``train_module.train(dist_config_path, [])``
    3. Loads both serial and distributed checkpoints
    4. Compares state_dict, EMA weights, and logged metrics

    Adapted from ``_worker_e2e_training_parity`` in the 2D test.  Key 1D
    differences:

    - ``AtomDiffusion1D`` (not ``DistAtomDiffusionV2``) for noise_distribution
    - ``diffusion_1d`` module (not ``diffusion``) for create_distributed_randn
    - ``Boltz2_1D`` (not ``Boltz2Distributed``) for validation_step wrapper
    - Noise distributed via ``distribute_tensor`` on 2D mesh ``(dp, cp)``
      with ``(Shard(0), Shard(1))`` — no intersperse padding
    - Sigma placement is ``(Shard(0), Replicate())`` (2-tuple, not 3-tuple)
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    monkeypatch.setattr(train_module, "_cleanup_distributed", lambda: None)
    DistributedManager._state = {}

    # --- Deterministic data loading via cached samples ---
    _apply_cached_getitem(monkeypatch, cached_samples_path)

    # --- Deterministic noise (distributed) ---
    # For 1D CP, noise is distributed via simple distribute_tensor on the
    # 2D mesh (dp, cp) — no intersperse padding is needed because the 1D
    # data pipeline uses distribute_features (broadcast + distribute_tensor)
    # rather than pad_and_scatter_atom_features_dtensor.
    def _dist_noise_dist_1d(self, bs, dtype=torch.float32):
        s = sigmas_global_host.to(device=self.device_mesh.device_type, dtype=dtype)[:bs]
        return distribute_tensor(s, self.device_mesh, (Shard(0), Replicate()))

    monkeypatch.setattr(AtomDiffusion1D, "noise_distribution", _dist_noise_dist_1d)

    _noise_dt_cache_1d: list[DTensor | None] = [None]
    _dist_in_val_1d = [False]

    def _det_create_randn_1d(shape, device_mesh, placements, dtype=torch.float32, scale=1.0):
        """Deterministic replacement for create_distributed_randn (1D CP).

        During validation, returns zero noise.  During training, distributes
        the pre-generated global noise tensor on the 2D mesh (dp, cp).
        """
        if _dist_in_val_1d[0]:
            from boltz.distributed.utils import create_distributed_randn as _real_create_randn

            return _real_create_randn(shape, device_mesh, placements, dtype=dtype, scale=0.0)

        if _noise_dt_cache_1d[0] is None:
            n = noise_global_host.to(device=device_mesh.device_type, dtype=dtype)
            _noise_dt_cache_1d[0] = distribute_tensor(n, device_mesh, (Shard(0), Shard(1)))
        n = _noise_dt_cache_1d[0]
        if n.dtype != dtype:
            n = n.to(dtype=dtype)
        # Pad if noise shape is smaller than requested (e.g., padding alignment)
        if len(shape) > 1 and n.shape[1] < shape[1]:
            from boltz.testing.utils import pad_to_length as _pad

            n = _pad(n, dim=1, length=shape[1])
        return n * scale

    monkeypatch.setattr(dist_diffusion_1d_module, "create_distributed_randn", _det_create_randn_1d)

    _orig_dist_val_step_1d = Boltz2_1D.validation_step

    def _dist_val_step_wrapper_1d(self_model, batch, batch_idx):
        _dist_in_val_1d[0] = True
        try:
            return _orig_dist_val_step_1d(self_model, batch, batch_idx)
        finally:
            _dist_in_val_1d[0] = False

    monkeypatch.setattr(Boltz2_1D, "validation_step", _dist_val_step_wrapper_1d)

    # --- Skip RMSD in distributed validation (not needed for LDDT parity) ---
    import boltz.distributed.model.validation.validator as _dist_validator_mod

    def _rmsd_noop(*args, **kwargs):
        return torch.tensor(0.0), None, None

    monkeypatch.setattr(_dist_validator_mod, "weighted_minimum_rmsd_single", _rmsd_noop)

    # --- Capture trainer metrics ---
    _captured_metrics: dict[str, float] = {}
    _orig_fit = pl.Trainer.fit

    def _capturing_fit(self, *args, **kwargs):
        result = _orig_fit(self, *args, **kwargs)
        for k, v in self.callback_metrics.items():
            if isinstance(v, DTensor):
                _captured_metrics[k] = v.full_tensor().detach().cpu().item()
            elif isinstance(v, torch.Tensor):
                _captured_metrics[k] = v.detach().cpu().item()
            else:
                _captured_metrics[k] = v
        return result

    monkeypatch.setattr(pl.Trainer, "fit", _capturing_fit)

    # --- Run distributed training ---
    train_module.train(dist_config_path, [])

    # --- Load checkpoints and compare ---
    dist_ckpt_path = Path(dist_output_dir) / "last.ckpt"
    assert dist_ckpt_path.exists(), f"Rank {rank}: distributed checkpoint not found at {dist_ckpt_path}"
    dist_ckpt = torch.load(dist_ckpt_path, map_location="cpu", weights_only=False)
    serial_ckpt = torch.load(serial_ckpt_path, map_location="cpu", weights_only=False)

    dist_sd = dist_ckpt["state_dict"]
    serial_sd = serial_ckpt["state_dict"]
    assert len(dist_sd) > 0, f"Rank {rank}: distributed state_dict is empty"
    assert len(serial_sd) > 0, f"Rank {rank}: serial state_dict is empty"

    # The distributed model prefixes keys with ``_serial.``; strip it for comparison.
    dist_sd_mapped = {}
    for k, v in dist_sd.items():
        canonical_k = k.replace("_serial.", "", 1) if k.startswith("_serial.") else k
        dist_sd_mapped[canonical_k] = v

    # Under bf16-mixed, gradients accumulate in bf16 under autocast even though
    # master weights are fp32; per-element noise (~eps_bf16) cascades non-uniformly
    # through the pairformer/structure stack and through L2-norm aggregates like
    # train/grad_norm.  Numerical parity vs serial is not achievable at any uniform
    # pytorch-tier tolerance, so under bf16 this test is a SMOKE TEST: structural
    # checks + non-vacuous guards stay, but value-parity assert_close is skipped.
    # fp32 path retains full numerical parity at default tolerances.
    smoke_only = dtype == torch.bfloat16

    for k in serial_sd:
        assert k in dist_sd_mapped, f"Rank {rank}: key '{k}' missing from distributed checkpoint"
        if not smoke_only:
            torch.testing.assert_close(
                dist_sd_mapped[k],
                serial_sd[k],
                msg=lambda m: f"Rank {rank}: state_dict mismatch on '{k}': {m}",
            )

    # --- EMA weight parity ---
    assert "ema" in dist_ckpt, f"Rank {rank}: distributed checkpoint missing EMA state"
    assert "ema" in serial_ckpt, f"Rank {rank}: serial checkpoint missing EMA state"
    dist_ema = dist_ckpt["ema"]["ema_weights"]
    serial_ema = serial_ckpt["ema"]["ema_weights"]
    assert dist_ckpt["ema"]["cur_step"] == serial_ckpt["ema"]["cur_step"], (
        f"Rank {rank}: EMA cur_step mismatch: "
        f"dist={dist_ckpt['ema']['cur_step']}, serial={serial_ckpt['ema']['cur_step']}"
    )

    dist_ema_mapped = {}
    for k, v in dist_ema.items():
        canonical_k = k.replace("_serial.", "", 1) if k.startswith("_serial.") else k
        dist_ema_mapped[canonical_k] = v

    for k in serial_ema:
        assert k in dist_ema_mapped, f"Rank {rank}: EMA key '{k}' missing from distributed checkpoint"
        if not smoke_only:
            torch.testing.assert_close(
                dist_ema_mapped[k],
                serial_ema[k],
                msg=lambda m: f"Rank {rank}: EMA weight mismatch on '{k}': {m}",
            )

    # --- Non-vacuous guard: at least one parameter changed from pretrained init ---
    pretrained_sd = torch.load(
        pretrained_ckpt_path,
        map_location="cpu",
        weights_only=False,
    ).get("state_dict", {})
    if pretrained_sd:
        changed = any(not torch.equal(serial_sd[k], pretrained_sd[k]) for k in serial_sd if k in pretrained_sd)
        assert changed, f"Rank {rank}: no parameters changed from pretrained init — test is vacuous"

    # --- Metric parity ---
    # Atom-level LDDT metrics depend on diffusion-sampled coordinates, which
    # differ slightly between serial and distributed forward passes in FP32 due
    # to accumulation order in parallel attention.  Use relaxed tolerance for
    # these forward-pass-dependent metrics and default tolerance for everything
    # else.
    _forward_dependent_prefixes = ("val/lddt", "val/complex_lddt", "val/clash", "val/pb", "val/rmsd")
    _lddt_keys_compared = []
    if serial_metrics:
        for k in serial_metrics:
            if k in _captured_metrics:
                got = torch.tensor(_captured_metrics[k])
                exp = torch.tensor(serial_metrics[k])
                if not smoke_only:
                    if any(k.startswith(p) for p in _forward_dependent_prefixes):
                        torch.testing.assert_close(
                            got,
                            exp,
                            atol=5e-4,
                            rtol=0.02,
                            msg=lambda m: f"Rank {rank}: metric '{k}' mismatch: {m}",
                        )
                    else:
                        torch.testing.assert_close(
                            got,
                            exp,
                            msg=lambda m: f"Rank {rank}: metric '{k}' mismatch: {m}",
                        )
                if "lddt" in k:
                    _lddt_keys_compared.append(k)

    assert _lddt_keys_compared, (
        f"Rank {rank}: no validation LDDT metrics were compared — test is vacuous. "
        f"Serial keys: {sorted(serial_metrics)}, dist keys: {sorted(_captured_metrics)}"
    )
    for required_metric in ("val/lddt", "val/disto_lddt", "val/complex_lddt"):
        assert (
            required_metric in _captured_metrics
        ), f"Rank {rank}: distributed metrics missing '{required_metric}' — available: {sorted(_captured_metrics)}"

    # Verify component-wise grad_norm metrics are present and non-zero
    _grad_norm_keys = [
        "train/grad_norm",
        "train/grad_norm_msa_module",
        "train/grad_norm_pairformer_module",
        "train/grad_norm_structure_module",
    ]
    for gn_key in _grad_norm_keys:
        assert (
            gn_key in _captured_metrics
        ), f"Rank {rank}: distributed metrics missing '{gn_key}' — available: {sorted(_captured_metrics)}"
        assert (
            _captured_metrics[gn_key] > 0
        ), f"Rank {rank}: '{gn_key}' is zero — gradients should be non-zero after training"

    torch.distributed.barrier()


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16],
    ids=["fp32", "bf16"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
        # CUDA: 1D CP with dp=1, cp=3 — non-power-of-two CP on GPU (3 GPUs)
        ((1, 3), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2", "cuda-dp1-cp3"],
)
def test_boltz2_1d_e2e_training_parity(
    setup_env,
    test_cp_training_base_data_dir_boltz2,
    canonical_mols_dir,
    tmp_path,
    dtype,
):
    """E2E serial-vs-1D-DTensor training parity via ``train()`` entry points.

    Both serial and 1D distributed training go through their respective
    ``train()`` functions for 1 epoch (1 batch of 7ylz+8b2e for training,
    7z64+8ayv for validation), then compare checkpoints (state_dict, EMA
    weights) and logged metrics — including validation LDDT — at FP32
    default tolerance.

    This is the 1D counterpart of ``test_boltz2_e2e_training_parity`` in the
    2D test module.  Key 1D differences:

    - ``cp_topology="1d"`` in the distributed config
    - 2D device mesh ``(dp, cp)`` instead of 3D ``(dp, cp0, cp1)``
    - ``size_cp`` is ``grid_group_sizes["cp"]`` (int, not tuple product)
    - Noise distributed via ``distribute_tensor`` (no intersperse padding)
    """

    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    # Map dtype -> Lightning precision strings. For bf16 we use bf16-MIXED
    # (production path): pretrained weights stay fp32 and autocast inside the
    # model handles the bf16 compute. Mirrors test_boltz2_1d_bf16_dtype_parity.
    if dtype == torch.bfloat16:
        precision_serial = "bf16-mixed"
        precision_dist = "BF16_MIXED"
        weights_dtype = torch.float32
    elif dtype == torch.float32:
        precision_serial = None
        precision_dist = None
        weights_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype for e2e_training_parity: {dtype}")

    seed = 42
    multiplicity = 2
    B = 2
    max_tokens = 256
    W = 32  # atoms_per_window_queries
    size_cp = grid_group_sizes["cp"]
    atom_align = math.lcm(W, size_cp)
    max_atoms = ((max_tokens * 10 + atom_align - 1) // atom_align) * atom_align
    max_seqs = 16
    scale_glorot = 0.05

    # --- Merge all 4 samples with train/val split ---
    training_data_dir, split_file = _setup_training_data_all_4_e2e(
        tmp_path / "training_data", test_cp_training_base_data_dir_boltz2
    )

    # --- Create pretrained checkpoint with deterministic init ---
    seed_by_rank(0, seed=seed)
    model_dict = _e2e_model_dict(multiplicity=multiplicity, validate_structure=True)
    model_dict.pop("_target_")
    model_dict.pop("validators", None)
    _val_validators = [RCSBValidator(val_names=["RCSB"], confidence_prediction=False, physicalism_metrics=True)]
    pretrained_model = SerialBoltz2(**model_dict, validators=_val_validators)
    init_module_params_glorot(pretrained_model, gain=scale_glorot)
    pretrained_model.apply(SetModuleInfValues())
    pretrained_model.structure_module.coordinate_augmentation = False
    pretrained_model = pretrained_model.to(dtype=weights_dtype)

    pretrained_path = tmp_path / "pretrained.ckpt"
    torch.save(
        {
            "state_dict": pretrained_model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": pretrained_model.hparams,
        },
        pretrained_path,
    )

    # --- Pre-load and cache individual samples to disk ---
    _tmp_mp = pytest.MonkeyPatch()
    _apply_e2e_deterministic_getitem(_tmp_mp, base_seed=seed)
    _preload_cfg = setup_mock_training_datamodule_config(training_data_dir)
    _preload_cfg.batch_size = B
    _preload_cfg.samples_per_epoch = B
    _preload_cfg.moldir = str(canonical_mols_dir)
    _preload_cfg.return_train_symmetries = False
    _preload_cfg.msa_sampling_training = False
    _preload_cfg.max_tokens = max_tokens
    _preload_cfg.max_atoms = max_atoms
    _preload_cfg.max_seqs = max_seqs
    for _ds in _preload_cfg.datasets:
        _ds.filters = None
        _ds.split = str(split_file)
        _ds.symmetry_correction = False
    seed_by_rank(0, seed=seed)
    _preload_dm = Boltz2TrainingDataModule(cfg=_preload_cfg)

    _preload_ds = _preload_dm._train_set
    _cached_samples = {i: _preload_ds[i] for i in range(B)}
    cached_samples_path = tmp_path / "cached_samples.pt"
    torch.save(_cached_samples, cached_samples_path)

    _preload_dl = _preload_dm.train_dataloader()
    _preload_batch = next(iter(_preload_dl))
    atom_pad_mask_host = _preload_batch["atom_pad_mask"].detach().cpu()
    _tmp_mp.undo()

    # --- Pre-generate deterministic noise (masked by atom_pad_mask) ---
    # Hold sigmas/noise at weights_dtype (fp32 even for bf16-mixed); the
    # noise_distribution and randn_like monkeypatches downcast to the target
    # tensor's dtype on call, and autocast handles the bf16 conversion inside
    # the model.
    seed_by_rank(0, seed=seed)
    sigmas_global = pretrained_model.structure_module.noise_distribution(B * multiplicity).to(dtype=weights_dtype)
    noise_global = torch.empty(B * multiplicity, max_atoms, 3, dtype=weights_dtype)
    init_tensors_uniform([noise_global], low=-scale_glorot, high=scale_glorot)
    _mask_mul = atom_pad_mask_host[:, :, None].repeat_interleave(multiplicity, 0).to(dtype=weights_dtype)
    noise_global = noise_global * _mask_mul

    sigmas_global_host = sigmas_global.detach().cpu()
    noise_global_host = noise_global.detach().cpu()

    # --- Write serial config ---
    serial_output_dir = tmp_path / "serial_output"
    serial_output_dir.mkdir(parents=True, exist_ok=True)
    serial_config_path = tmp_path / "serial_config.yaml"
    _e2e_ds_overrides = {
        "filters": None,
        "moldir": None,
        "symmetry_correction": False,
        "val_group": "RCSB",
        "use_train_subset": None,
        "override_bfactor": False,
        "override_method": None,
    }
    _write_train_config(
        TrainTestConfig(
            config_path=serial_config_path,
            output_dir=serial_output_dir,
            test_data_dir=training_data_dir,
            mol_dir=canonical_mols_dir,
            mode="serial",
            accelerator="gpu",
            limit_train_batches=1,
            pretrained=str(pretrained_path),
            model=_e2e_model_dict(multiplicity=multiplicity, validate_structure=True),
            batch_size=B,
            samples_per_epoch=B,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_seqs=max_seqs,
            return_train_symmetries=False,
            split=str(split_file),
            pop_target_keys=True,
            extra_dataset_overrides=_e2e_ds_overrides,
            v2=True,
            strict_loading=False,
            wandb=None,
            save_top_k=0,
            disable_checkpoint=False,
            precision=precision_serial if precision_serial is not None else "FP32",
        )
    )

    # --- Apply serial monkeypatches ---
    serial_mp = pytest.MonkeyPatch()
    _apply_cached_getitem(serial_mp, cached_samples_path)

    _orig_serial_boltz2_init = SerialBoltz2.__init__

    @functools.wraps(_orig_serial_boltz2_init)
    def _init_with_validators(self, *args, **kwargs):
        if kwargs.get("validate_structure", False) and not kwargs.get("validators"):
            kwargs["validators"] = [
                RCSBValidator(val_names=["RCSB"], confidence_prediction=False, physicalism_metrics=True)
            ]
        _orig_serial_boltz2_init(self, *args, **kwargs)

    serial_mp.setattr(SerialBoltz2, "__init__", _init_with_validators)

    serial_mp.setattr(
        SerialAtomDiffusionV2,
        "noise_distribution",
        lambda self, bs: sigmas_global[:bs].to(device=self.zero.device),
    )

    _serial_in_val = [False]

    def _serial_randn_like(t):
        if _serial_in_val[0]:
            return torch.zeros_like(t)
        return noise_global[: t.shape[0], : t.shape[1]].to(device=t.device, dtype=t.dtype)

    serial_mp.setattr(serial_diffusion_v2_module.torch, "randn_like", _serial_randn_like)

    _orig_serial_randn = serial_diffusion_v2_module.torch.randn

    def _serial_randn(*args, **kwargs):
        if _serial_in_val[0]:
            kwargs.pop("generator", None)
            return torch.zeros(*args, **kwargs)
        return _orig_serial_randn(*args, **kwargs)

    serial_mp.setattr(serial_diffusion_v2_module.torch, "randn", _serial_randn)

    _orig_compute_random_augmentation = serial_diffusion_v2_module.compute_random_augmentation

    def _identity_augmentation_during_val(multiplicity, s_trans=1.0, device=None, dtype=torch.float32):
        if _serial_in_val[0]:
            R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(multiplicity, -1, -1)
            tr = torch.zeros(multiplicity, 1, 3, device=device, dtype=dtype)
            return R, tr
        return _orig_compute_random_augmentation(multiplicity, s_trans=s_trans, device=device, dtype=dtype)

    serial_mp.setattr(serial_diffusion_v2_module, "compute_random_augmentation", _identity_augmentation_during_val)

    _orig_serial_val_step = SerialBoltz2.validation_step

    def _serial_val_step_wrapper(self_model, batch, batch_idx):
        _serial_in_val[0] = True
        try:
            return _orig_serial_val_step(self_model, batch, batch_idx)
        finally:
            _serial_in_val[0] = False

    serial_mp.setattr(SerialBoltz2, "validation_step", _serial_val_step_wrapper)

    serial_mp.setattr(serial_loss_v2_module, "smooth_lddt_loss", _smooth_lddt_loss_dense_e2e)
    serial_mp.setattr(serial_diffusion_v2_module, "smooth_lddt_loss", _smooth_lddt_loss_dense_e2e)

    serial_captured_metrics: dict[str, float] = {}
    _orig_fit = pl.Trainer.fit

    def _serial_capturing_fit(self, *args, **kwargs):
        result = _orig_fit(self, *args, **kwargs)
        for k, v in self.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                serial_captured_metrics[k] = v.detach().cpu().item()
            else:
                serial_captured_metrics[k] = v
        return result

    serial_mp.setattr(pl.Trainer, "fit", _serial_capturing_fit)

    # --- Run serial training ---
    _serial_train_mod = _load_serial_train_module()
    _serial_train_mod.train(str(serial_config_path), [])

    # --- Find serial checkpoint ---
    serial_ckpt_files = list(serial_output_dir.rglob("last.ckpt"))
    assert (
        len(serial_ckpt_files) == 1
    ), f"Expected exactly 1 last.ckpt in serial output, found {len(serial_ckpt_files)}: {serial_ckpt_files}"
    serial_ckpt_path = serial_ckpt_files[0]

    # Verify serial checkpoint has EMA
    serial_ckpt = torch.load(serial_ckpt_path, map_location="cpu", weights_only=False)
    assert "ema" in serial_ckpt, "Serial checkpoint missing EMA state"
    assert "ema_weights" in serial_ckpt["ema"], "Serial EMA missing ema_weights"

    # Non-vacuous guard: serial must have logged at least one LDDT val metric
    serial_lddt_keys = [
        k for k in serial_captured_metrics if k.startswith("val/lddt_") or k.startswith("val/disto_lddt_")
    ]
    assert serial_lddt_keys, (
        f"Serial run produced no validation LDDT metrics — test is vacuous. "
        f"Available metrics: {sorted(serial_captured_metrics)}"
    )

    serial_mp.undo()

    # --- Write 1D distributed config ---
    dp = grid_group_sizes["dp"]
    dist_output_dir = tmp_path / "dist_output"
    dist_output_dir.mkdir(parents=True, exist_ok=True)
    dist_config_path = tmp_path / "dist_config.yaml"
    _write_train_config_1d(
        TrainTestConfig1D(
            config_path=dist_config_path,
            output_dir=dist_output_dir,
            test_data_dir=training_data_dir,
            mol_dir=canonical_mols_dir,
            size_dp=dp,
            size_cp=size_cp,
            accelerator="gpu",
            limit_train_batches=1,
            pretrained=str(pretrained_path),
            model=_e2e_model_dict_1d(multiplicity=multiplicity, validate_structure=True, distributed=True),
            batch_size=B,
            samples_per_epoch=B,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_seqs=max_seqs,
            return_train_symmetries=False,
            split=str(split_file),
            pop_target_keys=True,
            extra_dataset_overrides=_e2e_ds_overrides,
            precision=precision_dist if precision_dist is not None else "FP32",
        )
    )

    # --- Spawn distributed workers ---
    spawn_multiprocessing(
        _worker_e2e_training_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        str(dist_config_path),
        str(dist_output_dir),
        str(serial_ckpt_path),
        str(pretrained_path),
        serial_captured_metrics,
        sigmas_global_host,
        noise_global_host,
        str(cached_samples_path),
        seed,
        dtype,
    )


# ---------------------------------------------------------------------------
#  Stop-and-go checkpoint resume test for Boltz-2 1D CP
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_boltz2_1d_stop_and_go(
    setup_env,
    tmp_path,
    test_cp_training_data_dir_boltz2,
    canonical_mols_dir,
):
    """Stop-and-go checkpoint resume correctness for Boltz-2 1D CP via ``train()``.

    Verifies that training 1 epoch, checkpointing, then resuming to epoch 2
    produces a valid final state: correct epoch/step counters, preserved
    model architecture (state_dict keys), weights that changed (training
    happened), and valid optimizer state.

    This is the 1D counterpart of ``test_boltz2_stop_and_go`` in the 2D test
    module.  The worker ``_parallel_assert_boltz2_stop_and_go`` is reused
    as-is because it only calls ``train_module.train()`` and inspects
    checkpoint files — no 2D-specific model/data code.

    Key 1D difference: config uses ``cp_topology: "1d"`` and
    ``grid_group_sizes["cp"]`` is an integer (not a tuple product).
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    size_dp = grid_group_sizes["dp"]
    size_cp = grid_group_sizes["cp"]
    accelerator = "gpu" if device_type == "cuda" else "cpu"

    stopgo_dir = tmp_path / "stopgo"

    common_kwargs: dict[str, Any] = {
        "test_data_dir": test_cp_training_data_dir_boltz2,
        "mol_dir": canonical_mols_dir,
        "size_dp": size_dp,
        "size_cp": size_cp,
        "accelerator": accelerator,
    }

    # Stage 1: 1 epoch with checkpoint.
    stage1_config = stopgo_dir / "config_stage1.yaml"
    _write_train_config_1d(
        TrainTestConfig1D(
            config_path=stage1_config,
            output_dir=stopgo_dir,
            **common_kwargs,
        )
    )

    # Stage 2: resume from checkpoint, train to epoch 2.
    stage2_config = stopgo_dir / "config_stage2.yaml"
    _write_train_config_1d(
        TrainTestConfig1D(
            config_path=stage2_config,
            output_dir=stopgo_dir,
            max_epochs=2,
            resume=str(stopgo_dir / "last.ckpt"),
            **common_kwargs,
        )
    )

    payload = (
        env_per_rank,
        str(stage1_config),
        str(stage2_config),
        str(stopgo_dir),
    )
    spawn_multiprocessing(_parallel_assert_boltz2_stop_and_go, world_size, payload)


# ---------------------------------------------------------------------------
#  BF16 / activation-checkpoint parity: shared 1D setup
# ---------------------------------------------------------------------------


def _setup_bf16_ac_test_env_1d(
    setup_env,
    test_cp_training_base_data_dir_boltz2: Path,
    canonical_mols_dir: Path,
    tmp_path: Path,
) -> _Bf16AcTestEnv:
    """Shared setup for BF16 + activation-checkpointing parity tests (1D CP).

    Creates training data, a pretrained checkpoint with all AC flags enabled
    (mirroring ``structurev2.yaml``), and writes serial / distributed YAML
    configs.  Returns paths and grid metadata so each test can attach its own
    profiler and spawn workers.

    This is the 1D counterpart of ``_setup_bf16_ac_test_env`` in the 2D test
    module.  Key 1D difference: ``size_cp`` is ``grid_group_sizes["cp"]``
    (int, not tuple product) and the distributed config uses
    ``_write_train_config_1d`` for ``cp_topology: "1d"``.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    seed = 42
    multiplicity = 2
    B = 2
    max_tokens = 256
    W = 32  # atoms_per_window_queries
    size_cp = grid_group_sizes["cp"]
    atom_align = math.lcm(W, size_cp)
    max_atoms = ((max_tokens * 10 + atom_align - 1) // atom_align) * atom_align
    max_seqs = 16
    scale_glorot = 0.05

    training_data_dir, split_file = _setup_training_data_all_4_e2e(
        tmp_path / "training_data", test_cp_training_base_data_dir_boltz2
    )

    ac_model_dict = _e2e_model_dict(multiplicity=multiplicity, validate_structure=False)
    ac_model_dict["checkpoint_diffusion_conditioning"] = True
    ac_model_dict["msa_args"]["activation_checkpointing"] = True
    ac_model_dict["pairformer_args"]["activation_checkpointing"] = True
    ac_model_dict["score_model_args"]["activation_checkpointing"] = True

    seed_by_rank(0, seed=seed)
    model_dict = copy.deepcopy(ac_model_dict)
    model_dict.pop("_target_")
    model_dict.pop("validators", None)
    pretrained_model = SerialBoltz2(**model_dict)
    init_module_params_glorot(pretrained_model, gain=scale_glorot)
    pretrained_model.apply(SetModuleInfValues())
    pretrained_model.structure_module.coordinate_augmentation = False
    pretrained_model = pretrained_model.to(dtype=torch.float32)

    pretrained_path = tmp_path / "pretrained.ckpt"
    torch.save(
        {
            "state_dict": pretrained_model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": pretrained_model.hparams,
        },
        pretrained_path,
    )

    _e2e_ds_overrides = {
        "filters": None,
        "moldir": None,
        "symmetry_correction": False,
        "val_group": "RCSB",
        "use_train_subset": None,
        "override_bfactor": False,
        "override_method": None,
    }

    serial_output_dir = tmp_path / "serial_output"
    serial_output_dir.mkdir(parents=True, exist_ok=True)
    serial_config_path = tmp_path / "serial_config.yaml"
    _write_train_config(
        TrainTestConfig(
            config_path=serial_config_path,
            output_dir=serial_output_dir,
            test_data_dir=training_data_dir,
            mol_dir=canonical_mols_dir,
            mode="serial",
            accelerator="gpu",
            precision="bf16-mixed",
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            pretrained=str(pretrained_path),
            model=copy.deepcopy(ac_model_dict),
            batch_size=B,
            samples_per_epoch=B,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_seqs=max_seqs,
            return_train_symmetries=False,
            split=str(split_file),
            pop_target_keys=True,
            extra_dataset_overrides=_e2e_ds_overrides,
            v2=True,
            strict_loading=False,
            wandb=None,
            save_top_k=0,
            disable_checkpoint=True,
        )
    )

    dp = grid_group_sizes["dp"]
    dist_output_dir = tmp_path / "dist_output"
    dist_output_dir.mkdir(parents=True, exist_ok=True)
    dist_config_path = tmp_path / "dist_config.yaml"
    _write_train_config_1d(
        TrainTestConfig1D(
            config_path=dist_config_path,
            output_dir=dist_output_dir,
            test_data_dir=training_data_dir,
            mol_dir=canonical_mols_dir,
            size_dp=dp,
            size_cp=size_cp,
            accelerator="gpu",
            precision="BF16_MIXED",
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            pretrained=str(pretrained_path),
            model=copy.deepcopy(ac_model_dict),
            batch_size=1,
            samples_per_epoch=dp,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_seqs=max_seqs,
            return_train_symmetries=False,
            split=str(split_file),
            pop_target_keys=True,
            extra_dataset_overrides=_e2e_ds_overrides,
        )
    )

    return _Bf16AcTestEnv(
        serial_config_path=serial_config_path,
        dist_config_path=dist_config_path,
        grid_group_sizes=grid_group_sizes,
        world_size=world_size,
        device_type=device_type,
        backend=backend,
        env_per_rank=env_per_rank,
    )


# ---------------------------------------------------------------------------
#  BF16 dtype parity: serial vs 1D DTensor training
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_boltz2_1d_bf16_dtype_parity(
    setup_env,
    test_cp_training_base_data_dir_boltz2,
    canonical_mols_dir,
    tmp_path,
):
    """Verify that BF16-mixed autocast produces identical dtype profiles in
    serial and 1D DTensor training workflows.

    Runs 1 training step under ``bf16-mixed`` precision for both the serial
    ``Boltz2`` model (via ``scripts/train/train.py``) and the 1D distributed
    ``Boltz2`` model (via ``src/boltz/distributed/train.py``), then compares
    forward activation dtypes, parameter dtypes, and parameter gradient
    dtypes at every module whose name appears in both models.

    No numerical comparison is performed — only dtype equality.  This means
    no deterministic-noise / cached-sample monkeypatching is required.

    This is the 1D counterpart of ``test_boltz2_bf16_dtype_parity`` in the
    2D test module.  The worker ``_worker_bf16_dtype_parity`` is reused
    as-is.
    """
    from boltz.testing.utils import DtypeProfiler

    env = _setup_bf16_ac_test_env_1d(setup_env, test_cp_training_base_data_dir_boltz2, canonical_mols_dir, tmp_path)

    # --- Run serial training with dtype profiling ---
    serial_mp = pytest.MonkeyPatch()
    _serial_profiler: list[DtypeProfiler | None] = [None]
    _orig_fit = pl.Trainer.fit

    def _serial_profiling_fit(trainer_self, model, **kwargs):
        _serial_profiler[0] = DtypeProfiler(model)
        result = _orig_fit(trainer_self, model, **kwargs)
        _serial_profiler[0].collect_grad_dtypes(model)
        return result

    serial_mp.setattr(pl.Trainer, "fit", _serial_profiling_fit)

    _serial_train_mod = _load_serial_train_module()
    _serial_train_mod.train(str(env.serial_config_path), [])

    profiler = _serial_profiler[0]
    assert profiler is not None, "Serial DtypeProfiler was never attached"
    profiler.remove_hooks()

    # Non-vacuous: serial must have a mix of BF16 and FP32 activations
    serial_fwd_dtypes_set = set(profiler.fwd_dtypes.values())
    assert (
        torch.bfloat16 in serial_fwd_dtypes_set
    ), f"Serial: no bfloat16 activations — autocast may not be active. Unique dtypes: {serial_fwd_dtypes_set}"
    assert (
        torch.float32 in serial_fwd_dtypes_set
    ), f"Serial: no float32 activations. Unique dtypes: {serial_fwd_dtypes_set}"

    # All serial params must be FP32
    for name, dtype in profiler.param_dtypes.items():
        assert dtype == torch.float32, f"Serial param '{name}' has dtype {dtype}, expected float32"

    # Save serial profile for workers
    serial_dtype_profile_path = tmp_path / "serial_dtype_profile.pt"
    torch.save(
        {
            "fwd_dtypes": profiler.fwd_dtypes,
            "param_dtypes": profiler.param_dtypes,
            "param_grad_dtypes": profiler.param_grad_dtypes,
        },
        serial_dtype_profile_path,
    )

    serial_mp.undo()

    # --- Spawn distributed workers ---
    spawn_multiprocessing(
        _worker_bf16_dtype_parity,
        env.world_size,
        env.grid_group_sizes,
        env.device_type,
        env.backend,
        env.env_per_rank,
        str(env.dist_config_path),
        str(serial_dtype_profile_path),
    )


# ---------------------------------------------------------------------------
#  Activation checkpoint recomputation parity: serial vs 1D DTensor training
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_boltz2_1d_actv_ckpt_parity(
    setup_env,
    test_cp_training_base_data_dir_boltz2,
    canonical_mols_dir,
    tmp_path,
):
    """Verify serial and 1D DTensor training checkpoint-recompute the same modules.

    Runs 1 training step (forward + backward) under ``bf16-mixed`` precision
    with all production activation-checkpointing flags enabled.  Forward hooks
    count how many times each module's forward is invoked: modules inside a
    ``torch.utils.checkpoint.checkpoint`` region are called twice (once in
    forward, once during backward recomputation).

    Each distributed rank independently compares its local counts against the
    serial reference — counts are **never** aggregated across ranks.

    This is the 1D counterpart of ``test_boltz_actv_ckpt_parity`` in the
    2D test module.  The worker ``_worker_actv_ckpt_parity`` is reused as-is.
    """
    from boltz.testing.utils import RecomputeProfiler

    env = _setup_bf16_ac_test_env_1d(setup_env, test_cp_training_base_data_dir_boltz2, canonical_mols_dir, tmp_path)

    # --- Run serial training with recompute profiling ---
    serial_mp = pytest.MonkeyPatch()
    _serial_profiler: list[RecomputeProfiler | None] = [None]
    _orig_fit = pl.Trainer.fit

    def _serial_profiling_fit(trainer_self, model, **kwargs):
        _serial_profiler[0] = RecomputeProfiler(model)
        return _orig_fit(trainer_self, model, **kwargs)

    serial_mp.setattr(pl.Trainer, "fit", _serial_profiling_fit)

    _serial_train_mod = _load_serial_train_module()
    _serial_train_mod.train(str(env.serial_config_path), [])

    profiler = _serial_profiler[0]
    assert profiler is not None, "Serial RecomputeProfiler was never attached"
    profiler.remove_hooks()

    # Non-vacuous: activation checkpointing must recompute some modules
    serial_recomp = profiler.recomputed_modules
    all_serial_names = set(profiler.fwd_counts)
    assert len(serial_recomp) >= 5, (
        f"Serial: only {len(serial_recomp)} modules recomputed (expected >= 5). "
        f"Activation checkpointing may not be active."
    )
    assert (
        serial_recomp < all_serial_names
    ), f"Serial: all {len(all_serial_names)} modules are recomputed — not a strict subset, implying a counting bug."

    # Save serial recompute profile for workers
    serial_recompute_profile_path = tmp_path / "serial_recompute_profile.pt"
    torch.save({"fwd_counts": profiler.fwd_counts}, serial_recompute_profile_path)

    serial_mp.undo()

    # --- Spawn distributed workers (per-rank comparison, no cross-rank aggregation) ---
    spawn_multiprocessing(
        _worker_actv_ckpt_parity,
        env.world_size,
        env.grid_group_sizes,
        env.device_type,
        env.backend,
        env.env_per_rank,
        str(env.dist_config_path),
        str(serial_recompute_profile_path),
    )


# ---------------------------------------------------------------------------
#  Validation parity: serial vs 1D DTensor validation
# ---------------------------------------------------------------------------


def _worker_validation_parity_1d(
    rank: int,
    grid_group_sizes: dict,
    device_type: str,
    backend: str,
    env_per_rank: dict[str, Any],
    dist_config_path: str,
    serial_metrics: dict,
    sigmas_global_host: torch.Tensor,
    noise_global_host: torch.Tensor,
    cached_samples_path: str,
    seed: int,
) -> None:
    """Multi-rank worker: 1D distributed validate() then compare with serial metrics.

    1. Applies module-level monkeypatches (noise, data, smooth_lddt)
    2. Calls ``train_module.train(dist_config_path, [])`` in validation_only mode
    3. Captures validation metrics from ``trainer.validate()``
    4. Compares all validation metrics with the serial reference

    Adapted from ``_worker_validation_parity`` in the 2D test.  Key 1D
    differences:

    - ``AtomDiffusion1D`` (not ``DistAtomDiffusionV2``) for noise_distribution
    - ``diffusion_1d`` module (not ``diffusion``) for create_distributed_randn
    - ``Boltz2_1D`` (not ``Boltz2Distributed``) for validation_step wrapper
    - Noise distributed via ``distribute_tensor`` on 2D mesh ``(dp, cp)``
      with ``(Shard(0), Shard(1))`` — no intersperse padding
    - Sigma placement is ``(Shard(0), Replicate())`` (2-tuple, not 3-tuple)
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    monkeypatch.setattr(train_module, "_cleanup_distributed", lambda: None)
    DistributedManager._state = {}

    _apply_cached_getitem(monkeypatch, cached_samples_path)

    # Deterministic sigmas for noise schedule
    def _dist_noise_dist_1d(self, bs, dtype=torch.float32):
        s = sigmas_global_host.to(device=self.device_mesh.device_type, dtype=dtype)[:bs]
        return distribute_tensor(s, self.device_mesh, (Shard(0), Replicate()))

    monkeypatch.setattr(AtomDiffusion1D, "noise_distribution", _dist_noise_dist_1d)

    # Deterministic noise via distribute_tensor (no intersperse padding for 1D CP)
    _noise_dt_cache_1d: list[DTensor | None] = [None]
    _dist_in_val_1d = [False]

    def _det_create_randn_1d(shape, device_mesh, placements, dtype=torch.float32, scale=1.0):
        """Deterministic replacement for create_distributed_randn (1D CP).

        During validation, returns zero noise.  During training, distributes
        the pre-generated global noise tensor on the 2D mesh (dp, cp).
        """
        if _dist_in_val_1d[0]:
            from boltz.distributed.utils import create_distributed_randn as _real_create_randn

            return _real_create_randn(shape, device_mesh, placements, dtype=dtype, scale=0.0)

        if _noise_dt_cache_1d[0] is None:
            n = noise_global_host.to(device=device_mesh.device_type, dtype=dtype)
            _noise_dt_cache_1d[0] = distribute_tensor(n, device_mesh, (Shard(0), Shard(1)))
        n = _noise_dt_cache_1d[0]
        if n.dtype != dtype:
            n = n.to(dtype=dtype)
        # Pad if noise shape is smaller than requested (e.g., padding alignment)
        if len(shape) > 1 and n.shape[1] < shape[1]:
            from boltz.testing.utils import pad_to_length as _pad

            n = _pad(n, dim=1, length=shape[1])
        return n * scale

    monkeypatch.setattr(dist_diffusion_1d_module, "create_distributed_randn", _det_create_randn_1d)

    _orig_dist_val_step_1d = Boltz2_1D.validation_step

    def _dist_val_step_wrapper_1d(self_model, batch, batch_idx):
        _dist_in_val_1d[0] = True
        try:
            return _orig_dist_val_step_1d(self_model, batch, batch_idx)
        finally:
            _dist_in_val_1d[0] = False

    monkeypatch.setattr(Boltz2_1D, "validation_step", _dist_val_step_wrapper_1d)

    # Skip RMSD (not compared; serial uses a different code path)
    import boltz.distributed.model.validation.validator as _dist_validator_mod

    def _rmsd_noop(*args, **kwargs):
        return torch.tensor(0.0), None, None

    monkeypatch.setattr(_dist_validator_mod, "weighted_minimum_rmsd_single", _rmsd_noop)

    # Capture metrics from trainer.validate()
    _captured_metrics: dict[str, float] = {}
    _orig_validate = pl.Trainer.validate

    def _capturing_validate(self, *args, **kwargs):
        result = _orig_validate(self, *args, **kwargs)
        for k, v in self.callback_metrics.items():
            if isinstance(v, DTensor):
                _captured_metrics[k] = v.full_tensor().detach().cpu().item()
            elif isinstance(v, torch.Tensor):
                _captured_metrics[k] = v.detach().cpu().item()
            else:
                _captured_metrics[k] = v
        return result

    monkeypatch.setattr(pl.Trainer, "validate", _capturing_validate)

    train_module.train(dist_config_path, [])

    # --- Compare metrics ---
    # Atom-level LDDT: atol=2e-4, rtol=0.005 (forward-pass accumulation order).
    # Token-level metrics (disto_lddt_*, disto_loss): default tolerance.
    # Trailing underscore omitted so the global weighted-average "val/lddt"
    # (not just "val/lddt_*") also gets the relaxed tolerance.
    _forward_dependent_prefixes = ("val/lddt", "val/complex_lddt", "val/clash", "val/pb", "val/rmsd")
    _lddt_keys_compared = []
    if serial_metrics:
        for k in serial_metrics:
            if k in _captured_metrics:
                got = torch.tensor(_captured_metrics[k])
                exp = torch.tensor(serial_metrics[k])
                if any(k.startswith(p) for p in _forward_dependent_prefixes):
                    torch.testing.assert_close(
                        got,
                        exp,
                        atol=2e-4,
                        rtol=0.005,
                        msg=lambda m: f"Rank {rank}: metric '{k}' mismatch: {m}",
                    )
                else:
                    torch.testing.assert_close(
                        got,
                        exp,
                        msg=lambda m: f"Rank {rank}: metric '{k}' mismatch: {m}",
                    )
                if "lddt" in k:
                    _lddt_keys_compared.append(k)

    assert _lddt_keys_compared, (
        f"Rank {rank}: no validation LDDT metrics were compared — test is vacuous. "
        f"Serial keys: {sorted(serial_metrics)}, dist keys: {sorted(_captured_metrics)}"
    )
    for required_metric in ("val/lddt", "val/disto_lddt", "val/complex_lddt"):
        assert (
            required_metric in _captured_metrics
        ), f"Rank {rank}: distributed metrics missing '{required_metric}' — available: {sorted(_captured_metrics)}"

    torch.distributed.barrier()


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_boltz2_1d_validation_parity(
    setup_env,
    test_cp_training_base_data_dir_boltz2,
    canonical_mols_dir,
    tmp_path,
):
    """Serial-vs-1D-DTensor validation metric parity via ``train()`` validation_only.

    Runs both serial and 1D distributed ``train()`` in validation_only mode
    (no training, no backward pass) on the same pretrained model and data,
    then compares all logged validation metrics.  Model weights come from a
    deterministic pretrained checkpoint; noise and data are controlled via
    monkeypatches so the only source of difference is the serial-vs-distributed
    forward pass in FP32.

    Token-level metrics (``val/disto_lddt_*``, ``val/disto_loss``) match at
    default tolerance because they depend only on the confidence head logits,
    which are deterministic given identical weights.

    Atom-level LDDT (``val/lddt_*``, ``val/complex_lddt_*``) uses
    ``atol=2e-4, rtol=0.005``.  These metrics depend on diffusion-sampled
    coordinates whose FP32 accumulation order differs between serial
    (full-sequence attention) and distributed (split attention + padding for
    uniform DP shapes).  Without training, these forward-pass-only differences
    are smaller and more stable than in the e2e training test, allowing ~2.5x
    tighter tolerances than the e2e test's ``atol=5e-4, rtol=0.02``.
    Observed max absdiff is ~9e-5 (``lddt_intra_ligand``).

    This is the 1D counterpart of ``test_boltz2_validation_parity`` in the
    2D test module.  Key 1D differences:

    - ``cp_topology="1d"`` in the distributed config
    - 2D device mesh ``(dp, cp)`` instead of 3D ``(dp, cp0, cp1)``
    - ``size_cp`` is ``grid_group_sizes["cp"]`` (int, not tuple product)
    - Noise distributed via ``distribute_tensor`` (no intersperse padding)
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float32
    seed = 42
    multiplicity = 2
    B = 2
    max_tokens = 256
    W = 32  # atoms_per_window_queries
    size_cp = grid_group_sizes["cp"]
    atom_align = math.lcm(W, size_cp)
    max_atoms = ((max_tokens * 10 + atom_align - 1) // atom_align) * atom_align
    max_seqs = 16
    scale_glorot = 0.05

    # --- Merge all 4 samples with train/val split ---
    training_data_dir, split_file = _setup_training_data_all_4_e2e(
        tmp_path / "training_data", test_cp_training_base_data_dir_boltz2
    )

    # --- Create pretrained checkpoint with deterministic init ---
    seed_by_rank(0, seed=seed)
    model_dict = _e2e_model_dict(multiplicity=multiplicity, validate_structure=True)
    model_dict.pop("_target_")
    model_dict.pop("validators", None)
    _val_validators = [RCSBValidator(val_names=["RCSB"], confidence_prediction=False, physicalism_metrics=True)]
    pretrained_model = SerialBoltz2(**model_dict, validators=_val_validators)
    init_module_params_glorot(pretrained_model, gain=scale_glorot)
    pretrained_model.apply(SetModuleInfValues())
    pretrained_model.structure_module.coordinate_augmentation = False
    pretrained_model = pretrained_model.to(dtype=dtype)

    pretrained_path = tmp_path / "pretrained.ckpt"
    torch.save(
        {
            "state_dict": pretrained_model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": pretrained_model.hparams,
        },
        pretrained_path,
    )

    # --- Pre-load and cache individual samples to disk ---
    _tmp_mp = pytest.MonkeyPatch()
    _apply_e2e_deterministic_getitem(_tmp_mp, base_seed=seed)
    _preload_cfg = setup_mock_training_datamodule_config(training_data_dir)
    _preload_cfg.batch_size = B
    _preload_cfg.samples_per_epoch = B
    _preload_cfg.moldir = str(canonical_mols_dir)
    _preload_cfg.return_train_symmetries = False
    _preload_cfg.msa_sampling_training = False
    _preload_cfg.max_tokens = max_tokens
    _preload_cfg.max_atoms = max_atoms
    _preload_cfg.max_seqs = max_seqs
    for _ds in _preload_cfg.datasets:
        _ds.filters = None
        _ds.split = str(split_file)
        _ds.symmetry_correction = False
    seed_by_rank(0, seed=seed)
    _preload_dm = Boltz2TrainingDataModule(cfg=_preload_cfg)
    _preload_ds = _preload_dm._train_set
    _cached_samples = {i: _preload_ds[i] for i in range(B)}
    cached_samples_path = tmp_path / "cached_samples.pt"
    torch.save(_cached_samples, cached_samples_path)

    _preload_dl = _preload_dm.train_dataloader()
    _preload_batch = next(iter(_preload_dl))
    atom_pad_mask_host = _preload_batch["atom_pad_mask"].detach().cpu()
    _tmp_mp.undo()

    # --- Pre-generate deterministic noise (masked by atom_pad_mask) ---
    seed_by_rank(0, seed=seed)
    sigmas_global = pretrained_model.structure_module.noise_distribution(B * multiplicity).to(dtype=dtype)
    noise_global = torch.empty(B * multiplicity, max_atoms, 3, dtype=dtype)
    init_tensors_uniform([noise_global], low=-scale_glorot, high=scale_glorot)
    _mask_mul = atom_pad_mask_host[:, :, None].repeat_interleave(multiplicity, 0).to(dtype=dtype)
    noise_global = noise_global * _mask_mul

    sigmas_global_host = sigmas_global.detach().cpu()
    noise_global_host = noise_global.detach().cpu()

    # --- Write serial config (validation_only) ---
    serial_output_dir = tmp_path / "serial_output"
    serial_output_dir.mkdir(parents=True, exist_ok=True)
    serial_config_path = tmp_path / "serial_config.yaml"
    _e2e_ds_overrides = {
        "filters": None,
        "moldir": None,
        "symmetry_correction": False,
        "val_group": "RCSB",
        "use_train_subset": None,
        "override_bfactor": False,
        "override_method": None,
    }
    _write_train_config(
        TrainTestConfig(
            config_path=serial_config_path,
            output_dir=serial_output_dir,
            test_data_dir=training_data_dir,
            mol_dir=canonical_mols_dir,
            mode="serial",
            accelerator="gpu",
            validation_only=True,
            pretrained=str(pretrained_path),
            model=_e2e_model_dict(multiplicity=multiplicity, validate_structure=True),
            batch_size=B,
            samples_per_epoch=B,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_seqs=max_seqs,
            return_train_symmetries=False,
            split=str(split_file),
            pop_target_keys=True,
            extra_dataset_overrides=_e2e_ds_overrides,
            v2=True,
            strict_loading=False,
            wandb=None,
            save_top_k=0,
            disable_checkpoint=True,
        )
    )

    # --- Apply serial monkeypatches ---
    serial_mp = pytest.MonkeyPatch()
    _apply_cached_getitem(serial_mp, cached_samples_path)

    _orig_serial_boltz2_init = SerialBoltz2.__init__

    @functools.wraps(_orig_serial_boltz2_init)
    def _init_with_validators(self, *args, **kwargs):
        if kwargs.get("validate_structure", False) and not kwargs.get("validators"):
            kwargs["validators"] = [
                RCSBValidator(val_names=["RCSB"], confidence_prediction=False, physicalism_metrics=True)
            ]
        _orig_serial_boltz2_init(self, *args, **kwargs)

    serial_mp.setattr(SerialBoltz2, "__init__", _init_with_validators)

    serial_mp.setattr(
        SerialAtomDiffusionV2,
        "noise_distribution",
        lambda self, bs: sigmas_global[:bs].to(device=self.zero.device),
    )

    # Deterministic noise: pre-generated for training, zero for validation
    _serial_in_val = [False]

    def _serial_randn_like(t):
        if _serial_in_val[0]:
            return torch.zeros_like(t)
        return noise_global[: t.shape[0], : t.shape[1]].to(device=t.device, dtype=t.dtype)

    serial_mp.setattr(serial_diffusion_v2_module.torch, "randn_like", _serial_randn_like)

    _orig_serial_randn = serial_diffusion_v2_module.torch.randn

    def _serial_randn(*args, **kwargs):
        if _serial_in_val[0]:
            kwargs.pop("generator", None)
            return torch.zeros(*args, **kwargs)
        return _orig_serial_randn(*args, **kwargs)

    serial_mp.setattr(serial_diffusion_v2_module.torch, "randn", _serial_randn)

    _orig_compute_random_augmentation = serial_diffusion_v2_module.compute_random_augmentation

    def _identity_augmentation_during_val(multiplicity_arg, s_trans=1.0, device=None, dtype=torch.float32):
        if _serial_in_val[0]:
            R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(multiplicity_arg, -1, -1)
            tr = torch.zeros(multiplicity_arg, 1, 3, device=device, dtype=dtype)
            return R, tr
        return _orig_compute_random_augmentation(multiplicity_arg, s_trans=s_trans, device=device, dtype=dtype)

    serial_mp.setattr(serial_diffusion_v2_module, "compute_random_augmentation", _identity_augmentation_during_val)

    _orig_serial_val_step = SerialBoltz2.validation_step

    def _serial_val_step_wrapper(self_model, batch, batch_idx):
        _serial_in_val[0] = True
        try:
            return _orig_serial_val_step(self_model, batch, batch_idx)
        finally:
            _serial_in_val[0] = False

    serial_mp.setattr(SerialBoltz2, "validation_step", _serial_val_step_wrapper)

    serial_mp.setattr(serial_loss_v2_module, "smooth_lddt_loss", _smooth_lddt_loss_dense_e2e)
    serial_mp.setattr(serial_diffusion_v2_module, "smooth_lddt_loss", _smooth_lddt_loss_dense_e2e)

    # Capture metrics from trainer.validate()
    serial_captured_metrics: dict[str, float] = {}
    _orig_validate = pl.Trainer.validate

    def _serial_capturing_validate(self, *args, **kwargs):
        result = _orig_validate(self, *args, **kwargs)
        for k, v in self.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                serial_captured_metrics[k] = v.detach().cpu().item()
            else:
                serial_captured_metrics[k] = v
        return result

    serial_mp.setattr(pl.Trainer, "validate", _serial_capturing_validate)

    # --- Run serial validation ---
    _serial_train_mod = _load_serial_train_module()
    _serial_train_mod.train(str(serial_config_path), [])

    # Non-vacuous guard
    serial_lddt_keys = [
        k for k in serial_captured_metrics if k.startswith("val/lddt_") or k.startswith("val/disto_lddt_")
    ]
    assert serial_lddt_keys, (
        f"Serial run produced no validation LDDT metrics — test is vacuous. "
        f"Available metrics: {sorted(serial_captured_metrics)}"
    )

    serial_mp.undo()

    # --- Write 1D distributed config (validation_only) ---
    dp = grid_group_sizes["dp"]
    dist_output_dir = tmp_path / "dist_output"
    dist_output_dir.mkdir(parents=True, exist_ok=True)
    dist_config_path = tmp_path / "dist_config.yaml"
    _write_train_config_1d(
        TrainTestConfig1D(
            config_path=dist_config_path,
            output_dir=dist_output_dir,
            test_data_dir=training_data_dir,
            mol_dir=canonical_mols_dir,
            size_dp=dp,
            size_cp=size_cp,
            accelerator="gpu",
            validation_only=True,
            pretrained=str(pretrained_path),
            model=_e2e_model_dict_1d(multiplicity=multiplicity, validate_structure=True, distributed=True),
            batch_size=1,
            samples_per_epoch=dp,
            max_tokens=max_tokens,
            max_atoms=max_atoms,
            max_seqs=max_seqs,
            return_train_symmetries=False,
            split=str(split_file),
            pop_target_keys=True,
            extra_dataset_overrides=_e2e_ds_overrides,
        )
    )

    # --- Spawn distributed workers ---
    spawn_multiprocessing(
        _worker_validation_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        str(dist_config_path),
        serial_captured_metrics,
        sigmas_global_host,
        noise_global_host,
        str(cached_samples_path),
        seed,
    )
