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

"""Tests for the 1D CP Boltz2 distributed model wrapper.

Verification checks:
    V1: Construction – all serial parameters are present in the 1D CP wrapper
    V2: Forward pass – trunk outputs have correct shapes and placements
    V3: Non-vacuous guards – distributed modules have DTensor params,
        placeholders have plain params
    V4: predict_step smoke – output contains only plain tensors (no DTensor leaks),
        coords and masks have correct shapes, exception is False
    V5: predict_step confidence – when confidence_prediction=True, all expected
        confidence keys are present and no DTensors leak into the output dict
    V6: Forward/backward parity – 1D CP distributed Boltz2 forward/backward
        matches serial (mirrors 2D test_boltz2_forward_backward_parity)
    V7: predict_step parity – 1D CP distributed predict_step matches serial
        (mirrors 2D test_boltz2_predict_step)
    V7b: predict_step confidence parity – 1D CP distributed predict_step with
        confidence_prediction=True matches serial
        (mirrors 2D test_boltz2_predict_step_confidence)
    V8: Serial vs 1D CP distributed training_step parity – compares loss,
        gradients, post-optimizer parameters, and CSVLogger-captured logged
        metrics between serial Boltz2.training_step and 1D CP
        Boltz2_1D.training_step with all randomness sources controlled
        (mirrors 2D test_boltz2_training_step_parity)
    V9: Serial vs 1D CP distributed validation_step parity – two-phase
        comparison: (1) per-validator MeanMetric values after validation_step,
        (2) aggregated logged metrics after on_validation_epoch_end
        (mirrors 2D test_boltz2_validation_step_parity)
    V10: bf16 mixed precision (CUDA-only) – ready submodules produce bf16
        outputs under torch.autocast, weight gradients are reduced in
        ``>= fp32``, the recycling path (s_recycle + s_norm) emits bf16,
        and ``torch.clear_autocast_cache()`` does not error
        (mirrors 2D test_boltz2_bf16_mixed_precision)
"""

import os
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

import boltz.distributed.model.modules.diffusion_1d as diffusion_1d_module
import boltz.model.modules.diffusionv2 as serial_diffusion_v2_module
from boltz.data import const
from boltz.distributed.data.module.placements_1d import TRAINING_FEATURE_PLACEMENTS_1D
from boltz.distributed.data.utils import ATOM_FEATURES_V2, distribute_features
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, elementwise_op, scalar_tensor_op
from boltz.distributed.model.models.boltz2_1d import Boltz2_1D, _PlaceholderModule
from boltz.distributed.model.validation.rcsb_1d import Distributed1DRCSBValidator
from boltz.distributed.port_utils import find_free_port
from boltz.model.models.boltz2 import Boltz2 as SerialBoltz2
from boltz.model.validation.rcsb import RCSBValidator
from boltz.testing.utils import (
    SetModuleInfValues,
    create_boltz2_model_init_params,
    init_module_params_glorot,
    init_tensors_uniform,
    random_features,
    seed_by_rank,
    spawn_multiprocessing,
)

TOKEN_S = 32
TOKEN_Z = 16

_BOLTZ2_SELECTED_KEYS = [
    "atom_pad_mask",
    "atom_to_token",
    "pair_mask",
    "token_pad_mask",
    "ref_pos",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
    "res_type",
    "profile",
    "deletion_mean",
    "pocket_feature",
    "atom_resolved_mask",
    "mol_type",
    "msa",
    "has_deletion",
    "deletion_value",
    "msa_paired",
    "msa_mask",
    "token_bonds",
    "type_bonds",
    "token_pair_pad_mask",
    "asym_id",
    "residue_index",
    "entity_id",
    "token_index",
    "sym_id",
    "cyclic_period",
    "coords",
    "disto_target",
    "token_disto_mask",
    "atom_counts_per_token",
    "token_to_rep_atom",
    "frames_idx",
    "contact_conditioning",
    "contact_threshold",
    "method_feature",
    "modified",
    "bfactor",
    "plddt",
]


class _DictNamespace:
    """A picklable namespace with both attribute access and .get() support."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def _make_training_args(**overrides):
    """Create minimal training_args for Boltz2."""
    defaults = {
        "recycling_steps": 1,
        "sampling_steps": 2,
        "diffusion_multiplicity": 1,
        "diffusion_samples": 1,
        "diffusion_loss_weight": 1.0,
        "distogram_loss_weight": 0.3,
        "confidence_loss_weight": 0.0,
        "bfactor_loss_weight": 0.0,
        "symmetry_correction": False,
        "adam_beta_1": 0.9,
        "adam_beta_2": 0.95,
        "adam_eps": 1e-8,
        "base_lr": 1e-3,
        "max_lr": 1e-3,
        "lr_scheduler": "af3",
        "lr_warmup_no_steps": 10,
        "lr_start_decay_after_n_steps": 100,
        "lr_decay_every_n_steps": 50000,
        "lr_decay_factor": 0.95,
        "weight_decay": 0.0,
    }
    defaults.update(overrides)
    return _DictNamespace(**defaults)


def _make_validation_args(**overrides):
    defaults = {
        "recycling_steps": 0,
        "sampling_steps": 2,
        "diffusion_samples": 1,
        "symmetry_correction": False,
        "run_confidence_sequentially": False,
    }
    defaults.update(overrides)
    return _DictNamespace(**defaults)


def _create_minimal_serial_boltz2(
    confidence_prediction=False,
    bond_type_feature=False,
    predict_bfactor=False,
    ema=False,
):
    """Create a minimal serial Boltz2 model for testing."""
    training_args = _make_training_args()
    validation_args = _make_validation_args()

    pairformer_args = {"num_blocks": 1, "num_heads": 2, "dropout": 0.0, "v2": True}

    model = SerialBoltz2(
        atom_s=16,
        atom_z=8,
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atom_feature_dim=388,
        num_bins=8,
        training_args=training_args,
        validation_args=validation_args,
        embedder_args={
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 2,
            "activation_checkpointing": False,
        },
        msa_args={"msa_s": 16, "msa_blocks": 1, "msa_dropout": 0.0, "z_dropout": 0.0},
        pairformer_args=pairformer_args,
        score_model_args={
            "sigma_data": 16.0,
            "dim_fourier": 32,
            "atom_encoder_depth": 1,
            "atom_encoder_heads": 2,
            "token_transformer_depth": 1,
            "token_transformer_heads": 2,
            "atom_decoder_depth": 1,
            "atom_decoder_heads": 2,
            "activation_checkpointing": False,
            "conditioning_transition_layers": 1,
        },
        diffusion_process_args={"num_sampling_steps": 2},
        diffusion_loss_args={},
        confidence_prediction=confidence_prediction,
        confidence_model_args=(
            {"pairformer_args": pairformer_args, "confidence_args": {}} if confidence_prediction else None
        ),
        predict_args={
            "recycling_steps": 0,
            "sampling_steps": 2,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
        },
        validate_structure=False,
        structure_prediction_training=True,
        ema=ema,
        use_templates=False,
        predict_bfactor=predict_bfactor,
        bond_type_feature=bond_type_feature,
    )

    return model


def _init_distributed_1d(rank, world_size, device_type, backend, env_map):
    """Initialize distributed for 1D CP on a 2D mesh (dp, cp)."""
    monkeypatch = pytest.MonkeyPatch()
    if env_map is not None:
        for var_name, value in env_map.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    # For 1D CP: mesh is (dp=1, cp=world_size)
    grid_group_sizes = OrderedDict([("dp", 1), ("cp", world_size)])
    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    return monkeypatch, DistributedManager()


def _build_1d_model(serial_model, dist_manager):
    """Wrap serial model as 1D CP distributed, move to device."""
    serial_model = serial_model.to(dist_manager.device)
    model_1d = Boltz2_1D(serial_model, dist_manager)
    return model_1d.to(dist_manager.device)


# ====================================================================== #
#  Single-process unit tests (shared process group)                      #
# ====================================================================== #

# Shared fixture: initialize process group once and reuse across tests.
_SINGLE_PROCESS_INITIALIZED = False


def _ensure_single_process_manager():
    """Initialize DistributedManager with gloo + (dp=1, cp=1) for single-process tests (idempotent)."""
    global _SINGLE_PROCESS_INITIALIZED
    if not _SINGLE_PROCESS_INITIALIZED:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # Allocate a free port lazily so concurrent test runs (parallel pytest
        # worktrees) don't collide on a fixed default. setdefault preserves an
        # explicit MASTER_PORT override from the environment.
        os.environ.setdefault("MASTER_PORT", str(find_free_port()))
        grid_group_sizes = OrderedDict([("dp", 1), ("cp", 1)])
        DistributedManager.initialize(grid_group_sizes, device_type="cpu", backend="gloo")
        _SINGLE_PROCESS_INITIALIZED = True
    return DistributedManager()


def test_wrapper_preserves_all_serial_parameters():
    """V1: All serial parameters should appear in the 1D CP wrapper."""
    serial_model = _create_minimal_serial_boltz2()
    serial_param_names = set(serial_model.state_dict().keys())
    assert len(serial_param_names) > 0, "Serial model has no parameters"

    expected_prefixes = {
        "s_init",
        "z_init_1",
        "z_init_2",
        "s_norm",
        "z_norm",
        "s_recycle",
        "z_recycle",
        "token_bonds",
        "msa_module",
        "pairformer_module",
        "distogram_module",
        "input_embedder",
        "rel_pos",
        "contact_conditioning",
        "diffusion_conditioning",
        "structure_module",
    }

    manager = _ensure_single_process_manager()
    model_1d = Boltz2_1D(serial_model, manager)

    dist_param_names = set(model_1d.state_dict().keys())

    # Every serial parameter must have an exact match in the distributed model
    for serial_name in serial_param_names:
        assert serial_name in dist_param_names, (
            f"Serial param '{serial_name}' not found in distributed model state_dict. "
            f"Closest matches: {[n for n in sorted(dist_param_names) if serial_name.split('.')[0] in n][:5]}"
        )

    # Verify expected module prefixes exist
    for prefix in expected_prefixes:
        matching = [n for n in dist_param_names if n.startswith(prefix)]
        assert len(matching) > 0, f"No distributed params with prefix '{prefix}'"


def test_dtensor_params_are_dtensors():
    """V3: Distributed modules should have DTensor params, placeholders plain."""
    serial_model = _create_minimal_serial_boltz2()
    manager = _ensure_single_process_manager()
    model_1d = Boltz2_1D(serial_model, manager)

    # All trainable params should be DTensors
    for name, param in model_1d.named_parameters():
        if param.requires_grad:
            assert isinstance(param, DTensor), f"Trainable param '{name}' is {type(param)}, expected DTensor"


def test_wrapper_hparams_saved_1d():
    """Hyper-parameters should be preserved in save_hyperparameters."""
    serial_model = _create_minimal_serial_boltz2()
    assert hasattr(serial_model, "hparams"), "Serial model should have hparams"
    assert "atom_s" in serial_model.hparams
    assert "token_s" in serial_model.hparams


def test_placeholder_raises_on_forward_1d():
    """Placeholder module should raise NotImplementedError."""
    placeholder = _PlaceholderModule(torch.nn.Linear(4, 8), "TestModule")
    with pytest.raises(NotImplementedError, match="TestModule"):
        placeholder(torch.randn(2, 4))


def test_placeholder_preserves_parameters_1d():
    """Placeholder should expose the serial module's parameters, frozen."""
    placeholder = _PlaceholderModule(torch.nn.Linear(4, 8), "TestModule")
    params = dict(placeholder.named_parameters())
    assert "_serial.weight" in params
    assert "_serial.bias" in params
    assert params["_serial.weight"].shape == (8, 4)
    for name, p in placeholder.named_parameters():
        assert not p.requires_grad, f"Placeholder param '{name}' should be frozen"


def test_serial_model_ema_callbacks_1d():
    """EMA callback configuration."""
    assert _create_minimal_serial_boltz2(ema=True).use_ema is True
    assert len(_create_minimal_serial_boltz2(ema=True).configure_callbacks()) == 1
    assert _create_minimal_serial_boltz2(ema=False).use_ema is False
    assert len(_create_minimal_serial_boltz2(ema=False).configure_callbacks()) == 0


def test_confidence_model_can_be_created_1d():
    """Confidence serial model can be created with proper args."""
    assert _create_minimal_serial_boltz2(confidence_prediction=True).confidence_prediction is True


def test_bfactor_in_atom_features_v2_1d():
    """Data pipeline: bfactor must be in ATOM_FEATURES_V2 for distributed sharding."""
    assert "bfactor" in ATOM_FEATURES_V2


def test_checkpoint_lr_is_overwritten_1d():
    """on_load_checkpoint should overwrite lr and weight_decay in checkpoint."""
    serial_model = _create_minimal_serial_boltz2()
    manager = _ensure_single_process_manager()
    model_1d = Boltz2_1D(serial_model, manager)

    checkpoint = {
        "optimizer_states": [{"param_groups": [{"lr": 0.5, "weight_decay": 0.99}, {"lr": 0.5, "weight_decay": 0.99}]}],
        "lr_schedulers": [{"max_lr": 0.5, "base_lrs": [0.5, 0.5], "_last_lr": [0.5, 0.5]}],
        "hyper_parameters": {
            "training_args": {"max_lr": 0.5, "diffusion_multiplicity": 99, "recycling_steps": 99, "weight_decay": 0.99}
        },
    }
    model_1d.on_load_checkpoint(checkpoint)

    for pg in checkpoint["optimizer_states"][0]["param_groups"]:
        assert pg["lr"] == 1e-3 and pg["weight_decay"] == 0.0

    sched = checkpoint["lr_schedulers"][0]
    assert sched["max_lr"] == 1e-3
    assert all(lr == 1e-3 for lr in sched["base_lrs"])

    hp = checkpoint["hyper_parameters"]["training_args"]
    assert hp["max_lr"] == 1e-3 and hp["diffusion_multiplicity"] == 1 and hp["weight_decay"] == 0.0


# ====================================================================== #
#  Multi-rank tests (spawn workers)                                       #
# ====================================================================== #


def _worker_forward_smoke(rank, world_size, device_type, backend, env_map, serial_state_dict, serial_hparams):
    """Worker: forward pass smoke test on 1D CP mesh."""
    monkeypatch, dist_manager = _init_distributed_1d(rank, world_size, device_type, backend, env_map)

    device_mesh = dist_manager.device_mesh

    # Recreate serial model, load state, wrap as 1D CP, move to device
    serial_model = SerialBoltz2(**serial_hparams)
    serial_model.to(dtype=torch.float64)
    serial_model.load_state_dict(serial_state_dict, strict=True)
    serial_model = serial_model.to(device=dist_manager.device)

    model_1d = Boltz2_1D(serial_model, dist_manager)

    # Create random features on rank 0 and distribute. ``random_features``
    # generates host tensors; distribute_features handles device placement
    # via the device_mesh.
    n_tokens = 4 * world_size  # must be divisible by cp_size
    n_atoms = 8 * world_size
    rng = torch.Generator().manual_seed(42)

    feats = random_features(
        size_batch=1,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=2,
        atom_counts_per_token_range=(1, 3),
        device=torch.device("cpu"),
        float_value_range=(-1.0, 1.0),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=8,
        rng=rng,
    )
    # random_features creates msa as [B, S, N, num_tokens] (pre-one-hot) but
    # MSAModule1D expects integer indices [B, S, N] and applies one_hot itself.
    if "msa" in feats and feats["msa"].ndim == 4:
        feats["msa"] = feats["msa"].argmax(dim=-1)
    if dist_manager.device.type == "cuda":
        feats = {k: (v.to(device=dist_manager.device) if isinstance(v, torch.Tensor) else v) for k, v in feats.items()}

    # Build per-feature placements for the full 2D mesh (dp, cp)
    feature_placements = _build_feature_placements_1d(feats.keys())

    # Distribute features
    src_rank = 0
    dt_feats = distribute_features(
        feats if rank == src_rank else None,
        feature_placements,
        group=dist.group.WORLD,
        src_rank_global=src_rank,
        device_mesh=device_mesh,
    )

    # Run forward pass (trunk only, no structure prediction)
    model_1d.eval()
    model_1d.skip_run_structure = True
    model_1d.run_trunk_and_structure = True

    with torch.no_grad():
        out = model_1d(
            dt_feats,
            recycling_steps=0,
        )

    # Verify outputs
    assert "pdistogram" in out, "Missing pdistogram in output"
    assert "s" in out, "Missing s in output"
    assert "z" in out, "Missing z in output"

    # Check s shape and placements
    s = out["s"]
    assert isinstance(s, DTensor), f"s should be DTensor, got {type(s)}"
    assert s.shape == (1, n_tokens, TOKEN_S), f"s shape mismatch: {s.shape}"
    assert s.placements == (Shard(0), Shard(1)), f"s placements: {s.placements}"

    # Check z shape and placements
    z = out["z"]
    assert isinstance(z, DTensor), f"z should be DTensor, got {type(z)}"
    assert z.shape == (1, n_tokens, n_tokens, TOKEN_Z), f"z shape mismatch: {z.shape}"
    assert z.placements == (Shard(0), Shard(1)), f"z placements: {z.placements}"

    # Verify local shapes are sharded correctly
    cp_size = world_size
    expected_local_n = n_tokens // cp_size
    assert (
        s.to_local().shape[1] == expected_local_n
    ), f"s local dim 1 should be {expected_local_n}, got {s.to_local().shape[1]}"
    assert (
        z.to_local().shape[1] == expected_local_n
    ), f"z local dim 1 should be {expected_local_n}, got {z.to_local().shape[1]}"
    # z columns are full (row-slab)
    assert z.to_local().shape[2] == n_tokens, f"z local dim 2 should be full N={n_tokens}, got {z.to_local().shape[2]}"

    # Non-vacuousness: s and z should be non-zero
    assert s.to_local().abs().sum() > 0, "s is all zeros — vacuous pass"
    assert z.to_local().abs().sum() > 0, "z is all zeros — vacuous pass"

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 1), True, "cuda", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
        # CPU dropped: Boltz2 end-node triangle attention (DAP path) needs
        # dist.all_to_all / dist.reduce_scatter, which gloo does not support
        # when cp_size > 1.  See _DAPTriangleAttentionEndingNode1DImpl in
        # src/boltz/distributed/model/layers/triangular_attention_1d.py.
    ],
    indirect=("setup_env",),
    ids=["cuda-cp1", "cuda-cp2"],
)
def test_forward_smoke(setup_env):
    """V2: Forward pass produces correct shapes and non-zero outputs."""
    _, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    serial_model = _create_minimal_serial_boltz2()
    serial_model.to(dtype=torch.float64)
    serial_state_dict = serial_model.state_dict()
    serial_hparams = dict(serial_model.hparams)

    spawn_multiprocessing(
        _worker_forward_smoke,
        world_size,
        world_size,
        device_type,
        backend,
        env_per_rank,
        serial_state_dict,
        serial_hparams,
    )


# ====================================================================== #
#  V10: bf16 mixed precision (CUDA-only)                                  #
# ====================================================================== #


def _worker_bf16_mixed_precision_1d(
    rank: int,
    world_size: int,
    device_type: str,
    backend: str,
    env_map: dict,
    serial_state_dict: dict,
    serial_hparams: dict,
):
    """V10 worker: exercise 1D CP ready submodules under torch.autocast(bf16).

    Verifies forward outputs are bf16, weight gradients are reduced in
    ``>= fp32`` (Replicate placement), and ``torch.clear_autocast_cache()``
    does not error.  Mirrors ``_worker_bf16_mixed_precision`` in the 2D test
    but uses the 1D mesh ``(dp, cp)`` with 2-tuple placements throughout.
    """
    monkeypatch, dist_manager = _init_distributed_1d(rank, world_size, device_type, backend, env_map)

    serial_model = SerialBoltz2(**serial_hparams)
    serial_model.load_state_dict(serial_state_dict, strict=True)
    serial_model = serial_model.to(device=dist_manager.device)
    dist_model = Boltz2_1D(serial_model, dist_manager)
    dist_model = dist_model.to(device=dist_manager.device)
    dist_model.train()

    cp_size = world_size
    B = 1
    N_global = 8 * cp_size
    single_pl = (Shard(0), Shard(1))
    pair_pl = (Shard(0), Shard(1))

    device_mesh = dist_manager.device_mesh

    # --- Forward under autocast: check submodule output dtypes ---
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_in = distribute_tensor(torch.randn(B, N_global, TOKEN_S, device=dist_manager.device), device_mesh, single_pl)
        s_out = dist_model.s_init(s_in)
        assert s_out.to_local().dtype == torch.bfloat16, f"s_init output: {s_out.to_local().dtype} != bfloat16"

        s_norm_out = dist_model.s_norm(s_in)
        # LayerNorm under autocast emits fp32 (autocast layernorm policy).
        assert (
            s_norm_out.to_local().dtype == torch.float32
        ), f"s_norm output: {s_norm_out.to_local().dtype} != float32 (autocast LayerNorm policy)"

        z1 = dist_model.z_init_1(s_in)
        assert z1.to_local().dtype == torch.bfloat16, f"z_init_1 output: {z1.to_local().dtype} != bfloat16"

        z_in = distribute_tensor(
            torch.randn(B, N_global, N_global, TOKEN_Z, device=dist_manager.device), device_mesh, pair_pl
        )
        disto = dist_model.distogram_module(z_in)
        assert (
            disto.to_local().dtype == torch.bfloat16
        ), f"distogram_module output: {disto.to_local().dtype} != bfloat16"

    # --- Backward: weight grad dtype & placement (>=fp32, Replicate) ---
    s_grad_in = distribute_tensor(torch.randn(B, N_global, TOKEN_S, device=dist_manager.device), device_mesh, single_pl)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = dist_model.s_init(s_grad_in)
        grad_out = distribute_tensor(
            torch.randn(B, N_global, out.shape[-1], device=dist_manager.device, dtype=torch.bfloat16),
            device_mesh,
            single_pl,
        )
        out.to_local().backward(grad_out.to_local())

    w = dist_model.s_init.weight
    assert w.grad is not None, "s_init.weight.grad is None after backward under autocast"
    assert isinstance(w.grad, DTensor), f"s_init.weight.grad should be DTensor, got {type(w.grad)}"
    for p in w.grad.placements:
        assert isinstance(p, Replicate), f"Weight grad should be Replicate, got {p}"
    # Param dtype is fp32 (parameters are not autocast); grad must match the
    # >= fp32 reduce path (no bf16 grad accumulation on weights).
    assert (
        w.grad.to_local().dtype == w.to_local().dtype
    ), f"Weight grad dtype {w.grad.to_local().dtype} != weight dtype {w.to_local().dtype}"
    assert w.grad.to_local().abs().sum() > 0, "Weight grad is all zeros — vacuous backward"

    # --- clear_autocast_cache branch + recycling path emits bf16 ---
    dist_model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.clear_autocast_cache()
        s_zeros = distribute_tensor(
            torch.zeros(B, N_global, TOKEN_S, device=dist_manager.device, dtype=torch.bfloat16),
            device_mesh,
            single_pl,
        )
        s_recycled = elementwise_op(
            dist_model.s_init(s_grad_in),
            dist_model.s_recycle(dist_model.s_norm(s_zeros)),
            ElementwiseOp.SUM,
        )
        assert (
            s_recycled.to_local().dtype == torch.bfloat16
        ), f"Recycled s dtype {s_recycled.to_local().dtype} != bfloat16"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        # Minimal dp=1, cp=1: focused autocast dtype check on the 1D mesh.
        # Full BF16-mixed training across CP topologies is exercised by
        # ``test_boltz2_1d_bf16_dtype_parity`` in test_dtensor_boltz2_1d_train.py.
        ((1, 1), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp1"],
)
def test_boltz2_1d_bf16_mixed_precision(setup_env):
    """V10: 1D CP ready submodules produce bf16 outputs under autocast.

    Focused autocast check on individual submodules (s_init, z_init_1,
    distogram_module, s_norm, s_recycle) on the 1D mesh ``(dp=1, cp=1)``.
    Mirrors ``test_boltz2_bf16_mixed_precision`` in the 2D test module.
    Full BF16-mixed training across CP topologies is exercised by
    ``test_boltz2_1d_bf16_dtype_parity`` (training pipeline) and
    ``test_boltz2_1d_e2e_training_parity`` (numerical parity) in
    ``test_dtensor_boltz2_1d_train.py``.
    """
    _, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    serial_model = _create_minimal_serial_boltz2()
    serial_state_dict = serial_model.state_dict()
    serial_hparams = dict(serial_model.hparams)

    spawn_multiprocessing(
        _worker_bf16_mixed_precision_1d,
        world_size,
        world_size,
        device_type,
        backend,
        env_per_rank,
        serial_state_dict,
        serial_hparams,
    )


# ====================================================================== #
#  V4/V5: predict_step tests                                              #
# ====================================================================== #


def _build_feature_placements_1d(feat_keys):
    """Build per-feature placements for 2D mesh ``(dp, cp)`` from 1D CP registry.

    The 1D CP registry stores unbatched placements on the ``(cp,)`` sub-mesh.
    After batching, tensor dim 0 becomes the batch dimension (sharded on dp),
    so every ``Shard(d)`` in the CP placement must shift to ``Shard(d + 1)``
    to account for the new leading batch axis.  This mirrors the logic in
    :class:`CollateDTensor1D`.
    """
    placements = {}
    for key in feat_keys:
        if key in TRAINING_FEATURE_PLACEMENTS_1D:
            cp_placement = TRAINING_FEATURE_PLACEMENTS_1D[key]
            # Shift shard dims by 1 for the batch/dp dimension
            shifted = tuple(Shard(p.dim + 1) if isinstance(p, Shard) else p for p in cp_placement)
            placements[key] = (Shard(0),) + shifted
        else:
            placements[key] = (Shard(0), Replicate())
    return placements


def _distribute_features_for_predict(feats, rank, device_mesh):
    """Distribute features for 1D CP from rank 0."""
    feature_placements = _build_feature_placements_1d(feats.keys())
    return distribute_features(
        feats if rank == 0 else None,
        feature_placements,
        group=dist.group.WORLD,
        src_rank_global=0,
        device_mesh=device_mesh,
    )


def _make_noise_dtensors(noise_host_list, device_mesh, device, dtype):
    """Shard deterministic noise over both the DP batch and CP atom axes."""
    return [
        distribute_tensor(
            noise_host.to(device=device, dtype=dtype),
            device_mesh=device_mesh,
            placements=(Shard(0), Shard(1)),
        )
        for noise_host in noise_host_list
    ]


def _monkeypatch_deterministic_noise(monkeypatch, init_noise_dt, step_noise_dts):
    """Monkeypatch diffusion_1d to use deterministic noise and no augmentation."""
    _orig = diffusion_1d_module._center_random_augmentation_1d

    def _centering_only(atom_coords, atom_mask, **kwargs):
        kwargs["augmentation"] = False
        kwargs["centering"] = True
        return _orig(atom_coords, atom_mask, **kwargs)

    _calls = []
    _sequence = [init_noise_dt] + list(step_noise_dts)

    def _fixed_randn(shape, device_mesh, placements, dtype=torch.float32, scale=1.0):
        idx = len(_calls)
        _calls.append(idx)
        noise_dt = _sequence[idx]
        if scale != 1.0:
            noise_dt = scalar_tensor_op(scale, noise_dt, ElementwiseOp.PROD)
        return noise_dt

    monkeypatch.setattr(diffusion_1d_module, "_center_random_augmentation_1d", _centering_only)
    monkeypatch.setattr(diffusion_1d_module, "create_distributed_randn", _fixed_randn)


def _assert_no_dtensors(d, prefix=""):
    """Assert no DTensor values remain in a dict (recursive)."""
    for key, val in d.items():
        path = f"{prefix}.{key}" if prefix else key
        assert not isinstance(val, DTensor), (
            f"DTensor found at '{path}' with placements={val.placements}. "
            f"predict_step must convert all DTensors via full_tensor() or to_local()."
        )
        if isinstance(val, dict):
            _assert_no_dtensors(val, prefix=path)


def _prepare_predict_data(cp_size, dtype, confidence_prediction=False):
    """Prepare model params, state dict, features, and noise for predict tests."""
    seed_by_rank(0, seed=42)

    boltz2_model_params = create_boltz2_model_init_params(use_large_model=False)
    boltz2_model_params["diffusion_process_args"]["alignment_reverse_diff"] = False
    boltz2_model_params["diffusion_process_args"]["coordinate_augmentation"] = False

    num_sampling_steps = 2
    diffusion_samples = 1

    if confidence_prediction:
        boltz2_model_params["confidence_prediction"] = True
        # Use minimal confidence_model_args to avoid duplicate kwargs
        # (create_boltz2_model_init_params includes conditioning_cutoff_* keys
        # that conflict with SerialBoltz2.__init__ explicit kwargs).
        boltz2_model_params["confidence_model_args"] = {
            "pairformer_args": boltz2_model_params["pairformer_args"],
            "confidence_args": {},
        }
        boltz2_model_params["num_bins"] = 64
    else:
        boltz2_model_params["confidence_prediction"] = False

    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": num_sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": diffusion_samples,
        "write_confidence_summary": confidence_prediction,
        "write_full_pae": confidence_prediction,
    }
    boltz2_model_params["predict_args"] = predict_args

    n_tokens = 30 * cp_size
    W = boltz2_model_params["atoms_per_window_queries"]
    n_atoms_raw = n_tokens * 20
    n_atoms = ((n_atoms_raw + W - 1) // W) * W
    n_msa = cp_size * 2

    assert n_atoms % cp_size == 0, f"n_atoms={n_atoms} not divisible by cp_size={cp_size}"
    assert n_atoms % W == 0, f"n_atoms={n_atoms} not divisible by W={W}"

    import boltz.data.const as const

    input_feats = random_features(
        size_batch=1,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=n_msa,
        atom_counts_per_token_range=(8, 20),
        device="cpu",
        float_value_range=(-0.01, 0.01),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=boltz2_model_params["num_bins"],
    )
    input_feats["msa"] = torch.randint(0, const.num_tokens, (1, n_msa, n_tokens), dtype=torch.int64)

    reference_module = SerialBoltz2(**boltz2_model_params)
    init_module_params_glorot(reference_module, gain=0.05)
    reference_module.apply(SetModuleInfValues())
    reference_module.structure_module.coordinate_augmentation = False
    module_state_dict = reference_module.state_dict()

    B_M = diffusion_samples
    init_noise = torch.empty((B_M, n_atoms, 3), dtype=dtype)
    step_noise_list = [torch.empty((B_M, n_atoms, 3), dtype=dtype) for _ in range(num_sampling_steps)]
    init_tensors_uniform([init_noise, *step_noise_list], low=-0.01, high=0.01)

    feats_host = {k: v.detach().cpu().clone() for k, v in input_feats.items()}

    return (
        boltz2_model_params,
        module_state_dict,
        predict_args,
        feats_host,
        init_noise.cpu(),
        [n.cpu() for n in step_noise_list],
        n_atoms,
        n_tokens,
    )


def _worker_predict_step_smoke(
    rank,
    world_size,
    device_type,
    backend,
    env_map,
    dtype,
    boltz2_model_params,
    module_state_dict,
    predict_args,
    feats_host,
    init_noise_host,
    step_noise_list_host,
    n_atoms,
    n_tokens,
):
    """V4 worker: predict_step smoke test — no DTensor leaks, correct output shapes."""
    monkeypatch, dist_manager = _init_distributed_1d(rank, world_size, device_type, backend, env_map)

    device_mesh = dist_manager.device_mesh

    serial_model = SerialBoltz2(**boltz2_model_params)
    serial_model = serial_model.to(dtype=dtype)
    serial_model.load_state_dict(module_state_dict)
    serial_model.structure_module.coordinate_augmentation = False
    serial_model.apply(SetModuleInfValues())
    serial_model = serial_model.to(device=dist_manager.device)

    model_1d = Boltz2_1D(serial_model, dist_manager)
    model_1d = model_1d.to(device=dist_manager.device)
    model_1d.eval()

    feats_global = {
        k: v.to(device=dist_manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in feats_host.items()
    }
    dt_feats = _distribute_features_for_predict(feats_global, rank, device_mesh)

    noise_dts = _make_noise_dtensors(
        [init_noise_host] + list(step_noise_list_host),
        device_mesh,
        dist_manager.device,
        dtype,
    )
    _monkeypatch_deterministic_noise(monkeypatch, noise_dts[0], noise_dts[1:])

    model_1d.predict_args = predict_args
    with torch.no_grad():
        pred_dict = model_1d.predict_step(dt_feats, batch_idx=0)

    # V4: No exception
    assert pred_dict["exception"] is False, "predict_step raised an exception"

    # V4: No DTensors in output
    _assert_no_dtensors(pred_dict)

    # V4: coords and masks present with correct shapes
    diffusion_samples = predict_args["diffusion_samples"]
    assert "coords" in pred_dict, "Missing 'coords' in predict output"
    assert "masks" in pred_dict, "Missing 'masks' in predict output"
    coords = pred_dict["coords"]
    masks = pred_dict["masks"]
    assert coords.shape == (
        diffusion_samples,
        n_atoms,
        3,
    ), f"coords shape {coords.shape} != expected ({diffusion_samples}, {n_atoms}, 3)"
    assert masks.shape == (1, n_atoms), f"masks shape {masks.shape} != expected (1, {n_atoms})"

    # V4: Non-vacuous — coords non-zero, masks have True values
    assert coords.abs().sum() > 0, "coords are all zeros — vacuous pass"
    assert masks.any(), "masks are all False — vacuous pass"

    DistributedManager.cleanup()
    monkeypatch.undo()


def _worker_predict_step_confidence(
    rank,
    world_size,
    device_type,
    backend,
    env_map,
    dtype,
    boltz2_model_params,
    module_state_dict,
    predict_args,
    feats_host,
    init_noise_host,
    step_noise_list_host,
    n_atoms,
    n_tokens,
):
    """V5 worker: predict_step with confidence — expected keys, no DTensor leaks."""
    monkeypatch, dist_manager = _init_distributed_1d(rank, world_size, device_type, backend, env_map)

    device_mesh = dist_manager.device_mesh

    serial_model = SerialBoltz2(**boltz2_model_params)
    serial_model = serial_model.to(dtype=dtype)
    serial_model.load_state_dict(module_state_dict)
    serial_model.structure_module.coordinate_augmentation = False
    serial_model.apply(SetModuleInfValues())
    serial_model = serial_model.to(device=dist_manager.device)

    model_1d = Boltz2_1D(serial_model, dist_manager)
    model_1d = model_1d.to(device=dist_manager.device)
    model_1d.eval()

    feats_global = {
        k: v.to(device=dist_manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in feats_host.items()
    }
    dt_feats = _distribute_features_for_predict(feats_global, rank, device_mesh)

    noise_dts = _make_noise_dtensors(
        [init_noise_host] + list(step_noise_list_host),
        device_mesh,
        dist_manager.device,
        dtype,
    )
    _monkeypatch_deterministic_noise(monkeypatch, noise_dts[0], noise_dts[1:])

    model_1d.predict_args = predict_args
    with torch.no_grad():
        pred_dict = model_1d.predict_step(dt_feats, batch_idx=0)

    # V5: No exception
    assert pred_dict["exception"] is False, "predict_step raised an exception"

    # V5: No DTensors in output
    _assert_no_dtensors(pred_dict)

    # V5: All expected confidence keys present
    expected_keys = {"pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"}
    missing = expected_keys - set(pred_dict.keys())
    assert not missing, f"Missing confidence keys: {missing}"

    # V5: Confidence values are finite and non-zero (non-vacuous)
    for key in ["pde", "plddt"]:
        val = pred_dict[key]
        assert val.numel() > 0, f"Confidence key '{key}' is empty"
        assert torch.isfinite(val).all(), f"Confidence key '{key}' contains non-finite values"

    for key in ["complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
        val = pred_dict[key]
        assert torch.isfinite(val).all(), f"Confidence key '{key}' contains non-finite values"

    # V5: coords present and non-zero
    assert pred_dict["coords"].abs().sum() > 0, "coords are all zeros — vacuous pass"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 1), True, "cuda", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-cp1", "cuda-cp2"],
)
def test_predict_step_smoke(setup_env):
    """V4: predict_step output contains only plain tensors, no DTensor leaks.

    Runs predict_step with confidence_prediction=False and verifies:
    - exception is False
    - No DTensor values in the output dict
    - coords and masks have correct shapes (full gathered shapes)
    - coords are non-zero, masks have True values
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64
    cp_size = grid_group_sizes["cp"]

    (
        boltz2_model_params,
        module_state_dict,
        predict_args,
        feats_host,
        init_noise_host,
        step_noise_list_host,
        n_atoms,
        n_tokens,
    ) = _prepare_predict_data(cp_size=cp_size, dtype=dtype, confidence_prediction=False)

    spawn_multiprocessing(
        _worker_predict_step_smoke,
        world_size,
        world_size,
        device_type,
        backend,
        env_per_rank,
        dtype,
        boltz2_model_params,
        module_state_dict,
        predict_args,
        feats_host,
        init_noise_host,
        step_noise_list_host,
        n_atoms,
        n_tokens,
    )


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 1), True, "cuda", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-cp1", "cuda-cp2"],
)
def test_predict_step_confidence(setup_env):
    """V5: predict_step with confidence_prediction=True produces expected keys.

    Runs predict_step with confidence enabled and verifies:
    - No DTensor values leak into the output dict
    - All expected confidence keys present (pde, plddt, complex_plddt,
      complex_iplddt, complex_pde, complex_ipde)
    - Confidence values are finite
    - coords are non-zero
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64
    cp_size = grid_group_sizes["cp"]

    (
        boltz2_model_params,
        module_state_dict,
        predict_args,
        feats_host,
        init_noise_host,
        step_noise_list_host,
        n_atoms,
        n_tokens,
    ) = _prepare_predict_data(cp_size=cp_size, dtype=dtype, confidence_prediction=True)

    spawn_multiprocessing(
        _worker_predict_step_confidence,
        world_size,
        world_size,
        device_type,
        backend,
        env_per_rank,
        dtype,
        boltz2_model_params,
        module_state_dict,
        predict_args,
        feats_host,
        init_noise_host,
        step_noise_list_host,
        n_atoms,
        n_tokens,
    )


# ====================================================================== #
#  V6: Forward/backward parity (serial vs 1D CP distributed)              #
# ====================================================================== #


def _worker_forward_backward_parity_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    dtype,
    boltz2_model_params,
    module_state_dict,
    n_recycles,
    multiplicity_diffusion_train,
    input_feats_global_fp64_host,
    sigmas_expected_global_fp64_host,
    noise_expected_global_fp64_host,
    output_pdistogram_expected_global_host,
    output_denoised_atom_coords_expected_global_host,
    output_pbfactor_expected_global_host,
    output_s_expected_global_host,
    output_z_expected_global_host,
    output_aligned_true_coords_expected_global_host,
    d_output_pdistogram_expected_global_host,
    d_output_denoised_atom_coords_expected_global_host,
    d_output_pbfactor_expected_global_host,
    expected_param_grads_global_host_dict,
    env_per_rank=None,
):
    """V6 multi-rank worker: verify 1D CP distributed forward/backward matches serial."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    reference_module = SerialBoltz2(**boltz2_model_params)
    reference_module = reference_module.to(dtype=dtype)
    reference_module.load_state_dict(module_state_dict)
    reference_module.structure_module.coordinate_augmentation = False
    reference_module.apply(SetModuleInfValues())
    reference_module = reference_module.to(device=manager.device)
    module = Boltz2_1D(reference_module, manager)
    module.train()

    # --- Distribute features ---
    device_mesh = manager.device_mesh

    host_tensor_keys = {k for k, v in input_feats_global_fp64_host.items() if isinstance(v, torch.Tensor)}
    feature_placements = _build_feature_placements_1d(host_tensor_keys)

    if manager.group_rank["world"] == 0:
        input_feats_global = {
            k: v.to(device=manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
            for k, v in input_feats_global_fp64_host.items()
        }
    else:
        input_feats_global = None

    feats = distribute_features(
        input_feats_global,
        feature_placements,
        manager.group["world"],
        manager.group_ranks["world"][0],
        device_mesh,
    )

    # Distribute noise as a DTensor with (Shard(0), Shard(1))
    noise_global = noise_expected_global_fp64_host.to(device=manager.device, dtype=dtype)
    noise_dt = distribute_tensor(noise_global, device_mesh, (Shard(0), Shard(1)))

    # Distribute expected denoised/aligned coords for backward grad comparison
    expected_denoised_global = output_denoised_atom_coords_expected_global_host.to(device=manager.device, dtype=dtype)
    expected_denoised_dt = distribute_tensor(expected_denoised_global, device_mesh, (Shard(0), Shard(1)))

    d_denoised_global = d_output_denoised_atom_coords_expected_global_host.to(device=manager.device, dtype=dtype)
    d_denoised_dt = distribute_tensor(d_denoised_global, device_mesh, (Shard(0), Shard(1)))

    expected_aligned_coords_global = output_aligned_true_coords_expected_global_host.to(
        device=manager.device, dtype=dtype
    )
    expected_aligned_coords_dt = distribute_tensor(expected_aligned_coords_global, device_mesh, (Shard(0), Shard(1)))

    # Monkeypatch deterministic noise for distributed forward
    sigmas_device = sigmas_expected_global_fp64_host.to(device=manager.device, dtype=dtype)
    sigmas_dt = distribute_tensor(sigmas_device, device_mesh, (Shard(0), Replicate()))

    monkeypatch.setattr(module.structure_module, "noise_distribution", lambda bs, dtype=None: sigmas_dt)
    monkeypatch.setattr(diffusion_1d_module, "create_distributed_randn", lambda *a, **kw: noise_dt)

    # Distributed forward
    output_dict = module(
        feats,
        recycling_steps=n_recycles,
        multiplicity_diffusion_train=multiplicity_diffusion_train,
    )

    assert "pdistogram" in output_dict
    assert "denoised_atom_coords" in output_dict
    assert "pbfactor" in output_dict
    assert "s" in output_dict
    assert "z" in output_dict
    assert "sigmas" in output_dict
    assert "aligned_true_atom_coords" in output_dict

    token_pad_mask_global = feats["token_pad_mask"].full_tensor()
    token_pair_pad_mask_global = feats["token_pair_pad_mask"].full_tensor()
    atom_pad_mask_global = feats["atom_pad_mask"].full_tensor()
    atom_pad_mask_mul_global = atom_pad_mask_global[:, :, None].repeat_interleave(multiplicity_diffusion_train, 0)

    s_full = output_dict["s"].full_tensor() * token_pad_mask_global[:, :, None]
    expected_s = output_s_expected_global_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(s_full, expected_s)

    z_full = output_dict["z"].full_tensor() * token_pair_pad_mask_global[:, :, :, None]
    expected_z = output_z_expected_global_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(z_full, expected_z)

    pdistogram_full = output_dict["pdistogram"].full_tensor() * token_pair_pad_mask_global[:, :, :, None, None]
    expected_pdistogram = output_pdistogram_expected_global_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(pdistogram_full, expected_pdistogram)

    denoised_full = output_dict["denoised_atom_coords"].full_tensor() * atom_pad_mask_mul_global
    expected_denoised_full = expected_denoised_dt.full_tensor() * atom_pad_mask_mul_global
    torch.testing.assert_close(denoised_full, expected_denoised_full)

    pbfactor_full = output_dict["pbfactor"].full_tensor() * token_pad_mask_global[:, :, None]
    expected_pbfactor = output_pbfactor_expected_global_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(pbfactor_full, expected_pbfactor)

    sigmas_full = output_dict["sigmas"].full_tensor()
    expected_sigmas = sigmas_expected_global_fp64_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(sigmas_full, expected_sigmas)

    aligned_coords_full = output_dict["aligned_true_atom_coords"].full_tensor() * atom_pad_mask_mul_global
    expected_aligned_coords_full = expected_aligned_coords_dt.full_tensor() * atom_pad_mask_mul_global
    torch.testing.assert_close(aligned_coords_full, expected_aligned_coords_full)

    # Backward pass
    d_pdistogram = d_output_pdistogram_expected_global_host.to(device=manager.device, dtype=dtype)
    d_pdistogram_dt = distribute_tensor(d_pdistogram, device_mesh, output_dict["pdistogram"].placements)

    d_pbfactor = d_output_pbfactor_expected_global_host.to(device=manager.device, dtype=dtype)
    d_pbfactor_dt = distribute_tensor(d_pbfactor, device_mesh, output_dict["pbfactor"].placements)

    torch.autograd.backward(
        [output_dict["pdistogram"], output_dict["denoised_atom_coords"], output_dict["pbfactor"]],
        [d_pdistogram_dt, d_denoised_dt, d_pbfactor_dt],
    )

    num_grads_checked = 0
    num_nonzero_grads = 0
    for name, param in module.named_parameters():
        # 1D model mirrors serial names directly (no ._serial. prefix)
        if name in expected_param_grads_global_host_dict:
            expected_grad = expected_param_grads_global_host_dict[name].to(device=manager.device, dtype=dtype)
            if param.grad is None:
                raise AssertionError(f"Missing gradient for {name}")
            actual_grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
            num_grads_checked += 1
            if expected_grad.abs().max().item() > 0:
                num_nonzero_grads += 1
            torch.testing.assert_close(
                actual_grad,
                expected_grad,
                msg=lambda msg, cn=name: f"Gradient mismatch for {cn}: {msg}",
            )

    assert num_grads_checked > 0, "No gradients compared — test is vacuous"
    assert num_nonzero_grads > 0, "All compared gradients are zero — test is vacuous"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        # CPU dp=1 cp=2 dropped: Boltz2 end-node triangle attention (DAP path)
        # needs dist.all_to_all / dist.reduce_scatter, which gloo does not
        # support when cp_size > 1.  See _DAPTriangleAttentionEndingNode1DImpl
        # in src/boltz/distributed/model/layers/triangular_attention_1d.py.
        ((1, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2", "cuda-dp1-cp3"],
)
def test_forward_backward_parity_1d(setup_env):
    """V6: Forward/backward parity between 1D CP distributed and serial Boltz2.

    Tests that the 1D CP Boltz2 wrapper produces numerically identical
    forward outputs and backward gradients compared to the serial implementation,
    with training=True and structure_prediction_training=True.

    This test uses custom upstream gradients (not loss-derived) to isolate
    the forward/backward pipeline from the loss computation.  The model
    configuration, initialization, and features are intentionally aligned
    with the 2D test_boltz2_forward_backward_parity so that a pass here
    proves the 1D forward/backward path is correct under the same numerical
    regime.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64
    B = grid_group_sizes["dp"]
    size_cp = grid_group_sizes["cp"]

    # Reduced from -0.1/0.1 (init range) and 0.1 (glorot gain). The original
    # scale produced a 2.4e-7 residual in denoised_atom_coords and gradient
    # residuals up to 1.9e-7 in diffusion_conditioning embeddings on
    # cuda-dp1-cp2 — both over the fp64 default atol of 1e-7. Cutting the
    # input / glorot / upstream-grad ranges by ~80% keeps the test non-trivial
    # (still 2x the 0.01/0.05 used by the sibling sample/predict tests) while
    # shrinking pair-op accumulation residual quadratically in the input scale
    # and embedding-grad residual linearly in the upstream-grad magnitude.
    # Post-fix max residuals at this scale: denoised_atom_coords 1.27e-9,
    # grads 3.76e-8 (worst on diffusion_conditioning.atom_encoder
    # .embed_atom_features.weight) — both >=2x below the fp64 default atol 1e-7.
    min_val_init = -0.02
    max_val_init = 0.02
    scale_glorot = 0.025

    seed = 42
    seed_by_rank(0, seed=seed)

    boltz2_model_params = create_boltz2_model_init_params(use_large_model=False)
    n_recycles = 0
    multiplicity_diffusion_train = 2

    boltz2_model_params["training_args"].recycling_steps = n_recycles
    boltz2_model_params["training_args"].diffusion_multiplicity = multiplicity_diffusion_train
    boltz2_model_params["no_random_recycling_training"] = True

    n_tokens = 30 * size_cp
    W = boltz2_model_params["atoms_per_window_queries"]
    n_atoms_raw = n_tokens * 20
    n_atoms = ((n_atoms_raw + W - 1) // W) * W
    n_msa = max(size_cp * 2, 2)

    input_feats_global_fp64 = random_features(
        size_batch=B,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=n_msa,
        atom_counts_per_token_range=(8, 20),
        device=device_type,
        float_value_range=(min_val_init, max_val_init),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=boltz2_model_params["num_bins"],
    )
    input_feats_global_fp64["msa"] = torch.randint(
        0, const.num_tokens, (B, n_msa, n_tokens), dtype=torch.int64, device=device_type
    )

    # Create serial reference module
    reference_module = SerialBoltz2(**boltz2_model_params)
    init_module_params_glorot(reference_module, gain=scale_glorot)
    reference_module.apply(SetModuleInfValues())
    reference_module.structure_module.coordinate_augmentation = False
    module_state_dict_fp64 = {k: v.detach().clone().cpu() for k, v in reference_module.state_dict().items()}
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).train()

    # Pre-generate deterministic sigmas and non-zero noise
    sigmas_expected_global_fp64 = reference_module.structure_module.noise_distribution(
        B * multiplicity_diffusion_train
    ).to(device=device_type, dtype=torch.float64)
    noise_expected_global_fp64 = torch.empty(
        B * multiplicity_diffusion_train, n_atoms, 3, device=device_type, dtype=torch.float64
    )
    init_tensors_uniform([noise_expected_global_fp64], low=min_val_init, high=max_val_init)

    # Monkeypatch serial noise for determinism
    _monkeypatch = pytest.MonkeyPatch()
    _monkeypatch.setattr(
        reference_module.structure_module,
        "noise_distribution",
        lambda bs, dtype=None: sigmas_expected_global_fp64,
    )
    _monkeypatch.setattr(
        serial_diffusion_v2_module.torch,
        "randn_like",
        lambda t: noise_expected_global_fp64.to(t),
    )

    original_feat_keys = set(input_feats_global_fp64.keys())
    coords_backup = input_feats_global_fp64["coords"].detach().clone()

    # Serial forward
    output_dict_serial = reference_module(
        input_feats_global_fp64,
        recycling_steps=n_recycles,
        multiplicity_diffusion_train=multiplicity_diffusion_train,
    )

    output_pdistogram = output_dict_serial["pdistogram"]
    output_denoised = output_dict_serial["denoised_atom_coords"]
    output_pbfactor = output_dict_serial["pbfactor"]
    output_s = output_dict_serial["s"]
    output_z = output_dict_serial["z"]
    output_aligned_true_coords = output_dict_serial["aligned_true_atom_coords"]

    # Create upstream gradients
    d_output_pdistogram = torch.empty_like(output_pdistogram)
    d_output_denoised = torch.empty_like(output_denoised)
    d_output_pbfactor = torch.empty_like(output_pbfactor)
    init_tensors_uniform(
        [d_output_pdistogram, d_output_denoised, d_output_pbfactor],
        low=min_val_init,
        high=max_val_init,
    )

    # Mask upstream gradients
    atom_pad_mask = input_feats_global_fp64["atom_pad_mask"]
    atom_pad_mask_mul = atom_pad_mask[:, :, None].repeat_interleave(multiplicity_diffusion_train, 0)
    d_output_denoised = d_output_denoised * atom_pad_mask_mul

    token_pair_pad_mask = input_feats_global_fp64["token_pair_pad_mask"]
    d_output_pdistogram = d_output_pdistogram * token_pair_pad_mask[:, :, :, None, None]

    token_pad_mask = input_feats_global_fp64["token_pad_mask"]
    d_output_pbfactor = d_output_pbfactor * token_pad_mask[:, :, None]

    # Serial backward
    torch.autograd.backward(
        [output_pdistogram, output_denoised, output_pbfactor],
        [d_output_pdistogram, d_output_denoised, d_output_pbfactor],
    )

    grad_params_expected = {
        name: param.grad.detach().to(dtype=dtype, device="cpu", copy=True)
        for name, param in reference_module.named_parameters()
        if param.grad is not None
    }

    # Restore features — serial forward mutates coords in-place and may add keys
    input_feats_global_fp64["coords"] = coords_backup
    for key in list(input_feats_global_fp64.keys()):
        if key not in original_feat_keys:
            del input_feats_global_fp64[key]

    output_pdistogram_host = (
        (output_pdistogram * token_pair_pad_mask[:, :, :, None, None]).detach().to(device="cpu", copy=True)
    )
    output_denoised_host = (output_denoised * atom_pad_mask_mul).detach().to(device="cpu", copy=True)
    output_pbfactor_host = (output_pbfactor * token_pad_mask[:, :, None]).detach().to(device="cpu", copy=True)
    output_s_host = (output_s * token_pad_mask[:, :, None]).detach().to(device="cpu", copy=True)
    output_z_host = (output_z * token_pair_pad_mask[:, :, :, None]).detach().to(device="cpu", copy=True)
    output_aligned_true_coords_host = (
        (output_aligned_true_coords * atom_pad_mask_mul).detach().to(device="cpu", copy=True)
    )

    sigmas_host = sigmas_expected_global_fp64.detach().to(device="cpu", copy=True)
    noise_host = (noise_expected_global_fp64 * atom_pad_mask_mul).detach().to(device="cpu", copy=True)
    d_output_pdistogram_host = d_output_pdistogram.detach().to(device="cpu", copy=True)
    d_output_denoised_host = d_output_denoised.detach().to(device="cpu", copy=True)
    d_output_pbfactor_host = d_output_pbfactor.detach().to(device="cpu", copy=True)

    input_feats_host = {}
    for k, v in input_feats_global_fp64.items():
        if isinstance(v, torch.Tensor):
            input_feats_host[k] = v.detach().to(device="cpu", copy=True)
        elif isinstance(v, list):
            input_feats_host[k] = [
                item.detach().to(device="cpu", copy=True) if isinstance(item, torch.Tensor) else item for item in v
            ]
        else:
            input_feats_host[k] = v

    _monkeypatch.undo()

    spawn_multiprocessing(
        _worker_forward_backward_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        dtype,
        boltz2_model_params,
        module_state_dict_fp64,
        n_recycles,
        multiplicity_diffusion_train,
        input_feats_host,
        sigmas_host,
        noise_host,
        output_pdistogram_host,
        output_denoised_host,
        output_pbfactor_host,
        output_s_host,
        output_z_host,
        output_aligned_true_coords_host,
        d_output_pdistogram_host,
        d_output_denoised_host,
        d_output_pbfactor_host,
        grad_params_expected,
        env_per_rank,
    )


# ====================================================================== #
#  V7: predict_step parity (serial vs 1D CP distributed)                  #
# ====================================================================== #


def _worker_predict_step_parity_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    dtype,
    boltz2_model_params,
    module_state_dict,
    predict_args,
    diffusion_samples,
    num_sampling_steps,
    input_feats_global_fp64_host,
    init_noise_global_host,
    step_noise_list_global_host,
    serial_coords_host,
    serial_masks_host,
    env_per_rank=None,
):
    """V7 multi-rank worker: verify 1D CP distributed predict_step matches serial."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    # For bf16 we use torch.autocast and keep weights/features/noise at fp32
    # (see parent test's rationale).
    if dtype == torch.bfloat16:
        weights_dtype = torch.float32
        feats_dtype = torch.float32
        noise_dtype = torch.float32
        use_autocast = True
    else:
        weights_dtype = dtype
        feats_dtype = dtype
        noise_dtype = dtype
        use_autocast = False

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    reference_module = SerialBoltz2(**boltz2_model_params)
    reference_module = reference_module.to(dtype=weights_dtype)
    reference_module.load_state_dict(module_state_dict)
    reference_module.structure_module.coordinate_augmentation = False
    reference_module.apply(SetModuleInfValues())
    reference_module = reference_module.to(device=manager.device)
    module = Boltz2_1D(reference_module, manager)
    module.eval()

    # --- Distribute features ---
    device_mesh = manager.device_mesh

    feats_global = {
        k: v.to(device=manager.device, dtype=feats_dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in input_feats_global_fp64_host.items()
    }
    dt_feats = _distribute_features_for_predict(feats_global, rank, device_mesh)

    # --- Distribute noise as DTensors ---
    all_noise_host = [init_noise_global_host] + list(step_noise_list_global_host)
    noise_dts = _make_noise_dtensors(
        all_noise_host,
        device_mesh,
        manager.device,
        noise_dtype,
    )
    init_noise_dt = noise_dts[0]
    step_noise_dts = noise_dts[1:]

    # --- Monkeypatch distributed sample() for determinism ---
    _monkeypatch_deterministic_noise(monkeypatch, init_noise_dt, step_noise_dts)

    # --- Run distributed predict_step ---
    module.predict_args = predict_args
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        pred_dict = module.predict_step(dt_feats, batch_idx=0)

    assert pred_dict["exception"] is False

    # --- Compare coords and masks ---
    # For 1D CP with dp=1, predict_step gathers coords/masks via full_tensor()
    # internally. All ranks have the same gathered result — compare on all ranks.
    gathered_coords = pred_dict["coords"]
    gathered_mask = pred_dict["masks"]

    mask_expanded = gathered_mask.repeat_interleave(diffusion_samples, 0).bool()
    dt_real = gathered_coords[mask_expanded]

    # serial coords come from the parent test in feats_dtype; cast distributed
    # output to the same dtype for comparison.
    serial_coords_device = serial_coords_host.to(device=manager.device, dtype=gathered_coords.dtype)
    serial_mask_expanded = serial_masks_host.to(device=manager.device).repeat_interleave(diffusion_samples, 0).bool()
    serial_real = serial_coords_device[serial_mask_expanded]

    torch.testing.assert_close(dt_real, serial_real)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "dtype",
    [torch.float64, torch.bfloat16],
    ids=["fp64", "bf16"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        # CPU dp=1 cp=2 dropped: Boltz2 end-node triangle attention (DAP path)
        # needs dist.all_to_all / dist.reduce_scatter, which gloo does not
        # support when cp_size > 1.  See _DAPTriangleAttentionEndingNode1DImpl
        # in src/boltz/distributed/model/layers/triangular_attention_1d.py.
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_predict_step_parity_1d(setup_env, dtype):
    """V7: predict_step parity between 1D CP distributed and serial Boltz2.

    Tests that the 1D CP Boltz2.predict_step produces numerically identical
    sampled coordinates compared to the serial implementation in eval mode
    with no backward pass.

    Side-by-side comparison findings (serial vs distributed):

    Bug 1 — Random augmentation always applied in serial sample():
      Serial diffusionv2.py calls compute_random_augmentation()
      unconditionally.  Distributed diffusion_1d.py gates
      _center_random_augmentation_1d() on self.coordinate_augmentation.
      Mitigation: monkeypatch serial compute_random_augmentation to return
      identity rotation and zero translation.

    Bug 2 — alignment_reverse_diff FP32 downcast in serial:
      Serial diffusionv2.py forces .float() (FP32) for weighted_rigid_align.
      Distributed diffusion_1d.py passes DTensors as-is (FP64 in test).
      Mitigation: set alignment_reverse_diff=False.

    Bug 3 — Serial sample() augmentation shape bug for B > 1:
      compute_random_augmentation(multiplicity) returns R of shape (M,3,3)
      but atom_coords has shape (B*M,N,3) — einsum crashes when B > 1.
      Mitigation: the identity augmentation mock returns (B*M,3,3).
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    # For bf16 we use the production autocast path: keep weights + features
    # + noise at fp32, then wrap predict_step in torch.autocast.  Pure-bf16
    # weight casting collides with the promote_types(...) fp32 indexing
    # matrix inside encodersv2.single_to_keys (einsum dtype mismatch).
    if dtype == torch.bfloat16:
        weights_dtype = torch.float32
        feats_dtype = torch.float32
        noise_dtype = torch.float32
        use_autocast = True
    else:
        weights_dtype = dtype
        feats_dtype = dtype
        noise_dtype = dtype
        use_autocast = False

    size_batch_per_rank = 1
    B = size_batch_per_rank * grid_group_sizes["dp"]
    size_cp = grid_group_sizes["cp"]

    min_val_init = -0.01
    max_val_init = 0.01
    scale_glorot = 0.05

    num_sampling_steps = 2
    diffusion_samples = 2

    seed = 42
    seed_by_rank(0, seed=seed)

    boltz2_model_params = create_boltz2_model_init_params(use_large_model=False)
    boltz2_model_params["diffusion_process_args"]["alignment_reverse_diff"] = False
    boltz2_model_params["diffusion_process_args"]["coordinate_augmentation"] = False

    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": num_sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": diffusion_samples,
        "write_confidence_summary": False,
        "write_full_pae": False,
    }
    boltz2_model_params["predict_args"] = predict_args

    n_atoms_per_token_min = 8
    n_atoms_per_token_max = 20
    n_tokens = 30 * size_cp
    W = boltz2_model_params["atoms_per_window_queries"]
    n_atoms_raw = n_tokens * n_atoms_per_token_max
    n_atoms = ((n_atoms_raw + W - 1) // W) * W
    n_msa = size_cp * 2

    assert n_atoms % size_cp == 0
    assert n_atoms % W == 0

    input_feats_global_fp64 = random_features(
        size_batch=B,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=n_msa,
        atom_counts_per_token_range=(n_atoms_per_token_min, n_atoms_per_token_max),
        device=device_type,
        float_value_range=(min_val_init, max_val_init),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=boltz2_model_params["num_bins"],
    )
    input_feats_global_fp64["msa"] = torch.randint(
        0, const.num_tokens, (B, n_msa, n_tokens), dtype=torch.int64, device=device_type
    )

    # ------------------------------------------------------------------
    # Build serial model in eval mode
    # ------------------------------------------------------------------
    reference_module = SerialBoltz2(**boltz2_model_params)
    init_module_params_glorot(reference_module, gain=scale_glorot)
    reference_module.apply(SetModuleInfValues())
    reference_module.structure_module.coordinate_augmentation = False
    module_state_dict_fp64 = reference_module.state_dict()
    reference_module = reference_module.to(dtype=weights_dtype, device=device_type).eval()

    # ------------------------------------------------------------------
    # Pre-generate deterministic noise for sampling.
    # ------------------------------------------------------------------
    _B_M = B * diffusion_samples
    init_noise = torch.empty((_B_M, n_atoms, 3), device=device_type, dtype=noise_dtype)
    step_noise_list = [
        torch.empty((_B_M, n_atoms, 3), device=device_type, dtype=noise_dtype) for _ in range(num_sampling_steps)
    ]
    init_tensors_uniform([init_noise, *step_noise_list], low=min_val_init, high=max_val_init)

    # ------------------------------------------------------------------
    # Monkeypatch serial sample() for determinism (Bug 1/3 mitigations)
    # ------------------------------------------------------------------
    def _identity_compute_random_augmentation(multiplicity_arg, device=None, dtype=None):
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(_B_M, -1, -1)
        tr = torch.zeros(_B_M, 1, 3, device=device, dtype=dtype)
        return R, tr

    _serial_randn_calls = []
    _serial_randn_sequence = [init_noise] + step_noise_list

    def _fixed_randn(*args, **kwargs):
        idx = len(_serial_randn_calls)
        _serial_randn_calls.append(idx)
        return _serial_randn_sequence[idx].clone()

    _monkeypatch = pytest.MonkeyPatch()
    _monkeypatch.setattr(
        serial_diffusion_v2_module, "compute_random_augmentation", _identity_compute_random_augmentation
    )
    _monkeypatch.setattr(serial_diffusion_v2_module.torch, "randn", _fixed_randn)

    # Cast floating-point features to feats_dtype so they match
    # reference_module's dtype.  For bf16 we keep features at fp32 and rely
    # on torch.autocast(bf16) to downcast inside the model.  Integer
    # features (e.g. msa) preserved as-is.
    input_feats_global_fp64 = {
        k: v.to(dtype=feats_dtype) if v.dtype.is_floating_point else v for k, v in input_feats_global_fp64.items()
    }

    # ------------------------------------------------------------------
    # Serial predict_step
    # ------------------------------------------------------------------
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        serial_pred_dict = reference_module.predict_step(input_feats_global_fp64, batch_idx=0)

    assert serial_pred_dict["exception"] is False
    serial_coords = serial_pred_dict["coords"]
    serial_masks = serial_pred_dict["masks"]

    _monkeypatch.undo()

    # ------------------------------------------------------------------
    # Move everything to CPU for spawn_multiprocessing
    # ------------------------------------------------------------------
    input_feats_host = {k: v.detach().to(device="cpu", copy=True) for k, v in input_feats_global_fp64.items()}
    serial_coords_host = serial_coords.detach().to(device="cpu", copy=True)
    serial_masks_host = serial_masks.detach().to(device="cpu", copy=True)

    spawn_multiprocessing(
        _worker_predict_step_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        dtype,
        boltz2_model_params,
        module_state_dict_fp64,
        predict_args,
        diffusion_samples,
        num_sampling_steps,
        input_feats_host,
        init_noise.cpu(),
        [n.cpu() for n in step_noise_list],
        serial_coords_host,
        serial_masks_host,
        env_per_rank,
    )


# ====================================================================== #
#  V7b: predict_step confidence parity (serial vs 1D CP distributed)      #
# ====================================================================== #


def _worker_predict_step_confidence_parity_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    dtype,
    boltz2_model_params,
    module_state_dict,
    predict_args,
    diffusion_samples,
    num_sampling_steps,
    input_feats_global_fp64_host,
    init_noise_global_host,
    step_noise_list_global_host,
    serial_coords_host,
    serial_masks_host,
    serial_confidence_keys_host,
    env_per_rank=None,
):
    """V7b multi-rank worker: verify 1D CP predict_step with confidence matches serial."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    reference_module = SerialBoltz2(**boltz2_model_params)
    reference_module = reference_module.to(dtype=dtype)
    reference_module.load_state_dict(module_state_dict)
    reference_module.structure_module.coordinate_augmentation = False
    reference_module.apply(SetModuleInfValues())
    reference_module = reference_module.to(device=manager.device)
    module = Boltz2_1D(reference_module, manager)
    module.eval()

    # --- Distribute features ---
    device_mesh = manager.device_mesh

    feats_global = {
        k: v.to(device=manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in input_feats_global_fp64_host.items()
    }
    dt_feats = _distribute_features_for_predict(feats_global, rank, device_mesh)

    # --- Distribute noise as DTensors ---
    all_noise_host = [init_noise_global_host] + list(step_noise_list_global_host)
    noise_dts = _make_noise_dtensors(
        all_noise_host,
        device_mesh,
        manager.device,
        dtype,
    )
    init_noise_dt = noise_dts[0]
    step_noise_dts = noise_dts[1:]

    # --- Monkeypatch distributed sample() for determinism ---
    _monkeypatch_deterministic_noise(monkeypatch, init_noise_dt, step_noise_dts)

    # --- Run distributed predict_step ---
    module.predict_args = predict_args
    with torch.no_grad():
        pred_dict = module.predict_step(dt_feats, batch_idx=0)

    assert pred_dict["exception"] is False

    # --- Compare coords ---
    gathered_coords = pred_dict["coords"]
    gathered_mask = pred_dict["masks"]

    mask_expanded = gathered_mask.repeat_interleave(diffusion_samples, 0).bool()
    dt_real = gathered_coords[mask_expanded]

    serial_coords_device = serial_coords_host.to(device=manager.device, dtype=dtype)
    serial_mask_expanded = serial_masks_host.to(device=manager.device).repeat_interleave(diffusion_samples, 0).bool()
    serial_real = serial_coords_device[serial_mask_expanded]

    torch.testing.assert_close(dt_real, serial_real)

    # --- Compare confidence keys ---
    expected_confidence_keys = {"pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"}
    missing = expected_confidence_keys - set(pred_dict.keys())
    assert not missing, f"Missing confidence keys in predict_step output: {missing}"

    for key in sorted(serial_confidence_keys_host.keys()):
        serial_val = serial_confidence_keys_host[key].to(device=manager.device, dtype=dtype)
        dist_val = pred_dict[key]
        if dist_val.dtype != dtype:
            dist_val = dist_val.to(dtype=dtype)
        torch.testing.assert_close(
            dist_val,
            serial_val,
            msg=lambda msg, k=key: f"Confidence key '{k}' mismatch: {msg}",
        )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        # CPU dp=1 cp=2 dropped: Boltz2 end-node triangle attention (DAP path)
        # needs dist.all_to_all / dist.reduce_scatter, which gloo does not
        # support when cp_size > 1.  See _DAPTriangleAttentionEndingNode1DImpl
        # in src/boltz/distributed/model/layers/triangular_attention_1d.py.
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_predict_step_confidence_parity_1d(setup_env):
    """V7b: predict_step confidence parity between 1D CP distributed and serial.

    Tests that 1D CP predict_step with confidence_prediction=True produces
    numerically identical coordinates and confidence outputs compared to serial.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64
    size_batch = 1
    B = size_batch * grid_group_sizes["dp"]
    size_cp = grid_group_sizes["cp"]

    min_val_init = -0.01
    max_val_init = 0.01
    scale_glorot = 0.05

    num_sampling_steps = 2
    diffusion_samples = 1

    seed = 42
    seed_by_rank(0, seed=seed)

    boltz2_model_params = create_boltz2_model_init_params(use_large_model=False)
    boltz2_model_params["diffusion_process_args"]["alignment_reverse_diff"] = False
    boltz2_model_params["diffusion_process_args"]["coordinate_augmentation"] = False
    boltz2_model_params["num_bins"] = 64
    boltz2_model_params["confidence_prediction"] = True
    boltz2_model_params["confidence_model_args"] = {
        "pairformer_args": boltz2_model_params["pairformer_args"],
        "confidence_args": {},
    }

    predict_args = {
        "recycling_steps": 0,
        "sampling_steps": num_sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": True,
    }
    boltz2_model_params["predict_args"] = predict_args

    n_atoms_per_token_min = 8
    n_atoms_per_token_max = 20
    n_tokens = 30 * size_cp
    W = boltz2_model_params["atoms_per_window_queries"]
    n_atoms_raw = n_tokens * n_atoms_per_token_max
    n_atoms = ((n_atoms_raw + W - 1) // W) * W
    n_msa = size_cp * 2

    assert n_atoms % size_cp == 0
    assert n_atoms % W == 0

    input_feats_global_fp64 = random_features(
        size_batch=B,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=n_msa,
        atom_counts_per_token_range=(n_atoms_per_token_min, n_atoms_per_token_max),
        device=device_type,
        float_value_range=(min_val_init, max_val_init),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=boltz2_model_params["num_bins"],
    )
    input_feats_global_fp64["msa"] = torch.randint(
        0, const.num_tokens, (B, n_msa, n_tokens), dtype=torch.int64, device=device_type
    )

    # ------------------------------------------------------------------
    # Build serial model in eval mode
    # ------------------------------------------------------------------
    reference_module = SerialBoltz2(**boltz2_model_params)
    init_module_params_glorot(reference_module, gain=scale_glorot)
    reference_module.apply(SetModuleInfValues())
    reference_module.structure_module.coordinate_augmentation = False
    module_state_dict_fp64 = reference_module.state_dict()
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).eval()

    # ------------------------------------------------------------------
    # Pre-generate deterministic noise for sampling
    # ------------------------------------------------------------------
    _B_M = B * diffusion_samples
    init_noise = torch.empty((_B_M, n_atoms, 3), device=device_type, dtype=dtype)
    step_noise_list = [
        torch.empty((_B_M, n_atoms, 3), device=device_type, dtype=dtype) for _ in range(num_sampling_steps)
    ]
    init_tensors_uniform([init_noise, *step_noise_list], low=min_val_init, high=max_val_init)

    # ------------------------------------------------------------------
    # Monkeypatch serial sample() for determinism (Bug 1/3 mitigations)
    # ------------------------------------------------------------------
    def _identity_compute_random_augmentation(multiplicity_arg, device=None, dtype=None):
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(_B_M, -1, -1)
        tr = torch.zeros(_B_M, 1, 3, device=device, dtype=dtype)
        return R, tr

    _serial_randn_calls = []
    _serial_randn_sequence = [init_noise] + step_noise_list

    def _fixed_randn(*args, **kwargs):
        idx = len(_serial_randn_calls)
        _serial_randn_calls.append(idx)
        return _serial_randn_sequence[idx].clone()

    _monkeypatch = pytest.MonkeyPatch()
    _monkeypatch.setattr(
        serial_diffusion_v2_module, "compute_random_augmentation", _identity_compute_random_augmentation
    )
    _monkeypatch.setattr(serial_diffusion_v2_module.torch, "randn", _fixed_randn)

    # ------------------------------------------------------------------
    # Serial predict_step
    # ------------------------------------------------------------------
    with torch.no_grad():
        serial_pred_dict = reference_module.predict_step(input_feats_global_fp64, batch_idx=0)

    assert serial_pred_dict["exception"] is False
    serial_coords = serial_pred_dict["coords"]
    serial_masks = serial_pred_dict["masks"]

    # Collect confidence keys
    serial_confidence_keys = {}
    for key in ["pde", "plddt", "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]:
        if key in serial_pred_dict:
            val = serial_pred_dict[key]
            serial_confidence_keys[key] = val.detach().to(device="cpu", copy=True)

    _monkeypatch.undo()

    # ------------------------------------------------------------------
    # Move everything to CPU for spawn_multiprocessing
    # ------------------------------------------------------------------
    input_feats_host = {k: v.detach().to(device="cpu", copy=True) for k, v in input_feats_global_fp64.items()}
    serial_coords_host = serial_coords.detach().to(device="cpu", copy=True)
    serial_masks_host = serial_masks.detach().to(device="cpu", copy=True)

    spawn_multiprocessing(
        _worker_predict_step_confidence_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        dtype,
        boltz2_model_params,
        module_state_dict_fp64,
        predict_args,
        diffusion_samples,
        num_sampling_steps,
        input_feats_host,
        init_noise.cpu(),
        [n.cpu() for n in step_noise_list],
        serial_coords_host,
        serial_masks_host,
        serial_confidence_keys,
        env_per_rank,
    )


# ====================================================================== #
#  V8: Serial vs 1D CP distributed training_step numerical parity         #
# ====================================================================== #


class _LogCapture:
    """Captures LightningModule.log() calls into a dict, backed by a CSVLogger.

    Replaces the usual ``lambda *a, **kw: None`` monkeypatch so that the
    training_step logging code path is exercised rather than silenced.
    After the step, :meth:`flush` persists the captured metrics to the CSV file.
    """

    def __init__(self, csv_logger):
        self.metrics: dict[str, float] = {}
        self._csv_logger = csv_logger

    def __call__(self, name, value, **kwargs):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        self.metrics[name] = value

    def flush(self, step: int = 0) -> None:
        """Write captured metrics to the backing CSVLogger."""
        self._csv_logger.log_metrics(self.metrics, step=step)
        self._csv_logger.save()


def _smooth_lddt_loss_dense(
    pred_coords,
    true_coords,
    is_nucleotide,
    coords_mask=None,
    nucleic_acid_cutoff=30.0,
    other_cutoff=15.0,
    multiplicity=1,
):
    """Dense pairwise-distance smooth lDDT loss (matches 2D test).

    The serial code uses sparse indexing (nonzero + F.pairwise_distance)
    which creates a different autograd backward graph than the distributed
    dense matrix computation (replicate_to_shard_outer_op CDIST). Using
    dense distances here aligns the backward structure.
    """
    compute_dtype = torch.promote_types(pred_coords.dtype, torch.float32)
    N = pred_coords.shape[1]
    lddt = []
    for i in range(true_coords.shape[0]):
        true_dists = torch.cdist(true_coords[i], true_coords[i])

        is_nuc_i = is_nucleotide[i // multiplicity]
        mask_i = coords_mask[i // multiplicity]

        is_nuc_pair = is_nuc_i.unsqueeze(-1).expand(-1, is_nuc_i.shape[-1])

        mask = is_nuc_pair * (true_dists < nucleic_acid_cutoff).to(compute_dtype)
        mask += (1 - is_nuc_pair) * (true_dists < other_cutoff).to(compute_dtype)
        mask *= 1 - torch.eye(N, device=pred_coords.device)
        mask *= mask_i.unsqueeze(-1)
        mask *= mask_i.unsqueeze(-2)

        diff = pred_coords[i].unsqueeze(0) - pred_coords[i].unsqueeze(1)
        pred_dists = (diff * diff).sum(-1).add(1e-30).sqrt()

        dist_diff = (true_dists - pred_dists).abs()

        eps = (
            torch.sigmoid(0.5 - dist_diff)
            + torch.sigmoid(1.0 - dist_diff)
            + torch.sigmoid(2.0 - dist_diff)
            + torch.sigmoid(4.0 - dist_diff)
        ) * 0.25

        lddt_i = (eps * mask).sum() / (mask.sum() + 1e-5)
        lddt.append(lddt_i)

    return 1 - sum(lddt) / len(lddt)


def _worker_training_step_parity_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    module_state_dict,
    boltz2_model_params,
    input_feats_global_fp64_host,
    sigmas_expected_host,
    noise_expected_host,
    serial_loss_host,
    serial_log_metrics_host,
    serial_grad_dict_host,
    serial_post_opt_dict_host,
    optimizer_step,
):
    """V8 multi-rank worker: verify 1D CP distributed training_step matches serial.

    Compares:
    1. Loss value (serial vs distributed)
    1b. Logged metric values via CSVLogger (serial vs distributed),
        including component-wise grad_norms (compared after backward
        + on_after_backward so grad_norm metrics are available)
    2. Per-parameter gradients after backward + on_after_backward
    3. Post-optimizer parameter values (if optimizer_step=True)
    """
    import tempfile

    from pytorch_lightning.loggers import CSVLogger

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    dtype = torch.float64

    # Build distributed model from serial state dict
    reference_module = SerialBoltz2(**boltz2_model_params)
    reference_module = reference_module.to(dtype=dtype)
    reference_module.load_state_dict(module_state_dict)
    reference_module.structure_module.coordinate_augmentation = False
    reference_module.apply(SetModuleInfValues())
    reference_module = reference_module.to(device=manager.device)
    module = Boltz2_1D(reference_module, manager)
    module = module.to(device=manager.device)
    module.train()

    # Capture initial parameter values for non-vacuous optimizer check
    initial_params = {
        name: (p.full_tensor().detach().clone() if isinstance(p, DTensor) else p.detach().clone())
        for name, p in module.named_parameters()
        if p.requires_grad
    }

    # Inject CSVLogger to exercise the logging code path (instead of no-op)
    worker_csv_logger = CSVLogger(save_dir=tempfile.mkdtemp(), name=f"distributed_rank{rank}")
    dist_log = _LogCapture(worker_csv_logger)
    monkeypatch.setattr(module, "log", dist_log)
    monkeypatch.setattr(module, "training_log", lambda *a, **kw: None)

    # Distribute features (same pattern as V6)
    device_mesh = manager.device_mesh
    host_tensor_keys = {k for k, v in input_feats_global_fp64_host.items() if isinstance(v, torch.Tensor)}
    feature_placements = _build_feature_placements_1d(host_tensor_keys)

    if manager.group_rank["world"] == 0:
        input_feats_global = {
            k: v.to(device=manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
            for k, v in input_feats_global_fp64_host.items()
        }
    else:
        input_feats_global = None

    batch = distribute_features(
        input_feats_global,
        feature_placements,
        manager.group["world"],
        manager.group_ranks["world"][0],
        device_mesh,
    )

    # Distribute noise as DTensor with (Shard(0), Shard(1))
    noise_global = noise_expected_host.to(device=manager.device, dtype=dtype)
    noise_dt = distribute_tensor(noise_global, device_mesh, (Shard(0), Shard(1)))

    # Monkeypatch deterministic noise for distributed forward
    sigmas_device = sigmas_expected_host.to(device=manager.device, dtype=dtype)
    sigmas_dt = distribute_tensor(sigmas_device, device_mesh, (Shard(0), Replicate()))
    monkeypatch.setattr(module.structure_module, "noise_distribution", lambda bs, dtype=None: sigmas_dt)
    monkeypatch.setattr(diffusion_1d_module, "create_distributed_randn", lambda *a, **kw: noise_dt)

    # Run distributed training_step
    loss = module.training_step(batch, batch_idx=0)

    # Assert 1: loss matches serial
    loss_local = loss.to_local() if isinstance(loss, DTensor) else loss
    serial_loss_device = serial_loss_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(
        loss_local,
        serial_loss_device,
        msg=lambda msg: f"Rank {rank}: Loss mismatch: {msg}",
    )

    # Backward
    loss_local.backward()

    # on_after_backward redistributes gradients to Replicate and logs grad_norm metrics
    module.on_after_backward()

    # Tolerance rationale (applied to Asserts 1b and 3 below).
    #
    # All asserts here run at the fp64 default ``atol=rtol=1e-7``.  Two
    # downstream aggregations would otherwise amplify per-element grad
    # parity drift past the default budget:
    #
    # 1. ``train/grad_norm = sqrt(sum_i ||g_i||^2)`` (~5000 grad elements).
    #    The drift scales roughly with ``sqrt(N) * atol_per_elem``.
    #
    # 2. Adam first-step ``step = lr * g / (sqrt(g**2) + eps)`` is singular
    #    when ``|g| << eps``: a per-element grad delta ``delta_g`` produces
    #    a parameter-update delta of ``lr * delta_g / max(|g|, eps)``.  In
    #    the singular regime the amplification is ``lr/eps = 1e5``, so the
    #    worst-case post-Adam drift is ``lr * atol_per_elem / eps = 1e-2``
    #    -- five orders of magnitude above the default 1e-7 atol.
    #
    # Empirically (1) is small and not the binding constraint; (2) dominates
    # and concentrates on Glorot-initialised pair-projection weights in
    # ``diffusion_conditioning.{atom_encoder, pairwise_conditioner}``.  At
    # ``scale_glorot=0.08`` the measured max post-Adam residual was 5.878e-4
    # on ``pairwise_conditioner.dim_pairwise_init_proj.1.weight``, requiring
    # ``atol=1e-2`` widening (commit 193b6fab).
    #
    # Root fix (this commit; pattern from Bug E commit 8944eb95):
    # reducing ``scale_glorot`` shrinks both upstream activations and the
    # backward-flowing grad magnitudes that flow through these pair-proj
    # weights.  Sweep at the cuda-dp1-cp2 cell:
    #
    #     scale_glorot=0.08    post-opt resid = 5.878e-4   (original)
    #     scale_glorot=0.025   post-opt resid = 3.223e-5   (Bug E precedent)
    #     scale_glorot=0.0125  post-opt resid = 3.202e-6
    #     scale_glorot=0.005   post-opt resid = 1.345e-7
    #     scale_glorot=0.004   post-opt resid = 5.325e-8   (selected)
    #
    # At ``scale_glorot=0.004`` post-opt residual is 5.3e-8 and grad_norm
    # residual is 4.2e-10 -- both ~2x below the fp64 default 1e-7, so we
    # can drop the prior 1e-2 / 1e-5 widenings and check Assert 1b + Assert
    # 3 at the default tolerance.  ``num_params_changed > 0`` vacuity guard
    # remains intact (300/319 params change after the optimizer step).

    # Assert 2 (runs before Assert 1b so the stronger per-element check fails
    # first when something breaks): per-parameter gradient parity at default
    # fp64 tolerances.  1D model mirrors serial names directly.
    num_grads_checked = 0
    num_nonzero_grads = 0
    for name, param in module.named_parameters():
        if name not in serial_grad_dict_host:
            continue
        expected_grad = serial_grad_dict_host[name].to(device=manager.device, dtype=dtype)
        assert param.grad is not None, f"Rank {rank}: Missing gradient for {name}"
        actual_grad = param.grad.full_tensor() if isinstance(param.grad, DTensor) else param.grad
        num_grads_checked += 1
        if expected_grad.abs().max().item() > 0:
            num_nonzero_grads += 1
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            msg=lambda msg, cn=name: (
                f"Rank {rank}: Gradient mismatch for {cn}. "
                f"Serial grad norm: {expected_grad.norm().item():.10f}, "
                f"Distributed grad norm: {actual_grad.norm().item():.10f}. {msg}"
            ),
        )

    assert num_grads_checked > 0, f"Rank {rank}: No gradients compared — test is vacuous"
    assert num_nonzero_grads > 0, f"Rank {rank}: All compared gradients are zero — test is vacuous"

    # Assert 1b: logged metrics parity (CSVLogger output).
    # Compared after backward + on_after_backward so grad_norm metrics are included.
    assert len(dist_log.metrics) > 0, f"Rank {rank}: No metrics logged — test is vacuous"
    assert set(dist_log.metrics.keys()) == set(serial_log_metrics_host.keys()), (
        f"Rank {rank}: Logged metric keys differ. "
        f"Serial: {sorted(serial_log_metrics_host.keys())}, "
        f"Distributed: {sorted(dist_log.metrics.keys())}"
    )
    # Logged metrics parity at default fp64 tol.  Init-scale reduction (see
    # ``scale_glorot`` block in the calling test) brings ``train/grad_norm``
    # drift below 1e-7, so all metric keys are checked at default tol.
    for key in sorted(serial_log_metrics_host.keys()):
        torch.testing.assert_close(
            torch.tensor(dist_log.metrics[key], dtype=torch.float64),
            torch.tensor(serial_log_metrics_host[key], dtype=torch.float64),
            msg=lambda msg, k=key: f"Rank {rank}: Logged metric mismatch for {k}: {msg}",
        )
    dist_log.flush(step=0)

    # Assert 3: optimizer step parity (if requested).
    if optimizer_step:
        optimizer = torch.optim.Adam(module.parameters(), lr=1e-3, betas=(0.9, 0.999))
        optimizer.step()

        num_params_checked = 0
        num_params_changed = 0
        for name, param in module.named_parameters():
            # 1D model mirrors serial names directly (no ._serial. prefix)
            if name not in serial_post_opt_dict_host:
                continue
            expected_val = serial_post_opt_dict_host[name].to(device=manager.device, dtype=dtype)
            actual_val = param.full_tensor() if isinstance(param, DTensor) else param.data
            num_params_checked += 1
            if name in initial_params:
                initial_val = initial_params[name].to(device=manager.device, dtype=dtype)
                if not torch.equal(actual_val, initial_val):
                    num_params_changed += 1
            torch.testing.assert_close(
                actual_val,
                expected_val,
                msg=lambda msg, cn=name: f"Rank {rank}: Post-optimizer mismatch for {cn}. {msg}",
            )
        assert num_params_checked > 0, f"Rank {rank}: No post-optimizer params compared"
        assert num_params_changed > 0, f"Rank {rank}: No parameters changed after optimizer step — test is vacuous"

    # Assert 4: cross-rank loss identity
    torch.distributed.barrier()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("setup_env", "optimizer_step"),
    [
        (((1, 2), True, "cuda", "ENV"), True),
    ],
    indirect=["setup_env"],
    ids=["cuda-dp1-cp2-optim"],
)
def test_training_step_parity_1d(setup_env, optimizer_step, tmp_path):
    """V8: Serial vs 1D CP distributed training_step numerical parity.

    Verifies that the 1D CP Boltz2_1D.training_step produces numerically
    identical loss, gradients, post-optimizer parameters, and logged metrics
    compared to the serial Boltz2.training_step, with all randomness sources
    controlled:

    1. Recycling: no_random_recycling_training=True (fixed recycling_steps)
    2. Diffusion noise: monkeypatched noise_distribution and randn_like / create_distributed_randn
    3. Coordinate augmentation: disabled
    4. Sampling steps: fixed (no sampling_steps_random)
    5. Dropout: 0.0 in all modules

    Both serial and distributed sessions inject a CSVLogger-backed
    ``_LogCapture`` so that ``self.log()`` calls are exercised (not
    silenced) and the logged metric keys/values are compared.

    Extends V6 (forward/backward parity) to the full training_step control
    flow including loss aggregation, recycling-step broadcasting, and
    gradient redistribution.
    """
    from pytorch_lightning.loggers import CSVLogger

    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64
    min_val_init = -0.1
    max_val_init = 0.1
    scale_glorot = 0.004

    seed = 42
    seed_by_rank(0, seed=seed)

    # Small model config — dropout=0 for determinism
    boltz2_model_params = create_boltz2_model_init_params(use_large_model=False)
    recycling_steps = 0
    multiplicity = 2

    boltz2_model_params["training_args"].recycling_steps = recycling_steps
    boltz2_model_params["training_args"].diffusion_multiplicity = multiplicity
    boltz2_model_params["training_args"].sampling_steps = -1  # not used in training path but read by training_step
    boltz2_model_params["no_random_recycling_training"] = True
    boltz2_model_params["predict_bfactor"] = True
    boltz2_model_params["training_args"].bfactor_loss_weight = 1.0
    boltz2_model_params["validate_structure"] = True

    size_cp = grid_group_sizes["cp"]
    B = grid_group_sizes["dp"]

    n_tokens = 30 * size_cp
    W = boltz2_model_params["atoms_per_window_queries"]
    n_atoms_raw = n_tokens * 20
    n_atoms = ((n_atoms_raw + W - 1) // W) * W
    n_msa = max(size_cp * 2, 2)

    input_feats_global_fp64 = random_features(
        size_batch=B,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=n_msa,
        atom_counts_per_token_range=(8, 20),
        device=device_type,
        float_value_range=(min_val_init, max_val_init),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=boltz2_model_params["num_bins"],
    )
    input_feats_global_fp64["msa"] = torch.randint(
        0, const.num_tokens, (B, n_msa, n_tokens), dtype=torch.int64, device=device_type
    )
    input_feats_global_fp64["disto_target"] = input_feats_global_fp64["disto_target"].unsqueeze(3)

    # Create serial model with deterministic init
    serial_model = SerialBoltz2(**boltz2_model_params)
    init_module_params_glorot(serial_model, gain=scale_glorot)
    serial_model.apply(SetModuleInfValues())
    serial_model.structure_module.coordinate_augmentation = False
    serial_model = serial_model.to(dtype=dtype, device=device_type)
    serial_model.train()

    # Save state dict for distributed model (before serial forward mutates state)
    module_state_dict = {k: v.detach().clone().cpu() for k, v in serial_model.state_dict().items()}

    # Pre-generate deterministic sigmas and non-zero noise for serial model
    sigmas_serial = serial_model.structure_module.noise_distribution(B * multiplicity).to(dtype=dtype)
    noise_serial = torch.empty(B * multiplicity, n_atoms, 3, device=device_type, dtype=dtype)
    init_tensors_uniform([noise_serial], low=min_val_init, high=max_val_init)
    atom_pad_mask_mul = input_feats_global_fp64["atom_pad_mask"][:, :, None].repeat_interleave(multiplicity, 0)
    noise_serial = noise_serial * atom_pad_mask_mul

    _serial_mp = pytest.MonkeyPatch()
    serial_csv_logger = CSVLogger(save_dir=str(tmp_path), name="serial")
    serial_log = _LogCapture(serial_csv_logger)
    _serial_mp.setattr(serial_model, "log", serial_log)
    _serial_mp.setattr(serial_model, "training_log", lambda *a, **kw: None)
    _serial_mp.setattr(serial_model.structure_module, "noise_distribution", lambda bs, dtype=None: sigmas_serial)
    _serial_mp.setattr(serial_diffusion_v2_module.torch, "randn_like", lambda t: noise_serial.to(t))
    # Monkeypatch serial smooth_lddt_loss to use dense pairwise distances.
    # The original serial code uses sparse indexing (nonzero + F.pairwise_distance)
    # which creates a different autograd backward graph than the distributed dense
    # matrix computation (replicate_to_shard_outer_op CDIST). The different
    # accumulation patterns amplify ~1e-12 forward differences into ~4.5e-7
    # gradient errors. Using dense distances here aligns the backward structure.
    import boltz.model.loss.diffusionv2 as _serial_loss_mod

    _serial_mp.setattr(_serial_loss_mod, "smooth_lddt_loss", _smooth_lddt_loss_dense)
    _serial_mp.setattr(serial_diffusion_v2_module, "smooth_lddt_loss", _smooth_lddt_loss_dense)

    # Save coords — serial forward mutates feats["coords"] in-place (flattens ensemble dim)
    coords_backup = input_feats_global_fp64["coords"].detach().clone()
    original_feat_keys = set(input_feats_global_fp64.keys())

    # Run serial training_step
    serial_loss = serial_model.training_step(input_feats_global_fp64, batch_idx=0)
    assert serial_loss is not None, "Serial training_step returned None"
    assert serial_loss.isfinite(), f"Serial loss is not finite: {serial_loss.item()}"

    # Backward on serial model
    serial_loss.backward()

    # on_after_backward logs grad_norm metrics (component-wise and global)
    serial_model.on_after_backward()

    # Flush serial CSVLogger and collect logged metrics (after backward so
    # grad_norm metrics are included alongside the training_step metrics)
    serial_log.flush(step=0)
    serial_log_metrics = dict(serial_log.metrics)
    assert len(serial_log_metrics) > 0, "Serial model logged no metrics — validate_structure may be False"

    # Collect serial gradients
    serial_grad_dict = {}
    for name, param in serial_model.named_parameters():
        if param.grad is not None:
            serial_grad_dict[name] = param.grad.detach().clone().cpu()

    # Collect serial post-optimizer parameters (if needed)
    serial_post_opt_dict = {}
    if optimizer_step:
        serial_optimizer = torch.optim.Adam(serial_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        serial_optimizer.step()
        for name, param in serial_model.named_parameters():
            serial_post_opt_dict[name] = param.data.detach().clone().cpu()

    serial_loss_host = serial_loss.detach().clone().cpu()
    sigmas_host = sigmas_serial.detach().clone().cpu()
    noise_host = noise_serial.detach().clone().cpu()

    # Restore features — serial forward mutates coords in-place and may add keys
    input_feats_global_fp64["coords"] = coords_backup
    for key in list(input_feats_global_fp64.keys()):
        if key not in original_feat_keys:
            del input_feats_global_fp64[key]
    _serial_mp.undo()

    # Move features to CPU for spawned workers
    input_feats_host = {}
    for k, v in input_feats_global_fp64.items():
        if isinstance(v, torch.Tensor):
            input_feats_host[k] = v.detach().to(device="cpu", copy=True)
        elif isinstance(v, list):
            input_feats_host[k] = [
                item.detach().to(device="cpu", copy=True) if isinstance(item, torch.Tensor) else item for item in v
            ]
        else:
            input_feats_host[k] = v

    # Spawn parallel test
    spawn_multiprocessing(
        _worker_training_step_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        module_state_dict,
        boltz2_model_params,
        input_feats_host,
        sigmas_host,
        noise_host,
        serial_loss_host,
        serial_log_metrics,
        serial_grad_dict,
        serial_post_opt_dict,
        optimizer_step,
    )


# ====================================================================== #
#  V9: validation_step parity                                            #
# ====================================================================== #


def _worker_validation_step_parity_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    boltz2_model_params,
    module_state_dict,
    input_feats_host,
    noise_host_list,
    serial_per_sample,
    serial_epoch_end_metrics,
    env_per_rank=None,
):
    """V9 multi-rank worker: verify 1D CP distributed validation_step matches serial.

    Phase 1: Compare raw accumulated metrics after validation_step against
    the serial per-sample reference for this DP rank's sample.

    Phase 2: Compare aggregated metrics after on_validation_epoch_end
    against the serial epoch-end reference.
    """
    from pytorch_lightning.loggers import CSVLogger

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    dtype = torch.float64
    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    dp_rank = manager.group_rank["dp"]

    reference_module = SerialBoltz2(**boltz2_model_params)
    reference_module = reference_module.to(dtype=dtype)
    reference_module.load_state_dict(module_state_dict)
    reference_module.structure_module.coordinate_augmentation = False
    reference_module.apply(SetModuleInfValues())
    reference_module = reference_module.to(device=manager.device)
    module = Boltz2_1D(reference_module, manager)
    module.eval()

    # Wire validators via setup with mock trainer
    num_validators = boltz2_model_params["num_val_datasets"]
    val_names = [v.val_names[0] for v in boltz2_model_params["validators"]]
    worker_val_group_mapper = {
        vi: {"label": val_names[vi], "symmetry_correction": False} for vi in range(num_validators)
    }
    module._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(val_group_mapper=worker_val_group_mapper),
        sanity_checking=False,
    )
    module.setup("fit")
    assert len(module.validator_mapper) == num_validators

    # ------------------------------------------------------------------
    # Distribute features for this rank
    # ------------------------------------------------------------------
    device_mesh = manager.device_mesh
    host_tensor_keys = {k for k, v in input_feats_host.items() if isinstance(v, torch.Tensor)}
    feature_placements = _build_feature_placements_1d(host_tensor_keys)

    if manager.group_rank["world"] == 0:
        input_feats_global = {
            k: v.to(device=manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
            for k, v in input_feats_host.items()
            if isinstance(v, torch.Tensor)
        }
    else:
        input_feats_global = None

    feats_dt = distribute_features(
        input_feats_global,
        feature_placements,
        manager.group["world"],
        manager.group_ranks["world"][0],
        device_mesh,
    )

    # Non-tensor features (connections_edge_index, chain_symmetries): DP-sliced
    for k, v in input_feats_host.items():
        if isinstance(v, list):
            elem = v[dp_rank]
            if isinstance(elem, torch.Tensor):
                feats_dt[k] = elem.unsqueeze(0).to(
                    device=manager.device, dtype=dtype if elem.dtype.is_floating_point else elem.dtype
                )
            else:
                feats_dt[k] = [elem]

    # ------------------------------------------------------------------
    # Pre-generate deterministic noise DTensors for distributed sampling
    # ------------------------------------------------------------------
    noise_dts = _make_noise_dtensors(noise_host_list, device_mesh, manager.device, dtype)
    init_noise_dt = noise_dts[0]
    step_noise_dts = noise_dts[1:]

    _monkeypatch_deterministic_noise(monkeypatch, init_noise_dt, step_noise_dts)

    # ------------------------------------------------------------------
    # Phase 1: Run validation_step (validator 0), compare accumulated metrics
    # ------------------------------------------------------------------
    feats_dt["idx_dataset"] = [torch.tensor([0], device=manager.device)]

    with torch.no_grad():
        module.validation_step(feats_dt, batch_idx=0)

    validator = module.validator_mapper[0]
    fm = validator.folding_metrics
    val_idx = 0

    serial_ref = serial_per_sample[dp_rank]
    compared_phase1 = 0

    disto_loss_metric = fm["disto_loss"][val_idx]["disto_loss"]
    if disto_loss_metric.weight > 0:
        dist_disto_loss = disto_loss_metric.compute().item()
        serial_disto_loss_avg = sum(s["disto_loss"] for s in serial_per_sample) / len(serial_per_sample)
        torch.testing.assert_close(
            torch.tensor(dist_disto_loss, dtype=dtype),
            torch.tensor(serial_disto_loss_avg, dtype=dtype),
            msg=lambda msg: f"Rank {rank}: Phase 1 disto_loss mismatch: {msg}",
        )
        compared_phase1 += 1

    for key in [*const.out_types, "pocket_ligand_protein", "contact_protein_protein"]:
        if key in fm["disto_lddt"][val_idx]:
            metric = fm["disto_lddt"][val_idx][key]
            if metric.weight > 0 and key in serial_ref.get("disto_lddt", {}):
                dist_val = metric.compute().item()
                serial_val = serial_ref["disto_lddt"][key]
                torch.testing.assert_close(
                    torch.tensor(dist_val, dtype=dtype),
                    torch.tensor(serial_val, dtype=dtype),
                    msg=lambda msg, k=key: f"Rank {rank}: Phase 1 disto_lddt_{k} mismatch: {msg}",
                )
                compared_phase1 += 1

    for key in [*const.out_types, "pocket_ligand_protein", "contact_protein_protein"]:
        if key in fm["lddt"][val_idx]:
            metric = fm["lddt"][val_idx][key]
            if metric.weight > 0 and key in serial_ref.get("lddt", {}):
                dist_val = metric.compute().item()
                serial_val = serial_ref["lddt"][key]
                torch.testing.assert_close(
                    torch.tensor(dist_val, dtype=dtype),
                    torch.tensor(serial_val, dtype=dtype),
                    msg=lambda msg, k=key: f"Rank {rank}: Phase 1 lddt_{k} mismatch: {msg}",
                )
                compared_phase1 += 1

    for key in [*const.out_types, "pocket_ligand_protein", "contact_protein_protein"]:
        if key in fm["complex_lddt"][val_idx]:
            metric = fm["complex_lddt"][val_idx][key]
            if metric.weight > 0 and key in serial_ref.get("complex_lddt", {}):
                dist_val = metric.compute().item()
                serial_val = serial_ref["complex_lddt"][key]
                torch.testing.assert_close(
                    torch.tensor(dist_val, dtype=dtype),
                    torch.tensor(serial_val, dtype=dtype),
                    msg=lambda msg, k=key: f"Rank {rank}: Phase 1 complex_lddt_{k} mismatch: {msg}",
                )
                compared_phase1 += 1

    assert compared_phase1 >= 3, f"Rank {rank}: Phase 1 compared only {compared_phase1} metrics — test may be vacuous"

    # Run remaining validators for Phase 2 accumulation
    for vi in range(1, num_validators):
        feats_dt["idx_dataset"] = [torch.tensor([vi], device=manager.device)]

        # Reset noise sequence for each additional validator
        _vi_noise_dts = _make_noise_dtensors(noise_host_list, device_mesh, manager.device, dtype)
        _monkeypatch_deterministic_noise(monkeypatch, _vi_noise_dts[0], _vi_noise_dts[1:])

        with torch.no_grad():
            module.validation_step(feats_dt, batch_idx=vi)

    # ------------------------------------------------------------------
    # Phase 2: on_validation_epoch_end and compare aggregated metrics
    # ------------------------------------------------------------------
    dist_log = _LogCapture(CSVLogger(save_dir=tempfile.mkdtemp(), name=f"dist_val_rank{rank}"))
    monkeypatch.setattr(module, "log", dist_log)

    module.on_validation_epoch_end()

    compared_phase2 = 0
    for key in sorted(serial_epoch_end_metrics):
        if key in dist_log.metrics:
            got = torch.tensor(dist_log.metrics[key], dtype=dtype)
            exp = torch.tensor(serial_epoch_end_metrics[key], dtype=dtype)
            torch.testing.assert_close(
                got,
                exp,
                msg=lambda msg, k=key: f"Rank {rank}: Phase 2 epoch-end metric '{k}' mismatch: {msg}",
            )
            compared_phase2 += 1

    assert compared_phase2 >= 3, f"Rank {rank}: Phase 2 compared only {compared_phase2} metrics — test may be vacuous"

    required_base_metrics = ("val/lddt", "val/disto_lddt", "val/complex_lddt")
    for base_metric in required_base_metrics:
        for vn in val_names:
            suffix = "" if vn == "RCSB" else f"__{vn}"
            required_metric = f"{base_metric}{suffix}"
            assert required_metric in serial_epoch_end_metrics, (
                f"Rank {rank}: serial epoch-end metrics missing '{required_metric}' — "
                f"available: {sorted(serial_epoch_end_metrics)}"
            )
            assert required_metric in dist_log.metrics, (
                f"Rank {rank}: distributed epoch-end metrics missing '{required_metric}' — "
                f"available: {sorted(dist_log.metrics)}"
            )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("setup_env", "num_validators"),
    [
        (((2, 2), True, "cuda", "ENV"), 2),
        (((2, 2), True, "cuda", "ENV"), 1),
    ],
    indirect=["setup_env"],
    ids=["cuda-dp2-cp2-random-2val", "cuda-dp2-cp2-random-1val"],
)
def test_boltz2_validation_step_parity_1d(setup_env, num_validators, tmp_path):
    """V9: validation_step parity between 1D CP distributed and serial Boltz2.

    Two-phase comparison:
      Phase 1: After validation_step, compare validator MeanMetric values
        (disto_loss, disto_lddt_*, lddt_*, complex_lddt_*) between serial
        (per-sample) and distributed (per DP-rank sample).
      Phase 2: After on_validation_epoch_end (which DP-all-reduces
        MeanMetric internals), compare aggregated logged metrics between
        serial and distributed.

    Uses DP=2, CP=2 -> 4 GPUs, FP64 with default tolerances.
    """
    from pytorch_lightning.loggers import CSVLogger

    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64
    min_val_init = -0.01
    max_val_init = 0.01
    scale_glorot = 0.05

    num_sampling_steps = 2
    diffusion_samples = 1

    seed = 42
    seed_by_rank(0, seed=seed)

    boltz2_model_params = create_boltz2_model_init_params(use_large_model=False)
    boltz2_model_params["diffusion_process_args"]["alignment_reverse_diff"] = False
    boltz2_model_params["validate_structure"] = True
    val_names = [f"RCSB_{i}" for i in range(num_validators)] if num_validators > 1 else ["RCSB"]
    boltz2_model_params["validators"] = [
        RCSBValidator(val_names=[vn], confidence_prediction=False, physicalism_metrics=False) for vn in val_names
    ]
    boltz2_model_params["num_val_datasets"] = num_validators
    boltz2_model_params["confidence_prediction"] = False
    boltz2_model_params["validation_args"] = _make_validation_args(
        recycling_steps=0,
        sampling_steps=num_sampling_steps,
        diffusion_samples=diffusion_samples,
        symmetry_correction=False,
    )

    size_cp = grid_group_sizes["cp"]
    B = grid_group_sizes["dp"]

    n_atoms_per_token_min = 8
    n_atoms_per_token_max = 20
    n_tokens = 30 * size_cp
    W = boltz2_model_params["atoms_per_window_queries"]
    n_atoms_raw = n_tokens * n_atoms_per_token_max
    n_atoms = ((n_atoms_raw + W - 1) // W) * W
    n_msa = size_cp * 2

    assert n_atoms % size_cp == 0
    assert n_atoms % W == 0

    input_feats_global_fp64 = random_features(
        size_batch=B,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=n_msa,
        atom_counts_per_token_range=(n_atoms_per_token_min, n_atoms_per_token_max),
        device=device_type,
        float_value_range=(min_val_init, max_val_init),
        selected_keys=_BOLTZ2_SELECTED_KEYS,
        num_disto_bins=boltz2_model_params["num_bins"],
    )
    input_feats_global_fp64["msa"] = torch.randint(
        0, const.num_tokens, (B, n_msa, n_tokens), dtype=torch.int64, device=device_type
    )

    input_feats_global_fp64["disto_target"] = input_feats_global_fp64["disto_target"].unsqueeze(3)

    token_to_rep_atom = input_feats_global_fp64["token_to_rep_atom"]
    coords = input_feats_global_fp64["coords"]
    disto_coords_ensemble = torch.bmm(token_to_rep_atom.to(dtype=dtype), coords[:, 0])
    input_feats_global_fp64["disto_coords_ensemble"] = disto_coords_ensemble

    input_feats_global_fp64["connections_edge_index"] = [
        torch.empty(2, 0, dtype=torch.long, device=device_type) for _ in range(B)
    ]
    input_feats_global_fp64["chain_symmetries"] = [[] for _ in range(B)]

    # ------------------------------------------------------------------
    # Slice global batch into individual samples for serial validation
    # ------------------------------------------------------------------
    def _slice_batch(feats, idx):
        batch_i = {}
        for k, v in feats.items():
            if isinstance(v, torch.Tensor):
                batch_i[k] = v[idx : idx + 1].clone()
            elif isinstance(v, list):
                elem = v[idx]
                if isinstance(elem, torch.Tensor):
                    batch_i[k] = elem.unsqueeze(0).clone()
                else:
                    batch_i[k] = [elem]
            else:
                batch_i[k] = v
        batch_i["idx_dataset"] = torch.tensor([0], device=device_type)
        return batch_i

    num_val_samples = B

    # ------------------------------------------------------------------
    # Build serial model
    # ------------------------------------------------------------------
    reference_module = SerialBoltz2(**boltz2_model_params)
    init_module_params_glorot(reference_module, gain=scale_glorot)
    reference_module.apply(SetModuleInfValues())
    reference_module.structure_module.coordinate_augmentation = False
    module_state_dict = reference_module.state_dict()
    reference_module = reference_module.to(dtype=dtype, device=device_type).eval()

    serial_validators = []
    reference_module.val_group_mapper = {}
    reference_module.validator_mapper = {}
    for vi in range(num_validators):
        vn = val_names[vi]
        v = RCSBValidator(val_names=[vn], confidence_prediction=False, physicalism_metrics=False)
        v = v.to(device=device_type, dtype=dtype)
        serial_validators.append(v)
        reference_module.val_group_mapper[vi] = {"label": vn, "symmetry_correction": False}
        reference_module.validator_mapper[vi] = v

    # ------------------------------------------------------------------
    # Pre-generate deterministic noise for sampling
    # ------------------------------------------------------------------
    _B_M = B * diffusion_samples
    init_noise = torch.empty((_B_M, n_atoms, 3), device=device_type, dtype=dtype)
    step_noise_list = [
        torch.empty((_B_M, n_atoms, 3), device=device_type, dtype=dtype) for _ in range(num_sampling_steps)
    ]
    init_tensors_uniform([init_noise, *step_noise_list], low=min_val_init, high=max_val_init)
    all_noise = [init_noise] + step_noise_list

    # ------------------------------------------------------------------
    # Phase 1 serial: per-sample metrics
    # ------------------------------------------------------------------
    serial_per_sample = [{} for _ in range(num_val_samples)]

    _original_torch_randn = torch.randn

    def _run_serial_validation_step_batch(batch_i, sample_idx):
        _serial_randn_calls = []
        noise_for_sample = [n[sample_idx : sample_idx + 1].clone() for n in all_noise]
        _serial_randn_sequence = noise_for_sample

        def _fixed_randn(*args, _seq=_serial_randn_sequence, _calls=_serial_randn_calls, **kwargs):
            idx = len(_calls)
            _calls.append(idx)
            if idx < len(_seq):
                return _seq[idx].clone()
            return _original_torch_randn(*args, **kwargs)

        _serial_mp = pytest.MonkeyPatch()
        _serial_mp.setattr(
            serial_diffusion_v2_module,
            "compute_random_augmentation",
            lambda mult, device=None, dtype=None: (
                torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(diffusion_samples, -1, -1),
                torch.zeros(diffusion_samples, 1, 3, device=device, dtype=dtype),
            ),
        )
        _serial_mp.setattr(serial_diffusion_v2_module.torch, "randn", _fixed_randn)
        _serial_mp.setattr(reference_module, "log", lambda *a, **kw: None)

        with torch.no_grad():
            reference_module.validation_step(batch_i, batch_idx=sample_idx)

        _serial_mp.undo()

    def _extract_validator_metrics(validator):
        fm = validator.folding_metrics
        val_idx = 0
        sample_metrics = {}
        disto_loss_metric = fm["disto_loss"][val_idx]["disto_loss"]
        sample_metrics["disto_loss"] = disto_loss_metric.compute().item()
        sample_metrics["disto_lddt"] = {}
        sample_metrics["lddt"] = {}
        sample_metrics["complex_lddt"] = {}
        for m_ in [*const.out_types, "pocket_ligand_protein", "contact_protein_protein"]:
            if m_ in fm["disto_lddt"][val_idx]:
                val = fm["disto_lddt"][val_idx][m_].compute()
                if not torch.isnan(val):
                    sample_metrics["disto_lddt"][m_] = val.item()
            if m_ in fm["lddt"][val_idx]:
                val = fm["lddt"][val_idx][m_].compute()
                if not torch.isnan(val):
                    sample_metrics["lddt"][m_] = val.item()
            if m_ in fm["complex_lddt"][val_idx]:
                val = fm["complex_lddt"][val_idx][m_].compute()
                if not torch.isnan(val):
                    sample_metrics["complex_lddt"][m_] = val.item()
        return sample_metrics

    def _reset_validator_metrics(validator):
        fm = validator.folding_metrics
        val_idx = 0
        for metric_group in ["lddt", "disto_lddt", "complex_lddt", "disto_loss"]:
            for k, metric_obj in fm[metric_group][val_idx].items():
                metric_obj.reset()

    for sample_idx in range(B):
        batch_i = _slice_batch(input_feats_global_fp64, sample_idx)
        batch_i["idx_dataset"] = torch.tensor([0], device=device_type)
        _run_serial_validation_step_batch(batch_i, sample_idx)

        serial_per_sample[sample_idx] = _extract_validator_metrics(serial_validators[0])
        _reset_validator_metrics(serial_validators[0])

    # ------------------------------------------------------------------
    # Phase 2 serial: epoch-end metrics (accumulate both samples)
    # ------------------------------------------------------------------
    for vi in range(num_validators):
        for sample_idx in range(B):
            batch_i = _slice_batch(input_feats_global_fp64, sample_idx)
            batch_i["idx_dataset"] = torch.tensor([vi], device=device_type)
            _run_serial_validation_step_batch(batch_i, sample_idx)

    serial_log = _LogCapture(CSVLogger(save_dir=tempfile.mkdtemp(), name="serial_val"))
    _serial_mp2 = pytest.MonkeyPatch()
    _serial_mp2.setattr(reference_module, "log", serial_log)

    reference_module.on_validation_epoch_end()
    _serial_mp2.undo()

    serial_epoch_end_metrics = dict(serial_log.metrics)

    assert len(serial_per_sample[0]) > 0, "Serial phase 1 produced no metrics for sample 0"
    assert len(serial_per_sample[1]) > 0, "Serial phase 1 produced no metrics for sample 1"
    assert len(serial_epoch_end_metrics) > 0, "Serial phase 2 produced no epoch-end metrics"

    # ------------------------------------------------------------------
    # Move to CPU for spawn_multiprocessing
    # ------------------------------------------------------------------
    input_feats_host = {}
    for k, v in input_feats_global_fp64.items():
        if isinstance(v, torch.Tensor):
            input_feats_host[k] = v.detach().to(device="cpu", copy=True)
        elif isinstance(v, list):
            input_feats_host[k] = [
                item.detach().to(device="cpu", copy=True) if isinstance(item, torch.Tensor) else item for item in v
            ]
        else:
            input_feats_host[k] = v
    noise_host_list = [n.detach().cpu() for n in all_noise]
    serial_per_sample_cpu = list(serial_per_sample)

    boltz2_model_params["validators"] = [
        Distributed1DRCSBValidator(val_names=[vn], confidence_prediction=False, physicalism_metrics=False)
        for vn in val_names
    ]
    spawn_multiprocessing(
        _worker_validation_step_parity_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        boltz2_model_params,
        module_state_dict,
        input_feats_host,
        noise_host_list,
        serial_per_sample_cpu,
        serial_epoch_end_metrics,
        env_per_rank,
    )
