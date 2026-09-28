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

"""1D CP coverage for checkpoint conversion and module-traversal utilities.

Tests the ``convert_serial_checkpoint_to_distributed_state_dict`` /
``convert_distributed_checkpoint_to_serial_state_dict`` helpers with
``Shard`` placements on a 2D ``(dp, cp)`` mesh, and the
``SetTriAttnBackend`` / ``SetAttnPairBiasBackend`` module-traversal
callables that explicitly enumerate the 1D submodule types
``PairformerLayer1D`` and ``AttentionPairBias1D``.
"""

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.modules.utils import (
    SDPAWithBiasBackend,
    SetAttnPairBiasBackend,
    SetTriAttnBackend,
    TriAttnBackend,
    convert_distributed_checkpoint_to_serial_state_dict,
    convert_serial_checkpoint_to_distributed_state_dict,
    has_dtensors,
)
from boltz.testing.utils import create_boltz2_model_init_params, spawn_multiprocessing
from tests.distributed.model.modules._test_utils import _find_free_port

# ---------------------------------------------------------------------------
# Tests A & B — Sharded checkpoint conversion on 1D mesh
# ---------------------------------------------------------------------------


def _parallel_assert_sharded_conversion_1d(
    rank: int,
    payload: tuple,
) -> None:
    """Multi-rank worker: verify Shard-placement checkpoint conversion on 1D mesh.

    Covers both the save leg (convert_dtensors → plain tensors) and the load
    leg (plain tensors → DTensor via distribute_tensor), including a full
    round-trip regression guard.
    """
    grid_group_sizes, device_type, backend, env_per_rank, serial_weight, serial_bias, placements, shard_dim = payload

    monkeypatch = pytest.MonkeyPatch()
    for var_name, value in env_per_rank.items():
        monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    DistributedManager._state = {}
    try:
        DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
        manager = DistributedManager()

        # Build a template with 1D Shard placements on the 2D (dp, cp) mesh.
        template_weight = distribute_tensor(
            torch.zeros_like(serial_weight, device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements,
        )
        template_bias = torch.zeros_like(serial_bias, device=manager.device)
        template = {
            "layer.weight": template_weight,
            "layer.bias": template_bias,
        }
        checkpoint = {
            "state_dict": {
                "layer.weight": serial_weight.clone(),
                "layer.bias": serial_bias.clone(),
            }
        }

        converted = convert_serial_checkpoint_to_distributed_state_dict(
            checkpoint=checkpoint,
            strict=True,
            state_dict_template=template,
        )

        weight_dtensor = converted["layer.weight"]

        # (i) Must be a DTensor with the expected placements.
        assert isinstance(weight_dtensor, DTensor), f"rank {rank}: expected DTensor, got {type(weight_dtensor)}"
        assert weight_dtensor.placements == template_weight.placements, (
            f"rank {rank}: placements mismatch — "
            f"got {weight_dtensor.placements}, expected {template_weight.placements}"
        )

        # (i cont.) Sharding must be active: local shape < global shape on sharded dim.
        local_shape = weight_dtensor.to_local().shape
        full_shape = weight_dtensor.full_tensor().shape
        assert local_shape[shard_dim] < full_shape[shard_dim], (
            f"rank {rank}: sharding not active on dim {shard_dim} — " f"local={local_shape}, full={full_shape}"
        )

        # (ii) full_tensor() must equal the original serial tensor.
        torch.testing.assert_close(weight_dtensor.full_tensor().cpu(), serial_weight)

        # (iii) Local shard must match the manually-computed slice.
        cp_rank = manager.group_rank["cp"]
        cp_size = len(manager.group_ranks["cp"])
        expected_local = torch.chunk(serial_weight, cp_size, dim=shard_dim)[cp_rank]
        torch.testing.assert_close(weight_dtensor.to_local().cpu(), expected_local)

        # (iii cont.) local_shard != full_tensor() — guards against silent replication.
        assert not torch.equal(
            weight_dtensor.to_local().cpu(), serial_weight
        ), f"rank {rank}: local shard equals full tensor — sharding silently replicated"

        # Bias is plain (Replicate path) — values unchanged.
        torch.testing.assert_close(converted["layer.bias"].cpu(), serial_bias)

        # (iv) Round-trip back to plain tensors.
        serialized = convert_distributed_checkpoint_to_serial_state_dict({"state_dict": converted})
        assert not has_dtensors(serialized), "Serialized checkpoint must not contain DTensors"
        torch.testing.assert_close(serialized["layer.weight"], serial_weight)
        torch.testing.assert_close(serialized["layer.bias"], serial_bias)

        # (v) Reload: serial → DTensor must preserve placements and local shard.
        roundtrip = convert_serial_checkpoint_to_distributed_state_dict(
            checkpoint={"state_dict": serialized},
            strict=True,
            state_dict_template=template,
        )
        roundtrip_weight = roundtrip["layer.weight"]
        assert isinstance(roundtrip_weight, DTensor)
        assert roundtrip_weight.placements == template_weight.placements
        torch.testing.assert_close(roundtrip_weight.full_tensor().cpu(), serial_weight)
        torch.testing.assert_close(roundtrip_weight.to_local().cpu(), expected_local)
    finally:
        DistributedManager.cleanup()
        DistributedManager._state = {}
        monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp1-cp2"],
)
def test_sharded_template_checkpoint_conversion_pair_1d(setup_env):
    """serial→distributed conversion with 1D pair-slab placements (Shard(0), Shard(1))."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    # 4×4 weight: both dim-0 (dp=1, Shard(0)) and dim-1 (cp=2, Shard(1)) must be divisible.
    serial_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    serial_bias = torch.arange(4, dtype=torch.float32)
    placements = (Shard(0), Shard(1))
    shard_dim = 1  # cp shards on dim-1 for the pair row-slab layout
    fresh_env = dict(env_per_rank) if env_per_rank else {}
    fresh_env["MASTER_PORT"] = str(_find_free_port())
    payload = (grid_group_sizes, device_type, backend, fresh_env, serial_weight, serial_bias, placements, shard_dim)
    spawn_multiprocessing(_parallel_assert_sharded_conversion_1d, world_size, payload)


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp1-cp2"],
)
def test_sharded_template_checkpoint_conversion_msa_1d(setup_env):
    """serial→distributed conversion with 1D MSA inner-axis placements (Shard(0), Shard(2))."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    # Shape (B=2, S=4, N=cp*4=8, C=8): Shard(0) on dp=1 (trivial), Shard(2) on cp=2.
    cp_size = world_size  # dp=1, so world_size == cp_size
    serial_weight = torch.arange(2 * 4 * cp_size * 4 * 8, dtype=torch.float32).reshape(2, 4, cp_size * 4, 8)
    serial_bias = torch.arange(2, dtype=torch.float32)
    placements = (Shard(0), Shard(2))
    shard_dim = 2  # cp shards on dim-2 (inner N axis) for MSA
    fresh_env = dict(env_per_rank) if env_per_rank else {}
    fresh_env["MASTER_PORT"] = str(_find_free_port())
    payload = (grid_group_sizes, device_type, backend, fresh_env, serial_weight, serial_bias, placements, shard_dim)
    spawn_multiprocessing(_parallel_assert_sharded_conversion_1d, world_size, payload)


# ---------------------------------------------------------------------------
# Test C — SetTriAttnBackend on 1D model (PairformerLayer1D)
# ---------------------------------------------------------------------------


def _parallel_assert_set_triattn_backend_1d(rank: int, env_per_rank, triattn_backend, boltz2_params):
    """Worker: verify SetTriAttnBackend targets PairformerLayer1D instances in Boltz2_1D."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    from boltz.distributed.model.layers.pairformer_1d import PairformerLayer1D
    from boltz.distributed.model.models.boltz2_1d import Boltz2_1D
    from boltz.model.models.boltz2 import Boltz2 as SerialBoltz2

    grid_group_sizes = {"dp": 1, "cp": 1}
    DistributedManager.initialize(grid_group_sizes, device_type="cuda", backend="nccl")
    manager = DistributedManager()

    serial_model = SerialBoltz2(**boltz2_params).to(device=manager.device).eval()
    dist_model = Boltz2_1D(serial_model, manager).eval()

    pairformer_layers_before = []
    for name, submodule in dist_model.named_modules():
        if isinstance(submodule, PairformerLayer1D):
            pairformer_layers_before.append(name)
            assert (
                submodule.triattn_backend == TriAttnBackend.REFERENCE
            ), f"{name}: expected REFERENCE before setter, got {submodule.triattn_backend}"

    # Vacuous-pass guard: model must contain at least one PairformerLayer1D.
    assert len(pairformer_layers_before) > 0, "Model must contain at least one PairformerLayer1D"

    dist_model.apply(SetTriAttnBackend(triattn_backend))

    for name, submodule in dist_model.named_modules():
        if isinstance(submodule, PairformerLayer1D):
            assert (
                submodule.triattn_backend == triattn_backend
            ), f"{name}: expected {triattn_backend} after setter, got {submodule.triattn_backend}"
        else:
            assert not hasattr(submodule, "triattn_backend"), (
                f"Non-PairformerLayer1D module {name} ({type(submodule).__name__}) "
                f"unexpectedly has triattn_backend attribute"
            )

    DistributedManager.cleanup()


@pytest.mark.parametrize(
    "setup_env",
    [((1, 1), False, "cuda", "ENV")],
    indirect=True,
    ids=["cuda-dp1-cp1"],
)
@pytest.mark.parametrize(
    "triattn_backend",
    [TriAttnBackend.CUEQ, TriAttnBackend.TRIFAST],
    ids=lambda b: b.value,
)
def test_set_triattn_backend_1d(setup_env, triattn_backend):
    """SetTriAttnBackend sets triattn_backend only on PairformerLayer1D instances in Boltz2_1D."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    # Use a fresh port to avoid TIME_WAIT collisions between parametrized tests.
    fresh_env = dict(env_per_rank) if env_per_rank else {}
    fresh_env["MASTER_PORT"] = str(_find_free_port())

    spawn_multiprocessing(
        _parallel_assert_set_triattn_backend_1d,
        world_size,
        fresh_env,
        triattn_backend,
        create_boltz2_model_init_params(use_large_model=False),
    )


# ---------------------------------------------------------------------------
# Test D — SetAttnPairBiasBackend on 1D model (AttentionPairBias1D)
# ---------------------------------------------------------------------------


def _parallel_assert_set_attn_pair_bias_backend_1d(rank: int, env_per_rank, sdpa_backend, boltz2_params):
    """Worker: verify SetAttnPairBiasBackend targets AttentionPairBias1D in Boltz2_1D."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            monkeypatch.setenv(var_name, f"{rank}" if value == "<INPUT_RANK>" else value)

    from boltz.distributed.model.layers.attention import AttentionPairBiasShardwise
    from boltz.distributed.model.layers.attention_1d import AttentionPairBias1D
    from boltz.distributed.model.models.boltz2_1d import Boltz2_1D
    from boltz.model.models.boltz2 import Boltz2 as SerialBoltz2

    grid_group_sizes = {"dp": 1, "cp": 1}
    DistributedManager.initialize(grid_group_sizes, device_type="cuda", backend="nccl")
    manager = DistributedManager()

    serial_model = SerialBoltz2(**boltz2_params).to(device=manager.device).eval()
    dist_model = Boltz2_1D(serial_model, manager).eval()

    attn_modules_before = []
    for name, submodule in dist_model.named_modules():
        if isinstance(submodule, AttentionPairBias1D):
            attn_modules_before.append(name)
            assert submodule.sdpa_with_bias_backend != sdpa_backend, (
                f"{name}: default backend already matches {sdpa_backend}, "
                "test cannot verify the setter changes anything"
            )

    # Vacuous-pass guard: model must contain at least one AttentionPairBias1D.
    assert len(attn_modules_before) > 0, "Model must contain at least one AttentionPairBias1D"

    # Record backends of AttentionPairBiasShardwise modules (diffusion transformer)
    # before the setter so we can assert they are NOT changed by SetAttnPairBiasBackend.
    shardwise_backends_before = {
        name: submodule.sdpa_with_bias_backend
        for name, submodule in dist_model.named_modules()
        if isinstance(submodule, AttentionPairBiasShardwise)
    }

    dist_model.apply(SetAttnPairBiasBackend(sdpa_backend))

    for name, submodule in dist_model.named_modules():
        if isinstance(submodule, AttentionPairBias1D):
            assert (
                submodule.sdpa_with_bias_backend == sdpa_backend
            ), f"{name}: expected {sdpa_backend} after setter, got {submodule.sdpa_with_bias_backend}"
        # Positive invariant: SetAttnPairBiasBackend must NOT modify AttentionPairBiasShardwise.
        if isinstance(submodule, AttentionPairBiasShardwise):
            assert (
                submodule.sdpa_with_bias_backend == shardwise_backends_before[name]
            ), f"AttentionPairBiasShardwise {name} was unexpectedly changed by SetAttnPairBiasBackend"

    DistributedManager.cleanup()


@pytest.mark.parametrize(
    "setup_env",
    [((1, 1), False, "cuda", "ENV")],
    indirect=True,
    ids=["cuda-dp1-cp1"],
)
@pytest.mark.parametrize(
    "sdpa_backend",
    [SDPAWithBiasBackend.TORCH_FLEX_ATTN],
    ids=lambda b: b.value,
)
def test_set_attn_pair_bias_backend_1d(setup_env, sdpa_backend):
    """SetAttnPairBiasBackend sets sdpa_with_bias_backend only on AttentionPairBias1D in Boltz2_1D."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    # Use a fresh port to avoid TIME_WAIT collisions between parametrized tests.
    fresh_env = dict(env_per_rank) if env_per_rank else {}
    fresh_env["MASTER_PORT"] = str(_find_free_port())

    spawn_multiprocessing(
        _parallel_assert_set_attn_pair_bias_backend_1d,
        world_size,
        fresh_env,
        sdpa_backend,
        create_boltz2_model_init_params(use_large_model=False),
    )
