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

"""Tests for 1D CP feature distribution and placement registry.

Placements are 1-element tuples on the CP sub-mesh ``(cp,)``, following the
same convention as the 2D CP registry where placements target the CP sub-mesh
only. The DP dimension is handled separately by ``CollateDTensor``.

Verifies correct placements, shapes, values, and round-trip parity via
``distribute_features`` on the CP sub-mesh.

NOTE: These tests exercise the ``distribute_features`` path (broadcast + distribute).
The full featurizer path (``_pack_and_pad_features``) has a square-mesh requirement
that will need a separate 1D-aware featurizer. That is tracked as a follow-up.
"""

from typing import Dict, Optional

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from boltz.distributed.data.module.placements_1d import (
    BASE_FEATURE_PLACEMENTS_1D,
    INFERENCE_FEATURE_PLACEMENTS_1D,
    PLACEMENT_1D_ATOM,
    PLACEMENT_1D_ENSEMBLE,
    PLACEMENT_1D_MSA,
    PLACEMENT_1D_PAIR,
    PLACEMENT_1D_REPLICATE,
    PLACEMENT_1D_SINGLE,
    TRAINING_FEATURE_PLACEMENTS_1D,
)
from boltz.distributed.data.utils import (
    broadcast_tensors,
    distribute_features,
)
from boltz.distributed.manager import DistributedManager
from boltz.testing.utils import spawn_multiprocessing


def _worker_placement_registry_coverage(
    rank: int,
    device_type: str,
    backend: str,
    grid_group_sizes: Dict[str, int],
    env_map: Optional[dict[str, str]] = None,
):
    """Verify placement registry constants and coverage.

    Checks:
    - All placement constants are 1-element tuples (for the CP sub-mesh)
    - Shard dims match feature semantics
    - BASE_FEATURE_PLACEMENTS_1D covers all expected features
    - TRAINING and INFERENCE extend BASE correctly
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_map is not None:
        for var_name, value in env_map.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)

    try:
        # All placement constants must be 1-element tuples (for CP sub-mesh)
        for name, placement in [
            ("SINGLE", PLACEMENT_1D_SINGLE),
            ("PAIR", PLACEMENT_1D_PAIR),
            ("MSA", PLACEMENT_1D_MSA),
            ("ATOM", PLACEMENT_1D_ATOM),
            ("REPLICATE", PLACEMENT_1D_REPLICATE),
            ("ENSEMBLE", PLACEMENT_1D_ENSEMBLE),
        ]:
            assert len(placement) == 1, f"PLACEMENT_1D_{name} must have 1 element (CP sub-mesh), got {len(placement)}"

        # SINGLE/PAIR/ATOM shard tensor dim 0 (the token/atom/row dim)
        assert isinstance(PLACEMENT_1D_SINGLE[0], Shard) and PLACEMENT_1D_SINGLE[0].dim == 0
        assert isinstance(PLACEMENT_1D_PAIR[0], Shard) and PLACEMENT_1D_PAIR[0].dim == 0
        assert isinstance(PLACEMENT_1D_ATOM[0], Shard) and PLACEMENT_1D_ATOM[0].dim == 0
        # MSA shards tensor dim 1 (N in [S, N, C])
        assert isinstance(PLACEMENT_1D_MSA[0], Shard) and PLACEMENT_1D_MSA[0].dim == 1
        # ENSEMBLE shards tensor dim 1 (N in [E, N, ...])
        assert isinstance(PLACEMENT_1D_ENSEMBLE[0], Shard) and PLACEMENT_1D_ENSEMBLE[0].dim == 1
        # REPLICATE is fully replicated on CP
        assert isinstance(PLACEMENT_1D_REPLICATE[0], Replicate)

        # Pair features (N x N) must use PAIR placement (row-slab: first N on cp)
        pair_feature_names = {
            "disto_target",
            "token_bonds",
            "type_bonds",
            "token_pair_pad_mask",
            "pair_mask",
            "contact_conditioning",
            "contact_threshold",
        }
        for name in pair_feature_names:
            assert name in BASE_FEATURE_PLACEMENTS_1D, f"Pair feature {name} missing from registry"
            assert (
                BASE_FEATURE_PLACEMENTS_1D[name] == PLACEMENT_1D_PAIR
            ), f"Feature {name} should use PLACEMENT_1D_PAIR, got {BASE_FEATURE_PLACEMENTS_1D[name]}"

        # MSA features must use MSA placement
        msa_feature_names = {"msa", "msa_paired", "deletion_value", "has_deletion", "msa_mask"}
        for name in msa_feature_names:
            assert name in BASE_FEATURE_PLACEMENTS_1D, f"MSA feature {name} missing from registry"
            assert (
                BASE_FEATURE_PLACEMENTS_1D[name] == PLACEMENT_1D_MSA
            ), f"Feature {name} should use PLACEMENT_1D_MSA, got {BASE_FEATURE_PLACEMENTS_1D[name]}"

        # method_feature is replicated on cp (scalar-like, no token/atom dimension)
        assert BASE_FEATURE_PLACEMENTS_1D["method_feature"] == PLACEMENT_1D_REPLICATE

        # TRAINING extends BASE with training-specific features
        assert "temp_feature" in TRAINING_FEATURE_PLACEMENTS_1D
        assert "ph_feature" in TRAINING_FEATURE_PLACEMENTS_1D
        for key in BASE_FEATURE_PLACEMENTS_1D:
            assert key in TRAINING_FEATURE_PLACEMENTS_1D, f"BASE key {key} missing from TRAINING"
            assert TRAINING_FEATURE_PLACEMENTS_1D[key] == BASE_FEATURE_PLACEMENTS_1D[key]

        # INFERENCE extends BASE with inference-specific features
        assert "affinity_token_mask" in INFERENCE_FEATURE_PLACEMENTS_1D
        for key in BASE_FEATURE_PLACEMENTS_1D:
            assert key in INFERENCE_FEATURE_PLACEMENTS_1D, f"BASE key {key} missing from INFERENCE"
            assert INFERENCE_FEATURE_PLACEMENTS_1D[key] == BASE_FEATURE_PLACEMENTS_1D[key]

    finally:
        DistributedManager.cleanup()
        monkeypatch.undo()


def _worker_distribute_and_roundtrip(
    rank: int,
    device_type: str,
    backend: str,
    grid_group_sizes: Dict[str, int],
    env_map: Optional[dict[str, str]] = None,
):
    """Test feature distribution with 1D CP placements on the CP sub-mesh.

    Creates synthetic unbatched features for each placement type, distributes
    them on the ``(cp,)`` sub-mesh using ``distribute_features``, and verifies:
    - Correct placements on the resulting DTensors
    - Local shard shape < global shape on sharded dims (sharding is active)
    - Round-trip: ``full_tensor()`` matches the original serial tensor
    - MSA S-dimension is replicated across CP ranks
    - Pair features are row-slab sharded (first N on cp, second N full)
    - Replicated features are identical across CP ranks
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_map is not None:
        for var_name, value in env_map.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    try:
        device_mesh = manager.device_mesh
        cp_size = device_mesh.shape[1]

        # Build the CP sub-mesh (1D mesh with just the cp dimension).
        # For a 1D submesh, get_group() returns the group directly —
        # no flattening needed (get_flattened_group would conflict on name).
        cp_submesh = device_mesh["cp"]
        cp_group = cp_submesh.get_group(0)
        src_rank = min(torch.distributed.get_process_group_ranks(cp_group))
        is_src = rank == src_rank

        rng = torch.Generator().manual_seed(42)
        # Unbatched feature sizes (no batch dim — matches the per-sample
        # distribution path used by the dataset __getitem__)
        N = 8 * cp_size  # token dim, must be divisible by cp
        S = 6  # MSA sequences, not sharded on cp
        E = 2  # ensembles
        N_atoms = 12 * cp_size  # atom dim, must be divisible by cp
        C = 5  # channel dim

        # Build synthetic unbatched features on src rank, None on others
        if is_src:
            features = {
                # Single repr: [N, C] or [N]
                "token_index": torch.randint(0, 100, (N,), generator=rng),
                "residue_index": torch.randint(0, 100, (N,), generator=rng),
                "token_pad_mask": torch.ones(N),
                # Pair repr: [N, N]
                "disto_target": torch.randn(N, N, generator=rng),
                "token_bonds": torch.randint(0, 2, (N, N), generator=rng).float(),
                # MSA: [S, N, C]
                "msa": torch.randint(0, 20, (S, N, C), generator=rng),
                "msa_mask": torch.ones(S, N),
                # Atom: [N_atoms, 3]
                "ref_pos": torch.randn(N_atoms, 3, generator=rng),
                "atom_pad_mask": torch.ones(N_atoms),
                # Ensemble: [E, N_atoms, 3] and [E, N, 3]
                "coords": torch.randn(E, N_atoms, 3, generator=rng),
                "disto_coords_ensemble": torch.randn(E, N, 3, generator=rng),
                # Replicate on CP: [C]
                "method_feature": torch.randn(C, generator=rng),
            }
        else:
            features = None

        # Subset of placements matching our synthetic features
        placement_subset = {
            "token_index": PLACEMENT_1D_SINGLE,
            "residue_index": PLACEMENT_1D_SINGLE,
            "token_pad_mask": PLACEMENT_1D_SINGLE,
            "disto_target": PLACEMENT_1D_PAIR,
            "token_bonds": PLACEMENT_1D_PAIR,
            "msa": PLACEMENT_1D_MSA,
            "msa_mask": PLACEMENT_1D_MSA,
            "ref_pos": PLACEMENT_1D_ATOM,
            "atom_pad_mask": PLACEMENT_1D_ATOM,
            "coords": PLACEMENT_1D_ENSEMBLE,
            "disto_coords_ensemble": PLACEMENT_1D_ENSEMBLE,
            "method_feature": PLACEMENT_1D_REPLICATE,
        }

        dtensors = distribute_features(
            features=features,
            placements=placement_subset,
            group=cp_group,
            src_rank_global=src_rank,
            device_mesh=cp_submesh,
        )

        # Broadcast original features to all ranks for round-trip comparison
        # using broadcast_tensors (handles metadata correctly, avoids pickle)
        if is_src:
            tensor_features = dict(features)
        else:
            tensor_features = None
        original_features = broadcast_tensors(
            tensor_features,
            group=cp_group,
            src_rank_global=src_rank,
            device=cp_submesh.device_type,
        )

        # === Verify placements ===
        for name, dt in dtensors.items():
            expected_placements = tuple(placement_subset[name])
            actual_placements = dt.placements
            assert (
                actual_placements == expected_placements
            ), f"Feature {name}: expected placements {expected_placements}, got {actual_placements}"

        # === Verify sharding is active (local shape < global on sharded dims) ===
        for name, dt in dtensors.items():
            for mesh_dim, placement in enumerate(dt.placements):
                if isinstance(placement, Shard):
                    mesh_size = cp_submesh.shape[mesh_dim]
                    if mesh_size > 1:
                        local_size = dt.to_local().shape[placement.dim]
                        global_size = dt.shape[placement.dim]
                        assert local_size < global_size, (
                            f"Feature {name}: sharding not active on mesh dim {mesh_dim}. "
                            f"local={local_size}, global={global_size}"
                        )

        # === Round-trip: full_tensor() matches original ===
        for name, dt in dtensors.items():
            full = dt.full_tensor()
            original = original_features[name].to(device=cp_submesh.device_type)
            torch.testing.assert_close(
                full,
                original,
                atol=0,
                rtol=0,
                msg=f"Round-trip mismatch for feature {name}",
            )

        # === MSA: S-dimension is replicated across CP ranks ===
        # For MSA features [S, N, C], the S dim should be identical across CP ranks
        msa_local = dtensors["msa"].to_local()
        # Local shard should have all S sequences but only N/cp tokens
        assert msa_local.shape[0] == S, f"MSA local S dim should be {S} (replicated), got {msa_local.shape[0]}"
        if cp_size > 1:
            assert (
                msa_local.shape[1] == N // cp_size
            ), f"MSA local N dim should be {N // cp_size}, got {msa_local.shape[1]}"

        # === Pair features: row-slab (first N on cp, second N full) ===
        pair_local = dtensors["disto_target"].to_local()
        if cp_size > 1:
            # First N (rows) sharded, second N (cols) full
            assert (
                pair_local.shape[0] == N // cp_size
            ), f"Pair local row dim should be {N // cp_size}, got {pair_local.shape[0]}"
            assert pair_local.shape[1] == N, f"Pair local col dim should be {N} (full), got {pair_local.shape[1]}"

        # === Replicated features: identical across CP ranks ===
        method_local = dtensors["method_feature"].to_local()
        cp_pg = cp_submesh.get_group(0)
        gathered = [torch.empty_like(method_local) for _ in range(cp_size)]
        torch.distributed.all_gather(gathered, method_local, group=cp_pg)
        for i in range(1, len(gathered)):
            torch.testing.assert_close(
                gathered[0],
                gathered[i],
                atol=0,
                rtol=0,
                msg=f"Replicated feature method_feature differs between CP rank 0 and {i}",
            )

        # === Ensemble features: E dim replicated, N dim sharded ===
        coords_local = dtensors["coords"].to_local()
        assert (
            coords_local.shape[0] == E
        ), f"Ensemble local E dim should be {E} (replicated), got {coords_local.shape[0]}"
        if cp_size > 1:
            assert (
                coords_local.shape[1] == N_atoms // cp_size
            ), f"Ensemble local N_atoms dim should be {N_atoms // cp_size}, got {coords_local.shape[1]}"

    finally:
        DistributedManager.cleanup()
        monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 4), True, "cuda", "ENV"),
        ((2, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
def test_1d_cp_placement_registry_coverage(
    setup_env: tuple[dict, int, str, str, str, dict[str, str]],
):
    """Verify 1D CP placement constants and feature registry coverage."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        _worker_placement_registry_coverage,
        world_size,
        device_type,
        backend,
        grid_group_sizes,
        env_per_rank,
    )


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 4), True, "cuda", "ENV"),
        ((2, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
def test_1d_cp_distribute_and_roundtrip(
    setup_env: tuple[dict, int, str, str, str, dict[str, str]],
):
    """Test 1D CP feature distribution, sharding, and round-trip correctness."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        _worker_distribute_and_roundtrip,
        world_size,
        device_type,
        backend,
        grid_group_sizes,
        env_per_rank,
    )
