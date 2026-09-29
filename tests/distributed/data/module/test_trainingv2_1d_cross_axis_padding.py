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

"""Registry-driven cross-axis padding inside ``trainingv2_1d`` prep.

Regression for MR !459 P2: the cross-axis pad dispatch inside
:meth:`boltz.distributed.data.module.trainingv2_1d._BaseDatasetCP1D._prepare_features_for_distribution`
must drive axis-label decisions from
:data:`boltz.distributed.data.module.placements_1d.FEATURE_AXIS_SEMANTICS_1D`,
not a runtime size-match heuristic.  The previous heuristic compared
each non-sharded axis size against ``original_N_tokens`` /
``original_N_atoms`` and picked the first matching branch — when
``N_tokens == N_atoms`` (rare, very small atom-resolution tests) the
``tokens``-first ``if`` branch always wins regardless of the feature's
actual axis semantics.

Two failure modes are guarded:

1. **Adversarial registry-driven dispatch** — monkey-patch the
   axis-semantics map to relabel a feature's non-sharded axis, then
   set ``target_n_tokens != target_n_atoms`` and assert the output
   axis is padded to the registry-declared target.  This proves the
   implementation reads from the registry, not from runtime size
   matching.

2. **``N_tokens == N_atoms`` happy path** — confirm that the degenerate
   case still produces the model-forward-compatible shape for the two
   classic cross-axis features (``atom_to_token`` and
   ``token_to_rep_atom``).  The two axes happen to align numerically,
   so this is a shape-correctness backstop, not a discrimination test.
"""

from __future__ import annotations

import importlib
import os
import socket
from typing import Any

import pytest
import torch
import torch.distributed
import torch.multiprocessing

from boltz.testing.utils import spawn_multiprocessing


def _setup_gloo_group(rank: int, world_size: int, port: int) -> torch.distributed.ProcessGroup:
    """Initialize a gloo process group used as the synthetic CP group."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    torch.distributed.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    return torch.distributed.group.WORLD


def _find_free_port() -> int:
    """Return a free TCP port. Avoids cross-test port collisions."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _FakeSerialDataset:
    """Returns a fixed feature dict; never indexed beyond zero in these tests."""

    def __init__(self, features: dict[str, Any]) -> None:
        self._features = features

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:  # noqa: ARG002
        # Return a fresh dict each call to avoid mutation leakage between
        # the dataset and the prep path (prep modifies in place).
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self._features.items()}


def _build_dataset_for_prep(world_size: int, features: dict[str, Any]) -> Any:
    """Construct a real ``TrainingDatasetCP1D`` over a 1D CP submesh."""
    from torch.distributed.device_mesh import init_device_mesh

    from boltz.distributed.data.module.trainingv2_1d import TrainingDatasetCP1D

    # (dp, cp) = (1, world_size). CPU meshes throughout — the prep path
    # only reads ``cp_submesh.shape[0]`` and ``device_mesh.get_coordinate``.
    cpu_mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("dp_cpu", "cp_cpu"))
    device_mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("dp", "cp"))
    return TrainingDatasetCP1D(
        serial_dataset=_FakeSerialDataset(features),
        device_mesh=device_mesh,
        device_mesh_cpu=cpu_mesh,
    )


def _worker_n_tokens_eq_n_atoms_happy_shape(
    rank: int,  # noqa: ARG001 — required by torch.multiprocessing.spawn API
    world_size: int,
    port: int,
) -> None:
    """N_tokens==N_atoms==3, cp_size=2: cross-axis features end up shape ``[4, 4]``.

    Shape backstop for the degenerate symmetric case.  In this case the
    old heuristic happens to produce the right shape (target_n_tokens
    == target_n_atoms == 4) so this test alone cannot discriminate the
    two implementations — it is paired with the adversarial dispatch
    test below.
    """
    try:
        _setup_gloo_group(rank, world_size, port)

        n_tokens = 3
        n_atoms = 3
        features = {
            "token_pad_mask": torch.ones(n_tokens, dtype=torch.bool),
            "atom_pad_mask": torch.ones(n_atoms, dtype=torch.bool),
            "atom_to_token": torch.arange(n_atoms * n_tokens, dtype=torch.float32).reshape(n_atoms, n_tokens),
            "token_to_rep_atom": torch.arange(n_tokens * n_atoms, dtype=torch.float32).reshape(n_tokens, n_atoms),
        }
        dataset = _build_dataset_for_prep(world_size, features)

        tensor_features_all, _ = dataset._prepare_features_for_distribution(dataset.serial_dataset[0])

        # cp_size=2 → both N axes get padded from 3 to 4.
        assert tensor_features_all["token_pad_mask"].shape == (4,)
        assert tensor_features_all["atom_pad_mask"].shape == (4,)
        assert tensor_features_all["atom_to_token"].shape == (
            4,
            4,
        ), f"atom_to_token expected [4,4], got {tuple(tensor_features_all['atom_to_token'].shape)}"
        assert tensor_features_all["token_to_rep_atom"].shape == (
            4,
            4,
        ), f"token_to_rep_atom expected [4,4], got {tuple(tensor_features_all['token_to_rep_atom'].shape)}"
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _worker_registry_drives_dispatch(
    rank: int,  # noqa: ARG001
    world_size: int,
    port: int,
) -> None:
    """Adversarial: a relabeled axis is padded to the registry-declared target.

    Picks ``token_to_rep_atom`` (registry: ``("tokens", "atoms")``) and
    monkey-patches its semantics to ``("tokens", "tokens")``.  With
    ``N_tokens > N_atoms`` (after CP-divisibility pad), dim 1 must be
    padded to the *token* target rather than the (smaller) atom target.

    Asserts the registry-declared label wins.  This isolates the
    dispatch mechanism from any incidental shape coincidence.

    Failure mode if the heuristic regresses: dim 1 size matches
    ``original_n_atoms`` (the truthful pre-pad value), the elif branch
    fires, and the axis is padded to ``target_n_atoms`` — the wrong
    target.  The shape mismatch is observable.
    """
    try:
        _setup_gloo_group(rank, world_size, port)

        # Force divergent targets: N_atoms divisible by cp_size, N_tokens not.
        # cp_size=2, N_atoms=4 (no pad), N_tokens=3 (pad to 4) → still equal
        # post-pad. Use cp_size=2, N_atoms=2, N_tokens=3 (pad to 4) so
        # target_n_atoms=2 ≠ target_n_tokens=4. dim-1 size==2==N_atoms.
        n_tokens = 3
        n_atoms = 2
        features = {
            "token_pad_mask": torch.ones(n_tokens, dtype=torch.bool),
            "atom_pad_mask": torch.ones(n_atoms, dtype=torch.bool),
            "token_to_rep_atom": torch.arange(n_tokens * n_atoms, dtype=torch.float32).reshape(n_tokens, n_atoms),
        }

        # Monkey-patch the axis-semantics registry to relabel
        # ``token_to_rep_atom``'s dim 1 from "atoms" to "tokens".
        placements_1d = importlib.import_module("boltz.distributed.data.module.placements_1d")
        saved_label = placements_1d.FEATURE_AXIS_SEMANTICS_1D["token_to_rep_atom"]
        placements_1d.FEATURE_AXIS_SEMANTICS_1D["token_to_rep_atom"] = ("tokens", "tokens")
        try:
            dataset = _build_dataset_for_prep(world_size, features)
            tensor_features_all, _ = dataset._prepare_features_for_distribution(dataset.serial_dataset[0])

            # cp_size=2 → token_pad_mask 3→4, atom_pad_mask 2 (no pad).
            assert tensor_features_all["token_pad_mask"].shape == (4,)
            assert tensor_features_all["atom_pad_mask"].shape == (2,)

            # Sharded dim 0 (tokens) padded from 3→4 by the Shard loop.
            # Cross-axis dim 1: relabeled "tokens" → padded to N_tokens=4.
            # Old heuristic (size-match): size==2==original_n_atoms → padded to 2
            # (i.e. no pad). New registry-driven dispatch: padded to 4.
            actual = tuple(tensor_features_all["token_to_rep_atom"].shape)
            assert actual == (4, 4), f"expected registry-driven [4,4], got {actual} (heuristic would give [4,2])"
        finally:
            placements_1d.FEATURE_AXIS_SEMANTICS_1D["token_to_rep_atom"] = saved_label
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _worker_registry_coverage_invariant(
    rank: int,  # noqa: ARG001
    world_size: int,  # noqa: ARG001
    port: int,  # noqa: ARG001
) -> None:
    """The import-time coverage check still binds training + inference placements.

    Adversarial guard: if a future contributor adds a placement entry
    but forgets the axis-semantics annotation, the cross-axis padding
    path silently no-ops on that feature.  The registry-coverage check
    inside ``placements_1d`` fails fast at import.

    This runs in a subprocess to isolate the monkey-patched state from
    the parent pytest process.
    """
    placements_1d = importlib.import_module("boltz.distributed.data.module.placements_1d")

    # Inject a placement-only entry on a fresh copy; the coverage check
    # should reject it.
    training_keys = set(placements_1d.TRAINING_FEATURE_PLACEMENTS_1D.keys())
    inference_keys = set(placements_1d.INFERENCE_FEATURE_PLACEMENTS_1D.keys())
    semantic_keys = set(placements_1d.FEATURE_AXIS_SEMANTICS_1D.keys())
    placement_keys = training_keys | inference_keys

    # Sanity: at baseline they agree.
    assert (
        placement_keys == semantic_keys
    ), f"baseline registry drift: placements\\semantics={placement_keys - semantic_keys}, semantics\\placements={semantic_keys - placement_keys}"

    # Drift simulation: temporarily add a fake placement and re-run the check.
    placements_1d.TRAINING_FEATURE_PLACEMENTS_1D["__drift_test_feature__"] = placements_1d.PLACEMENT_1D_SINGLE
    try:
        with pytest.raises(RuntimeError, match="FEATURE_AXIS_SEMANTICS_1D and placement registries disagree"):
            placements_1d._check_registry_coverage()
    finally:
        del placements_1d.TRAINING_FEATURE_PLACEMENTS_1D["__drift_test_feature__"]


@pytest.mark.parametrize("world_size", [2])
def test_n_tokens_eq_n_atoms_cross_axis_shapes(world_size: int) -> None:
    """``N_tokens == N_atoms`` happy-path: cross-axis features padded to common dim.

    Shape backstop only — the degenerate symmetric case produces the
    same numerical shape under both the old heuristic and the new
    registry dispatch.  The adversarial discrimination is provided by
    :func:`test_registry_drives_cross_axis_dispatch`.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_n_tokens_eq_n_atoms_happy_shape, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [2])
def test_registry_drives_cross_axis_dispatch(world_size: int) -> None:
    """Cross-axis axis label is read from ``FEATURE_AXIS_SEMANTICS_1D``, not size.

    Adversarial guard: monkey-patches one feature's axis label so the
    registry-declared target diverges from the size-match heuristic's
    pick.  Failure mode if the heuristic regresses: output shape
    matches the heuristic's wrong pick instead of the registry's
    target.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_registry_drives_dispatch, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [1])
def test_registry_coverage_invariant(world_size: int) -> None:
    """Adding a placement without an axis-semantics entry fails import-style check.

    Guards against silent drift: any future placement-only addition
    must come with a matching axis-semantics annotation, or the
    cross-axis pad path on that feature silently no-ops.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_registry_coverage_invariant, world_size, world_size, port)


def _worker_cyclic_period_padded_to_cp_multiple(
    rank: int,  # noqa: ARG001
    world_size: int,
    port: int,
) -> None:
    """Every ``Shard(0)`` token feature — ``cyclic_period`` included — is padded
    to a multiple of ``cp_size`` by the prep path, with pad rows zero-filled.

    Regression for MR !481: even token sharding is a *data-pipeline*
    invariant, which is what lets ``RelativePositionEncoder1D`` assume uniform
    shards (and assert ``N % cp == 0``) instead of repairing heterogeneous
    ``cyclic_period`` shards inside the model forward.  With ``cp_size=2`` and
    ``N_tokens=3`` every relative-position token feature must come out padded to
    4; ``cyclic_period``'s pad rows must be 0 (padding tokens are non-cyclic).
    """
    try:
        _setup_gloo_group(rank, world_size, port)

        cp_size = world_size
        n_tokens = 3  # deliberately NOT divisible by cp_size=2
        n_atoms = 2
        rel_pos_keys = ["asym_id", "entity_id", "residue_index", "token_index", "sym_id", "cyclic_period"]

        features: dict[str, Any] = {
            "token_pad_mask": torch.ones(n_tokens, dtype=torch.bool),
            "atom_pad_mask": torch.ones(n_atoms, dtype=torch.bool),
        }
        for key in rel_pos_keys:
            # Non-zero values so zero-filled pad rows are distinguishable.
            features[key] = torch.arange(1, n_tokens + 1, dtype=torch.long)
        features["cyclic_period"] = torch.full((n_tokens,), 5, dtype=torch.long)

        dataset = _build_dataset_for_prep(world_size, features)
        tensor_features_all, _ = dataset._prepare_features_for_distribution(dataset.serial_dataset[0])

        expected_n = ((n_tokens + cp_size - 1) // cp_size) * cp_size  # 3 → 4
        for key in rel_pos_keys:
            shape0 = tensor_features_all[key].shape[0]
            assert shape0 == expected_n, f"{key} token dim {shape0} != expected {expected_n}"
            assert shape0 % cp_size == 0, f"{key} token dim {shape0} not divisible by cp_size {cp_size}"

        # cyclic_period must share the SAME padded token count as asym_id — the
        # encoder all-gathers all six rel-pos feats over the same token axis.
        assert (
            tensor_features_all["cyclic_period"].shape[0] == tensor_features_all["asym_id"].shape[0]
        ), "cyclic_period token dim must equal asym_id token dim after data-path padding"

        cyc = tensor_features_all["cyclic_period"]
        assert (cyc[:n_tokens] == 5).all(), "real cyclic_period values were corrupted by padding"
        assert (cyc[n_tokens:] == 0).all(), "cyclic_period pad rows must be 0 (padding tokens are non-cyclic)"
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
def test_cyclic_period_padded_to_cp_multiple(world_size: int) -> None:
    """Data path pads ``cyclic_period`` (and rel-pos token feats) to a cp multiple.

    Pins the even-sharding invariant that ``RelativePositionEncoder1D`` relies
    on (MR !481): the deadlock-prone heterogeneous-shard repair was removed
    from the model forward because the prep path guarantees uniform shards.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_cyclic_period_padded_to_cp_multiple, world_size, world_size, port)
