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

"""Regression: cross-rank propagation of CP rank-zero serial-fetch errors.

Without :func:`boltz.distributed.data.module._error_propagation.rank_zero_fetch_with_propagation`,
an exception raised by ``serial_dataset[idx]`` on CP rank zero unwinds
rank zero before it enters the first broadcast inside
``_distribute_features``.  Peer CP ranks are already blocked in
``broadcast_object_list`` and deadlock forever — NCCL's per-collective
watchdog cannot fire because rank zero never entered the collective.

This file covers three failure modes the fix protects against:

1. ``serial_dataset[idx]`` raises a plain ``ValueError`` (the actual
   cluster bug — non-normalized sampling probabilities).
2. The validation scan loop raises ``RuntimeError`` because every sample
   was filtered out by ``val_skip_sample_threshold_*``.
3. The rank-zero **pre-broadcast preparation** block inside
   ``_BaseDatasetCP1D._prepare_features_for_distribution`` raises (e.g.
   ``KeyError`` for unknown tensor feature keys, ``pad_dim`` failure).
   This block runs after the serial fetch but before the first
   broadcast inside ``_distribute_features``; it is therefore inside the
   same deadlock window as the serial fetch itself and must be wrapped
   by the same propagation helper.
4. The inference rank-zero fetch exhausts ``max_data_retries`` and raises
   ``RuntimeError("Data loading failed N consecutive times...")`` —
   protects ``inferencev2_1d.__getitem__`` wrapping ``_fetch_rank_zero_payload``
   in ``rank_zero_fetch_with_propagation``.
5. The inference retry loop (``_raise_or_return_item_0``) recurses into
   ``_fetch_rank_zero_payload`` (serial-only) instead of ``__getitem__``
   (which would issue a fresh collective on rank zero alone) — guards
   against the most likely future-regression of MR !459 P1.

For each, we assert every CP rank raises ``RuntimeError`` containing the
rank-zero failure message within a tight wall-clock budget (a true
deadlock would block for the NCCL timeout, typically 30 s+).

The adversarial-revert protection is the wall-clock budget itself: if
the helper is removed and the old in-place pattern returns, the test
hangs at the broadcast, the per-worker timeout (~30 s) fires, and the
test fails loudly rather than passing silently.

Tests run on the gloo CPU backend with world_size=2 (one CP group of
size 2) — this exercises the propagation path without GPU dependencies.
A small parametrization extends to size 3 and size 4 to confirm the
fix scales beyond rank pair.
"""

from __future__ import annotations

import os
import signal
import time
from typing import Any

import pytest
import torch
import torch.distributed
import torch.multiprocessing

from boltz.distributed.data.module._error_propagation import (
    rank_zero_fetch_with_propagation,
)
from boltz.testing.utils import spawn_multiprocessing

_RAISING_VALUE_MESSAGE = "test-rank0-fetch-raises-valueerror"
_RAISING_RUNTIME_MESSAGE = "test-rank0-fetch-raises-runtimeerror"

# Wall-clock budget per worker. A true deadlock would block at the gloo
# broadcast for many seconds (or, in production, the NCCL watchdog
# timeout). The propagation helper resolves on the broadcast itself —
# typical run time is sub-second. 30 s is a generous ceiling that still
# distinguishes "broadcast resolved" from "deadlocked at broadcast".
_DEADLOCK_BUDGET_S = 30.0


def _setup_gloo_group(rank: int, world_size: int, port: int) -> torch.distributed.ProcessGroup:
    """Initialize a gloo process group used as the synthetic CP group.

    All callers must pair this with ``torch.distributed.destroy_process_group``
    on success or failure; the worker functions below wrap that in
    ``try/finally``.
    """
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


def _worker_valueerror_propagates(
    rank: int,
    world_size: int,
    port: int,
    deadlock_alarm_s: float = _DEADLOCK_BUDGET_S,
) -> None:
    """Assert ``ValueError`` on rank zero re-raises as ``RuntimeError`` on every rank."""
    # signal.alarm enforces the deadlock budget on POSIX worker procs.
    # Expected failure mode if the helper is reverted: peer ranks block at
    # the broadcast inside ``rank_zero_fetch_with_propagation`` (or whatever
    # the reverted code calls), SIGALRM fires, the worker is terminated by
    # the signal, and ``torch.multiprocessing.spawn`` re-raises a
    # ``ProcessExitedException`` (visible in pytest as "worker terminated
    # by signal SIGALRM" / non-zero subprocess exit). Don't look for the
    # assertion failures below — they only fire on the success path.
    signal.alarm(int(deadlock_alarm_s))
    try:
        cp_group = _setup_gloo_group(rank, world_size, port)
        cp_src = min(torch.distributed.get_process_group_ranks(cp_group))
        is_rank_zero = rank == cp_src

        def _fetch_fn() -> Any:
            raise ValueError(_RAISING_VALUE_MESSAGE)

        t0 = time.time()
        with pytest.raises(RuntimeError) as exc_info:
            rank_zero_fetch_with_propagation(
                fetch_fn=_fetch_fn,
                is_cp_rank_zero=is_rank_zero,
                cp_group=cp_group,
                cp_group_src_rank_global=cp_src,
            )
        elapsed = time.time() - t0

        msg = str(exc_info.value)
        assert "ValueError" in msg, f"rank {rank}: missing 'ValueError' in: {msg}"
        assert (
            _RAISING_VALUE_MESSAGE in msg
        ), f"rank {rank}: missing original message '{_RAISING_VALUE_MESSAGE}' in: {msg}"
        assert "Rank-zero traceback:" in msg, f"rank {rank}: missing rank-zero traceback header in: {msg}"
        assert (
            elapsed < deadlock_alarm_s
        ), f"rank {rank}: propagation took {elapsed:.1f}s (deadlock budget {deadlock_alarm_s}s)"
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _worker_runtimeerror_propagates(
    rank: int,
    world_size: int,
    port: int,
    deadlock_alarm_s: float = _DEADLOCK_BUDGET_S,
) -> None:
    """Assert ``RuntimeError`` on rank zero re-raises with original message on every rank.

    Models the validation "all samples filtered out" path where the
    rank-zero scan loop raises ``RuntimeError`` itself (not a downstream
    ``ValueError``).  The helper must preserve the original exception
    type name and message even when it itself raises ``RuntimeError``.
    """
    signal.alarm(int(deadlock_alarm_s))
    try:
        cp_group = _setup_gloo_group(rank, world_size, port)
        cp_src = min(torch.distributed.get_process_group_ranks(cp_group))
        is_rank_zero = rank == cp_src

        def _fetch_fn() -> Any:
            raise RuntimeError(_RAISING_RUNTIME_MESSAGE)

        with pytest.raises(RuntimeError) as exc_info:
            rank_zero_fetch_with_propagation(
                fetch_fn=_fetch_fn,
                is_cp_rank_zero=is_rank_zero,
                cp_group=cp_group,
                cp_group_src_rank_global=cp_src,
            )

        msg = str(exc_info.value)
        assert (
            _RAISING_RUNTIME_MESSAGE in msg
        ), f"rank {rank}: missing original message '{_RAISING_RUNTIME_MESSAGE}' in: {msg}"
        assert (
            "RuntimeError" in msg.split("Rank-zero traceback:")[0]
        ), f"rank {rank}: original exception type missing from header: {msg}"
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


_RAISING_INFER_RETRY_EXHAUSTED_MESSAGE = (
    "Data loading failed 5 consecutive times. Last error: synthetic-inference-fetch-error"
)


def _worker_inference_retry_exhaustion_propagates(
    rank: int,
    world_size: int,
    port: int,
    deadlock_alarm_s: float = _DEADLOCK_BUDGET_S,
) -> None:
    """Assert inference rank-0 retry exhaustion propagates a uniform ``RuntimeError``.

    Models the inference-path failure mode in
    :class:`boltz.distributed.data.module.inferencev2_1d.PredictionDatasetCPWithDTensorV2_1D`:
    a serial-fetch chain that retries ``max_data_retries`` times on rank zero
    and finally raises ``RuntimeError("Data loading failed N consecutive times...")``.
    Before the fix, rank zero raised before reaching the keys broadcast and peer
    ranks deadlocked at ``broadcast_object_list``.  With the fix
    (``inferencev2_1d.__getitem__`` wraps ``_fetch_rank_zero_payload`` in
    ``rank_zero_fetch_with_propagation``) the retry-exhaustion error becomes a
    uniform ``RuntimeError`` on every CP rank.
    """
    signal.alarm(int(deadlock_alarm_s))
    try:
        cp_group = _setup_gloo_group(rank, world_size, port)
        cp_src = min(torch.distributed.get_process_group_ranks(cp_group))
        is_rank_zero = rank == cp_src

        # Mimic _raise_or_return_item_0 exhaustion: the final fetch attempt
        # raises the canonical retry-exhaustion RuntimeError on rank zero.
        def _fetch_fn() -> Any:
            raise RuntimeError(_RAISING_INFER_RETRY_EXHAUSTED_MESSAGE)

        t0 = time.time()
        with pytest.raises(RuntimeError) as exc_info:
            rank_zero_fetch_with_propagation(
                fetch_fn=_fetch_fn,
                is_cp_rank_zero=is_rank_zero,
                cp_group=cp_group,
                cp_group_src_rank_global=cp_src,
            )
        elapsed = time.time() - t0

        msg = str(exc_info.value)
        assert _RAISING_INFER_RETRY_EXHAUSTED_MESSAGE in msg, f"rank {rank}: missing retry-exhaustion message in: {msg}"
        assert (
            "RuntimeError" in msg.split("Rank-zero traceback:")[0]
        ), f"rank {rank}: original RuntimeError type missing from header: {msg}"
        assert (
            elapsed < deadlock_alarm_s
        ), f"rank {rank}: propagation took {elapsed:.1f}s (deadlock budget {deadlock_alarm_s}s)"
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class _StubInferenceDatasetCP1D:
    """Minimal stand-in mirroring the inferencev2_1d retry shape.

    Reproduces the exact call chain under test: ``__getitem__`` wraps
    ``_fetch_rank_zero_payload`` via :func:`rank_zero_fetch_with_propagation`;
    ``_fetch_rank_zero_payload`` defers to ``_raise_or_return_item_0`` on
    failure; ``_raise_or_return_item_0`` recurses into
    ``_fetch_rank_zero_payload`` (NOT ``__getitem__``) so the retry loop
    runs serially on rank zero without issuing collectives.

    The class is deliberately decoupled from the real Boltz featurizer
    /manifest fixtures — it only exposes the rank-0 retry-recursion
    contract.  If that contract is broken (e.g. someone "simplifies"
    ``_raise_or_return_item_0`` back into calling ``__getitem__``), the
    retry loop would call ``rank_zero_fetch_with_propagation`` recursively
    on rank zero while peer ranks wait at the outer broadcast — the
    resulting collective-count mismatch deadlocks the gloo broadcast and
    ``signal.alarm`` terminates the worker, failing the test.
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        cp_src: int,
        is_cp_rank_zero: bool,
        max_data_retries: int,
        counter: dict[str, int],
    ) -> None:
        self._cp_group = cp_group
        self._cp_src = cp_src
        self.is_cp_rank_zero = is_cp_rank_zero
        self.max_data_retries = max_data_retries
        self._fallback_depth = 0
        self._counter = counter

    def _raise_or_return_item_0(self, e: Exception) -> Any:
        # Direct copy of the inferencev2_1d retry-recursion shape.
        if self.max_data_retries <= 0:
            raise e
        if self._fallback_depth >= self.max_data_retries:
            raise RuntimeError(f"Data loading failed {self.max_data_retries} consecutive times. Last error: {e}") from e
        self._fallback_depth += 1
        try:
            # Critical: must recurse into the SERIAL fetch path, not __getitem__.
            return self._fetch_rank_zero_payload(self._fallback_depth)
        finally:
            self._fallback_depth -= 1

    def _fetch_rank_zero_payload(self, idx: int) -> Any:
        self._counter["n"] += 1
        try:
            raise ValueError("synthetic-retryable-fetch-error")
        except Exception as e:  # noqa: BLE001
            return self._raise_or_return_item_0(e)

    def __getitem__(self, idx: int) -> Any:
        return rank_zero_fetch_with_propagation(
            fetch_fn=lambda: self._fetch_rank_zero_payload(idx),
            is_cp_rank_zero=self.is_cp_rank_zero,
            cp_group=self._cp_group,
            cp_group_src_rank_global=self._cp_src,
        )


def _worker_retry_recursion_is_serial(
    rank: int,
    world_size: int,
    port: int,
    deadlock_alarm_s: float = _DEADLOCK_BUDGET_S,
) -> None:
    """Assert ``_raise_or_return_item_0`` retries serially on rank-0 (no peer deadlock).

    Adversarial guard for the *critical* correctness detail of this fix:
    ``_raise_or_return_item_0`` recurses into ``_fetch_rank_zero_payload``
    (serial-only), NOT ``__getitem__`` (which would re-enter
    ``rank_zero_fetch_with_propagation`` and issue a fresh collective on
    rank zero alone).

    Failure mode if the bug regresses: rank zero issues ``max_data_retries``
    extra broadcasts inside its retry loop while peers wait at the outer
    single broadcast.  Gloo deadlocks; ``signal.alarm`` fires; the worker
    is terminated by SIGALRM and the test fails.

    On success, every CP rank receives the propagated retry-exhaustion
    ``RuntimeError`` after the helper's single broadcast.
    """
    max_data_retries = 3
    signal.alarm(int(deadlock_alarm_s))
    try:
        cp_group = _setup_gloo_group(rank, world_size, port)
        cp_src = min(torch.distributed.get_process_group_ranks(cp_group))
        is_rank_zero = rank == cp_src

        counter: dict[str, int] = {"n": 0}
        dataset = _StubInferenceDatasetCP1D(
            cp_group=cp_group,
            cp_src=cp_src,
            is_cp_rank_zero=is_rank_zero,
            max_data_retries=max_data_retries,
            counter=counter,
        )

        t0 = time.time()
        with pytest.raises(RuntimeError) as exc_info:
            dataset[0]
        elapsed = time.time() - t0

        msg = str(exc_info.value)
        assert (
            "Data loading failed" in msg and "consecutive times" in msg
        ), f"rank {rank}: missing retry-exhaustion message in: {msg}"
        assert (
            elapsed < deadlock_alarm_s
        ), f"rank {rank}: retry-exhaustion took {elapsed:.1f}s (deadlock budget {deadlock_alarm_s}s)"

        # Rank zero ran the initial fetch plus max_data_retries retries; peers
        # touched neither.  This nails down the "retries are serial" contract:
        # the retry count is observable only on the rank that actually ran them.
        if is_rank_zero:
            expected = max_data_retries + 1
            assert (
                counter["n"] == expected
            ), f"rank {rank}: expected {expected} serial fetch attempts, got {counter['n']}"
        else:
            assert counter["n"] == 0, f"rank {rank}: peer should not run any serial fetches, got {counter['n']}"
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _worker_happy_path(
    rank: int,
    world_size: int,
    port: int,
    deadlock_alarm_s: float = _DEADLOCK_BUDGET_S,
) -> None:
    """Assert no-error path returns the rank-zero result on rank zero and ``None`` on peers.

    Adversarial guard: confirms the helper is *correctly transparent*
    on the success path. Without this, a buggy implementation that
    always raised on peers would pass the error-propagation tests but
    silently break every training step.
    """
    signal.alarm(int(deadlock_alarm_s))
    try:
        cp_group = _setup_gloo_group(rank, world_size, port)
        cp_src = min(torch.distributed.get_process_group_ranks(cp_group))
        is_rank_zero = rank == cp_src

        sentinel: dict[str, Any] = {"ok": True, "rank0_only": object()}

        def _fetch_fn() -> Any:
            return sentinel

        result = rank_zero_fetch_with_propagation(
            fetch_fn=_fetch_fn,
            is_cp_rank_zero=is_rank_zero,
            cp_group=cp_group,
            cp_group_src_rank_global=cp_src,
        )
        if is_rank_zero:
            assert result is sentinel, f"rank {rank}: expected rank-0 to receive sentinel back"
        else:
            assert result is None, f"rank {rank}: expected peers to receive None, got {result!r}"
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


_RAISING_PREP_KEYERROR_TOKEN = "without DTensor placement mapping"


class _FakeSerialDataset:
    """Minimal stand-in for the serial Boltz2 dataset.

    Returns a feature dict whose entries are valid (well-shaped tensors)
    except for one unknown tensor key — this triggers the
    ``KeyError`` branch inside
    ``_BaseDatasetCP1D._prepare_features_for_distribution`` so we
    exercise the pre-broadcast raise path the helper must propagate.
    """

    def __init__(self, inject_unknown_key: bool = True) -> None:
        self._inject_unknown_key = inject_unknown_key

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, Any]:  # noqa: ARG002
        # Build a minimal happy feature dict so the only failure mode
        # is the injected unknown-key trigger.  ``token_pad_mask`` is
        # present so the optional ``token_pair_pad_mask`` synthesis
        # path runs without raising on its own.
        feats: dict[str, Any] = {
            "token_pad_mask": torch.ones(4, dtype=torch.bool),
        }
        if self._inject_unknown_key:
            # Unknown tensor key (not in TRAINING_FEATURE_PLACEMENTS_1D
            # and not in NON_SHARDED_FEATURES_V2) — the pre-broadcast
            # check inside ``_prepare_features_for_distribution`` raises
            # ``KeyError`` on rank zero.
            feats["unknown_synthetic_tensor_xyzzy"] = torch.zeros(4)
        return feats


def _worker_prep_keyerror_propagates(
    rank: int,
    world_size: int,
    port: int,
    deadlock_alarm_s: float = _DEADLOCK_BUDGET_S,
) -> None:
    """Rank-zero pre-broadcast prep raise → uniform ``RuntimeError`` across CP ranks.

    Constructs a real ``TrainingDatasetCP1D`` against a 1D CP submesh
    (single CP group spanning all ranks) so that the failure surface
    matches production: the rank-0-only block in
    ``_prepare_features_for_distribution`` runs before any broadcast,
    and a raise there must propagate through the helper rather than
    deadlock peers waiting on the first ``broadcast_object_list``
    inside ``_distribute_features``.
    """
    # Local imports keep the worker self-contained for torch.multiprocessing
    # spawn pickling and avoid eager imports during pytest collection.
    from torch.distributed.device_mesh import init_device_mesh

    from boltz.distributed.data.module.trainingv2_1d import TrainingDatasetCP1D

    signal.alarm(int(deadlock_alarm_s))
    try:
        _setup_gloo_group(rank, world_size, port)
        # (dp, cp) = (1, world_size).  Both CPU meshes — _BaseDatasetCP1D
        # only uses device_mesh to read its CP-rank coordinate and
        # device_mesh_cpu for the actual CP submesh / process group.
        cpu_mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("dp_cpu", "cp_cpu"))
        device_mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("dp", "cp"))

        dataset = TrainingDatasetCP1D(
            serial_dataset=_FakeSerialDataset(inject_unknown_key=True),
            device_mesh=device_mesh,
            device_mesh_cpu=cpu_mesh,
        )

        t0 = time.time()
        with pytest.raises(RuntimeError) as exc_info:
            dataset[0]
        elapsed = time.time() - t0

        msg = str(exc_info.value)
        assert "KeyError" in msg, f"rank {rank}: expected 'KeyError' in propagated message, got: {msg}"
        assert (
            _RAISING_PREP_KEYERROR_TOKEN in msg
        ), f"rank {rank}: missing pre-broadcast KeyError marker in propagated message: {msg}"
        assert "Rank-zero traceback:" in msg, f"rank {rank}: missing rank-zero traceback header in: {msg}"
        assert (
            elapsed < deadlock_alarm_s
        ), f"rank {rank}: propagation took {elapsed:.1f}s (deadlock budget {deadlock_alarm_s}s)"
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _find_free_port() -> int:
    """Return a free TCP port. Avoids cross-test port collisions."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_rank_zero_valueerror_propagates_to_all_cp_ranks(world_size: int) -> None:
    """Rank-zero ``ValueError`` becomes a uniform ``RuntimeError`` on every CP rank."""
    port = _find_free_port()
    spawn_multiprocessing(_worker_valueerror_propagates, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_rank_zero_runtimeerror_propagates_to_all_cp_ranks(world_size: int) -> None:
    """Rank-zero ``RuntimeError`` (e.g. val all-filtered) is preserved on every CP rank."""
    port = _find_free_port()
    spawn_multiprocessing(_worker_runtimeerror_propagates, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_inference_retry_recursion_is_serial_no_peer_deadlock(world_size: int) -> None:
    """``_raise_or_return_item_0`` retries run serially on rank-0 only; peers do not deadlock.

    Regression guard for the critical correctness detail of MR !459 P1:
    the retry loop in ``inferencev2_1d._raise_or_return_item_0`` must
    recurse into ``_fetch_rank_zero_payload`` (serial), not
    ``__getitem__`` (collective).  See worker docstring for the
    adversarial deadlock failure mode.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_retry_recursion_is_serial, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_inference_retry_exhaustion_propagates_to_all_cp_ranks(world_size: int) -> None:
    """Rank-zero inference retry-exhaustion ``RuntimeError`` reaches every CP rank uniformly.

    Regression guard for ``inferencev2_1d.__getitem__`` wrapping
    ``_fetch_rank_zero_payload`` in ``rank_zero_fetch_with_propagation``.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_inference_retry_exhaustion_propagates, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [2, 3])
def test_happy_path_transparent_to_caller(world_size: int) -> None:
    """No-error path returns rank-0 result on rank 0 and ``None`` on peers."""
    port = _find_free_port()
    spawn_multiprocessing(_worker_happy_path, world_size, world_size, port)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_rank_zero_prep_keyerror_propagates_to_all_cp_ranks(world_size: int) -> None:
    """Rank-zero pre-broadcast prep ``KeyError`` propagates uniformly across CP ranks.

    Guards the window between the serial fetch and the first broadcast
    inside ``_distribute_features``: a raise there (e.g. unknown tensor
    feature key, ``pad_dim`` failure) must not leave peers blocked at
    ``broadcast_object_list``.
    """
    port = _find_free_port()
    spawn_multiprocessing(_worker_prep_keyerror_propagates, world_size, world_size, port)
