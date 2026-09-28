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

"""DTensor parity tests for ``weighted_rigid_align_1d``.

Tests the 1D-CP variant at
``src/boltz/distributed/model/loss/diffusion_1d.py::weighted_rigid_align_1d``
against the serial Boltz-2 reference
``boltz.model.loss.diffusionv2.weighted_rigid_align``.

The 1D-CP variant operates on a 2-axis ``(dp, cp)`` device mesh with
placements ``(Shard(0), Shard(1))`` on all four inputs and reduces the
weighted centroids and covariance over the flat ``cp`` group. SVD is
computed redundantly on every rank since the all-reduced inputs are
bit-identical post-reduction. Closes the 1D-CP module-level gap left
open by ``tests/distributed/model/loss/test_dtensor_weighted_rigid_align.py``
which exercises only the 2D-CP variant (a 3-axis mesh).
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.loss.diffusion_1d import weighted_rigid_align_1d
from boltz.model.loss.diffusionv2 import weighted_rigid_align as serial_weighted_rigid_align
from boltz.testing.utils import spawn_multiprocessing


def _build_alignment_inputs(
    *,
    B: int,
    N_atoms: int,
    cp_size: int,
    n_pad_per_rank: int,
    nontrivial_rotation_translation: bool,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
):
    """Build (true_coords, pred_coords, weights, mask) globals for one test case.

    If ``nontrivial_rotation_translation`` is False, ``pred_coords`` is
    drawn from an independent random distribution — this is the standard
    parity check used by the 2D-CP test.

    If True, ``pred_coords = R @ true_coords + t`` for a deterministic
    non-identity rotation ``R`` (rotation by 0.7 rad around the z-axis)
    and translation ``t = (5, -3, 2)``.  ``weighted_rigid_align`` recovers
    the inverse transform that maps true onto pred, so the aligned-true
    output is highly sensitive to any off-by-rank centroid or covariance
    aggregation bug — without correct cp-axis all-reduce, the recovered
    rotation/translation diverges from the serial reference.

    ``N_atoms`` MUST be a multiple of ``cp_size`` so atom-dim sharding is
    even.  ``n_pad_per_rank`` is the number of trailing zero-mask atoms
    per cp rank; setting > 0 exercises the mask-aware centroid path.
    """
    if N_atoms % cp_size != 0:
        raise ValueError(f"N_atoms={N_atoms} must be divisible by cp_size={cp_size}")
    n_per_rank = N_atoms // cp_size
    if n_pad_per_rank >= n_per_rank:
        raise ValueError(f"n_pad_per_rank={n_pad_per_rank} must be < n_per_rank={n_per_rank}")

    true_coords = torch.randn((B, N_atoms, 3), dtype=dtype, device=device, generator=generator) * 2.0 + 10.0

    if nontrivial_rotation_translation:
        theta = torch.tensor(0.7, dtype=dtype, device=device)
        c, s = theta.cos(), theta.sin()
        R = torch.stack(
            [
                torch.stack([c, -s, torch.zeros_like(c)]),
                torch.stack([s, c, torch.zeros_like(c)]),
                torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
            ]
        )
        t = torch.tensor([5.0, -3.0, 2.0], dtype=dtype, device=device)
        # pred = (R @ true.T).T + t = true @ R.T + t
        pred_coords = true_coords @ R.T + t
    else:
        pred_coords = torch.randn((B, N_atoms, 3), dtype=dtype, device=device, generator=generator) * 5.0 + 15.0

    # Mask: per-rank shape (B, cp_size, n_per_rank) with trailing zeros, then reshape.
    mask_per_rank = torch.zeros((B, cp_size, n_per_rank), dtype=dtype, device=device)
    if n_pad_per_rank > 0:
        mask_per_rank[:, :, :-n_pad_per_rank] = 1.0
    else:
        mask_per_rank[:, :, :] = 1.0
    mask = mask_per_rank.reshape(B, N_atoms)
    weights = mask.clone()

    return true_coords, pred_coords, weights, mask


def _parallel_assert_weighted_rigid_align_1d(
    rank: int,
    grid_group_sizes,
    device_type: str,
    backend: str,
    env_per_rank: Optional[dict],
    dtype: torch.dtype,
    B: int,
    N_atoms: int,
    n_pad_per_rank: int,
    nontrivial_rt: bool,
    true_coords_host: torch.Tensor,
    pred_coords_host: torch.Tensor,
    weights_host: torch.Tensor,
    mask_host: torch.Tensor,
    expected_aligned_host: torch.Tensor,
):
    """Per-rank worker: compare ``weighted_rigid_align_1d`` against serial."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
            else:
                monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    device = manager.device
    device_mesh = manager.device_mesh

    assert device_mesh.ndim == 2, f"Expected 2D (dp, cp) mesh for 1D-CP, got ndim={device_mesh.ndim}"
    assert device_mesh.mesh_dim_names == ("dp", "cp")
    cp_size = device_mesh.size(1)

    placements = (Shard(0), Shard(1))

    true_coords_global = true_coords_host.to(device=device, dtype=dtype)
    pred_coords_global = pred_coords_host.to(device=device, dtype=dtype)
    weights_global = weights_host.to(device=device, dtype=dtype)
    mask_global = mask_host.to(device=device, dtype=dtype)
    expected_aligned = expected_aligned_host.to(device=device, dtype=dtype)

    true_dt = distribute_tensor(true_coords_global, device_mesh, placements)
    pred_dt = distribute_tensor(pred_coords_global, device_mesh, placements)
    weights_dt = distribute_tensor(weights_global, device_mesh, placements)
    mask_dt = distribute_tensor(mask_global, device_mesh, placements)

    # Non-vacuous: confirm atom-axis sharding is active when cp_size > 1.
    if cp_size > 1:
        assert true_dt.to_local().shape[1] < true_coords_global.shape[1], (
            f"Rank {rank}: true_coords CP shard atom dim {true_dt.to_local().shape[1]} "
            f"not smaller than global {true_coords_global.shape[1]}; sharding inactive."
        )

    aligned_dt = weighted_rigid_align_1d(true_dt, pred_dt, weights_dt, mask_dt)

    # Output placements must match the input.
    assert (
        aligned_dt.placements == placements
    ), f"Rank {rank}: aligned_dt placements {aligned_dt.placements} != expected {placements}"
    assert (
        aligned_dt.shape == true_coords_global.shape
    ), f"Rank {rank}: aligned_dt shape {aligned_dt.shape} != true_coords global shape {true_coords_global.shape}"
    assert isinstance(aligned_dt, DTensor)

    # Parity: 1D-CP output must match serial reference globally.
    aligned_full = aligned_dt.full_tensor()
    torch.testing.assert_close(
        aligned_full,
        expected_aligned,
        msg=lambda m: (
            f"Rank {rank}: weighted_rigid_align_1d global output disagrees with serial "
            f"weighted_rigid_align (nontrivial_rt={nontrivial_rt}): {m}"
        ),
    )

    # Local-shard parity: distribute the serial reference using the same
    # placements and compare per-rank shards. This catches per-rank
    # transpose / sign / permutation bugs that would still average out
    # to the right global tensor (e.g., a swapped chunk-pair).
    expected_aligned_dt = distribute_tensor(expected_aligned, device_mesh, placements)
    torch.testing.assert_close(
        aligned_dt.to_local(),
        expected_aligned_dt.to_local(),
        msg=lambda m: f"Rank {rank}: weighted_rigid_align_1d local shard mismatch (nontrivial_rt={nontrivial_rt}): {m}",
    )

    # Round-trip identity sanity for the nontrivial_rt case: with weights = mask
    # and a true rigid pred->true mapping, aligned_true should be close to pred.
    # weighted_rigid_align recovers the rotation that maps pred onto true, then
    # returns R @ (true - true_centroid) + pred_centroid — which equals pred up
    # to floating-point and SVD noise when pred is a rigid transform of true and
    # all atoms are unmasked. (Skip when any padding is masked out, since the
    # alignment is then mask-weighted rather than full.)
    if nontrivial_rt and n_pad_per_rank == 0:
        torch.testing.assert_close(
            aligned_full,
            pred_coords_global,
            msg=lambda m: (
                f"Rank {rank}: nontrivial-rigid round-trip mismatch — aligned_true should equal pred "
                f"when pred = R @ true + t and no padding: {m}"
            ),
        )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "nontrivial_rt",
    [False, True],
    ids=["random_pred", "rigid_rt"],
)
@pytest.mark.parametrize(
    "n_pad_per_rank",
    [0, 2],
    ids=["no_pad", "pad2"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        # CUDA dp=1 cp=1: serial-equivalent sanity on 2D mesh (1 GPU).
        ((1, 1), True, "cuda", "ENV"),
        # CUDA dp=1 cp=2: minimal 1D-CP shard count (2 GPUs).
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp1", "cuda-dp1-cp2"],
)
def test_weighted_rigid_align_1d(setup_env, nontrivial_rt, n_pad_per_rank):
    """1D-CP parity test for ``weighted_rigid_align_1d`` vs serial.

    Parametrized over a 2-axis ``(dp, cp)`` mesh (cp=1 sanity, cp=2 minimal
    1D-CP shard count), padding (none vs trailing 2 atoms per rank), and
    pred-coords construction (random vs deterministic rigid transform of
    true coords). The ``rigid_rt`` × ``no_pad`` combination additionally
    asserts the round-trip identity ``aligned_true == pred`` up to
    floating-point noise; this catches per-rank cp-axis aggregation bugs
    that would still produce a valid rotation but a wrong one.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = int(grid_group_sizes["cp"])
    dp_size = int(grid_group_sizes["dp"])
    B = 2 * dp_size  # at least 2 per DP rank to keep mask-aware path exercised
    N_atoms = 32 * cp_size  # 32 per rank — small enough to keep tests fast, big enough to be non-degenerate

    # fp32: matches the dtype used in the existing 2D-CP test and the production
    # `AtomDiffusion.sample` call site. The SVD inside `weighted_rigid_align_1d`
    # is promoted to fp64 internally, so fp32 is the right outer dtype to test.
    dtype = torch.float32

    # Host-side global tensors built with a contained generator (no global RNG drift).
    cpu_device = torch.device("cpu")
    gen = torch.Generator(device=cpu_device).manual_seed(20260515)
    true_coords, pred_coords, weights, mask = _build_alignment_inputs(
        B=B,
        N_atoms=N_atoms,
        cp_size=cp_size,
        n_pad_per_rank=n_pad_per_rank,
        nontrivial_rotation_translation=nontrivial_rt,
        device=cpu_device,
        dtype=dtype,
        generator=gen,
    )

    # Serial reference on host (will be moved to device per-rank).
    expected_aligned = serial_weighted_rigid_align(
        true_coords.to(cpu_device),
        pred_coords.to(cpu_device),
        weights.to(cpu_device),
        mask.to(cpu_device),
    ).detach()

    spawn_multiprocessing(
        _parallel_assert_weighted_rigid_align_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        B,
        N_atoms,
        n_pad_per_rank,
        nontrivial_rt,
        true_coords.detach(),
        pred_coords.detach(),
        weights.detach(),
        mask.detach(),
        expected_aligned,
    )
