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

"""DTensor parity tests for ``minimum_lddt_symmetry_coords_1d``.

Tests the 1D CP symmetry correction at
``src/boltz/distributed/data/feature/symmetry.py`` against the serial
Boltz-2 ``minimum_lddt_symmetry_coords`` reference from ``boltz.data.mol``.

The 1D CP topology uses a 2D ``(dp, cp)`` device mesh with placements
``(Shard(0), Shard(1))`` on coords and the atom-pad mask. The serial
reference is the same single-rank function the 2D test compares against,
because the symmetry search itself is topology-independent — only the
gather/re-scatter machinery differs between the 1D and 2D CP variants.
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from boltz.data.mol import minimum_lddt_symmetry_coords as serial_minimum_lddt_symmetry_coords
from boltz.distributed.data.feature.symmetry import minimum_lddt_symmetry_coords_1d
from boltz.distributed.manager import DistributedManager
from boltz.testing.utils import spawn_multiprocessing


def _build_symmetry_features_for_batch(
    all_coords_batch: torch.Tensor,
    all_resolved_mask_batch: torch.Tensor,
    crop_to_all_atom_map_batch: torch.Tensor,
    chain_swaps_batch: list | None = None,
):
    """Build symmetry features for a batch of samples.

    Parameters
    ----------
    chain_swaps_batch : list or None
        If None, each sample gets a single identity (no-op) chain swap
        ``[[]]``. Otherwise, provide a per-sample list of swap combinations.

    """
    batch_size = all_coords_batch.shape[0]
    if chain_swaps_batch is None:
        chain_swaps_batch = [[[]] for _ in range(batch_size)]
    amino_acids_symmetries_batch = [[] for _ in range(batch_size)]
    ligand_symmetries_batch = [[] for _ in range(batch_size)]

    feats = {
        "all_coords": all_coords_batch,
        "all_resolved_mask": all_resolved_mask_batch,
        "crop_to_all_atom_map": crop_to_all_atom_map_batch,
        "chain_swaps": chain_swaps_batch,
        "amino_acids_symmetries": amino_acids_symmetries_batch,
        "ligand_symmetries": ligand_symmetries_batch,
    }
    return feats


def _make_two_chain_swaps(n_atoms: int) -> list:
    """Build ``chain_swaps`` for one sample with two equal-length swappable chains.

    Splits the atom range ``[0, n_atoms)`` into two halves (chain A and
    chain B) and returns identity + the A<->B swap. Each swap entry is
    ``(start1, end1, start2, end2, chainidx1, chainidx2)``.
    """
    half = n_atoms // 2
    identity: list = []
    swap_ab = [
        (0, half, half, 2 * half, 0, 1),
        (half, 2 * half, 0, half, 1, 0),
    ]
    return [identity, swap_ab]


def _worker_minimum_lddt_symmetry_coords_1d(rank, payload):
    """Worker: compare 1D CP ``minimum_lddt_symmetry_coords_1d`` to serial.

    The serial reference is invoked on the test driver with per-sample
    coords sliced from ``sample_coords_global``. Each DP rank owns
    ``B_local`` samples and the CP axis shards the atom dimension.
    """
    (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        atom_pad_mask_global_host,
        sample_coords_global_host,
        feats_symmetry_global,
        expected_true_coords_per_sample,
        expected_true_mask_per_sample,
    ) = payload

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    device = manager.device
    dtype = torch.float32

    device_mesh = manager.device_mesh  # (dp, cp)
    assert device_mesh.mesh_dim_names == ("dp", "cp"), f"Expected ('dp', 'cp') mesh, got {device_mesh.mesh_dim_names}"

    placements = (Shard(0), Shard(1))

    coords_global = sample_coords_global_host.to(device=device, dtype=dtype)
    atom_pad_mask_global = atom_pad_mask_global_host.to(device=device)

    coords_dtensor = distribute_tensor(coords_global, device_mesh, placements)
    atom_pad_mask_dtensor = distribute_tensor(atom_pad_mask_global, device_mesh, placements)

    # Non-vacuous: confirm sharding is active (local shape < global shape on sharded dims).
    cp_size = device_mesh.size(1)
    dp_size = device_mesh.size(0)
    if cp_size > 1:
        assert coords_dtensor.to_local().shape[1] < coords_global.shape[1], (
            f"Rank {rank}: coords CP shard size {coords_dtensor.to_local().shape[1]} "
            f"is not smaller than global {coords_global.shape[1]}; sharding inactive."
        )
    if dp_size > 1:
        assert coords_dtensor.to_local().shape[0] < coords_global.shape[0], (
            f"Rank {rank}: coords DP shard size {coords_dtensor.to_local().shape[0]} "
            f"is not smaller than global {coords_global.shape[0]}; sharding inactive."
        )

    num_dp_ranks = grid_group_sizes["dp"]
    global_batch_size = sample_coords_global_host.shape[0]
    local_batch_size = global_batch_size // num_dp_ranks
    rank_dp = manager.group_rank["dp"]
    local_start = rank_dp * local_batch_size
    local_end = local_start + local_batch_size

    feats_local = {
        "all_coords": feats_symmetry_global["all_coords"][local_start:local_end].to(device),
        "all_resolved_mask": feats_symmetry_global["all_resolved_mask"][local_start:local_end].to(device),
        "crop_to_all_atom_map": feats_symmetry_global["crop_to_all_atom_map"][local_start:local_end].to(device),
        "chain_swaps": feats_symmetry_global["chain_swaps"][local_start:local_end],
        "amino_acids_symmetries": feats_symmetry_global["amino_acids_symmetries"][local_start:local_end],
        "ligand_symmetries": feats_symmetry_global["ligand_symmetries"][local_start:local_end],
        "atom_pad_mask": atom_pad_mask_dtensor,
    }

    for i_batch_local in range(local_batch_size):
        global_batch_idx = local_start + i_batch_local

        true_coords_dtensor, true_mask_dtensor = minimum_lddt_symmetry_coords_1d(
            coords=coords_dtensor,
            feats=feats_local,
            index_batch_local=i_batch_local,
            i_batch_multiplicity_local=i_batch_local,
        )

        assert true_coords_dtensor.placements == placements, (
            f"Rank {rank} sample {i_batch_local}: true_coords placements "
            f"{true_coords_dtensor.placements} != expected {placements}"
        )
        assert true_mask_dtensor.placements == placements, (
            f"Rank {rank} sample {i_batch_local}: true_mask placements "
            f"{true_mask_dtensor.placements} != expected {placements}"
        )

        # Gather along CP only, keep DP sharding for direct comparison.
        coords_cp_gathered = true_coords_dtensor.redistribute(
            device_mesh,
            (true_coords_dtensor.placements[0], Replicate()),
        ).to_local()
        mask_cp_gathered = true_mask_dtensor.redistribute(
            device_mesh,
            (true_mask_dtensor.placements[0], Replicate()),
        ).to_local()
        atom_pad_mask_gathered = atom_pad_mask_dtensor.redistribute(
            device_mesh,
            (atom_pad_mask_dtensor.placements[0], Replicate()),
        ).to_local()

        real_atom_mask = atom_pad_mask_gathered[i_batch_local].bool()
        coords_no_pad = coords_cp_gathered[i_batch_local, real_atom_mask, :]
        mask_no_pad = mask_cp_gathered[i_batch_local, real_atom_mask]

        expected_coords = expected_true_coords_per_sample[global_batch_idx]
        expected_mask = expected_true_mask_per_sample[global_batch_idx]

        torch.testing.assert_close(
            coords_no_pad.cpu(),
            expected_coords,
            msg=f"Rank {rank} sample {i_batch_local} (global {global_batch_idx}): true_coords mismatch",
        )
        torch.testing.assert_close(
            mask_no_pad.cpu(),
            expected_mask.squeeze(0) if expected_mask.ndim > 1 else expected_mask,
            msg=f"Rank {rank} sample {i_batch_local} (global {global_batch_idx}): true_mask mismatch",
        )

        # Non-vacuous: mask must have at least one resolved atom (input is all-resolved).
        assert mask_no_pad.any(), (
            f"Rank {rank} sample {i_batch_local} (global {global_batch_idx}): "
            f"true_mask is all-zero, but the input feature set is all-resolved."
        )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "use_nontrivial_swaps",
    [False, True],
    ids=["identity_swaps", "nontrivial_swaps"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        # CUDA dp=1 cp=1: serial-equivalent sanity (1 GPU); validates the trivial-mesh path
        ((1, 1), True, "cuda", "ENV"),
        # CUDA dp=1 cp=2: minimal CP shard test (2 GPUs)
        ((1, 2), True, "cuda", "ENV"),
        # CUDA dp=2 cp=2: DP + CP combined (4 GPUs); exercises per-DP local-batch indexing
        ((2, 2), True, "cuda", "ENV"),
        # CUDA dp=1 cp=4: larger CP shard count (4 GPUs)
        ((1, 4), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp1", "cuda-dp1-cp2", "cuda-dp2-cp2", "cuda-dp1-cp4"],
)
def test_dtensor_minimum_lddt_symmetry_coords_1d(setup_env, use_nontrivial_swaps):
    """1D CP symmetry correction parity against Boltz-2 serial reference.

    Parametrized over flat ``(dp, cp)`` tuples and over identity vs.
    non-trivial (two-chain A<->B) chain swaps. The non-trivial swap case
    catches plausible implementation bugs such as wiring chain1/chain2
    halves in the wrong direction, since the serial reference re-orders
    the atom array according to the requested swap.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = int(grid_group_sizes["cp"])
    num_dp_ranks = int(grid_group_sizes["dp"])
    batch_size_per_dp_rank = 1
    batch_size_global = batch_size_per_dp_rank * num_dp_ranks

    # Choose n_atoms divisible by cp_size and even (so two halves swap cleanly).
    n_atoms = 2 * cp_size * 6  # e.g. cp=2 -> 24, cp=3 -> 36, cp=4 -> 48

    gen = torch.Generator(device="cpu").manual_seed(42)
    sample_coords_global = torch.randn(
        (batch_size_global, n_atoms, 3),
        dtype=torch.float32,
        generator=gen,
    )
    atom_pad_mask_global = torch.ones((batch_size_global, n_atoms), dtype=torch.bool)

    all_coords_global = sample_coords_global.clone()
    all_resolved_mask_global = torch.ones((batch_size_global, n_atoms), dtype=torch.bool)
    crop_to_all_atom_map_global = (
        torch.arange(n_atoms, dtype=torch.long).unsqueeze(0).expand(batch_size_global, -1).contiguous()
    )

    chain_swaps_batch = None
    if use_nontrivial_swaps:
        chain_swaps_batch = [_make_two_chain_swaps(n_atoms) for _ in range(batch_size_global)]

    feats_symmetry_global = _build_symmetry_features_for_batch(
        all_coords_global,
        all_resolved_mask_global,
        crop_to_all_atom_map_global,
        chain_swaps_batch=chain_swaps_batch,
    )

    expected_true_coords_per_sample = []
    expected_true_mask_per_sample = []
    for i in range(batch_size_global):
        expected_coords, expected_mask = serial_minimum_lddt_symmetry_coords(
            coords=sample_coords_global[i : i + 1],
            feats=feats_symmetry_global,
            index_batch=i,
        )
        expected_true_coords_per_sample.append(expected_coords.squeeze(0).detach().clone().cpu())
        expected_true_mask_per_sample.append(expected_mask.detach().clone().cpu())

    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        atom_pad_mask_global.detach().clone().cpu(),
        sample_coords_global.detach().clone().cpu(),
        feats_symmetry_global,
        expected_true_coords_per_sample,
        expected_true_mask_per_sample,
    )

    spawn_multiprocessing(_worker_minimum_lddt_symmetry_coords_1d, world_size, payload)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
