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

"""Regression test for ``pack_atom_features`` under 1D CP topology.

This test pins the fix from commit ``ac8d1c0c`` (``fix(1d-cp): no
atom_to_token offset under 1D-CP topology``). The bug was a latent
divergence between the 2D-CP and 1D-CP semantics of the local
``atom_to_token`` one-hot:

* **2D-CP (3D mesh ``(dp, cp_axis_0, cp_axis_1)``):** the local one-hot
  is block-diagonal with last-dim ``N_tokens_per_shard``, so
  ``shardwise_argmax`` produces shard-local indices that must be offset
  by ``rank * N_tokens_per_shard`` to recover global token IDs.
* **1D-CP (2D mesh ``(dp, cp)``):** the local one-hot has last-dim
  ``N_tokens_global`` (token dim is replicated across cp ranks), so
  ``shardwise_argmax`` already yields GLOBAL indices and any nonzero
  offset shifts rank>=1 token IDs out of range.

Before the fix, ``pack_atom_features`` unconditionally applied the
2D-CP offset under 1D-CP, corrupting every atom->token scatter
downstream (predict-1d at cp=2 produced ``matched_lddt=0.347`` vs the
golden ``0.968``). The bug hid for ~30 commits because the existing
:mod:`tests/distributed/data/test_dtensor_pack_and_pad_atom_features`
test exercises only the 3D mesh.

The assertions below are designed so that reverting the fix in
``src/boltz/distributed/data/feature/featurizer.py`` (lines ~600-610)
to the unconditional ``offset_per_rank = n_tokens_per_shard`` would
fail the in-range check on every rank with ``cp >= 2``.
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from boltz.distributed.data.feature.featurizer import pack_atom_features
from boltz.distributed.manager import DistributedManager
from boltz.testing.utils import spawn_multiprocessing

# Atom feature keys that pack_atom_features consumes.
_ATOM_FEATURE_KEYS_PACKED = {
    "atom_pad_mask",
    "atom_to_token",
    "ref_pos",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
}


def _build_1d_cp_feats(
    *,
    B_global: int,
    N_atoms_global: int,
    N_tokens_global: int,
    cp_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Build global feature tensors with the layout expected by pack_atom_features.

    All tensors are returned on ``device`` and have batch B on dim 0,
    N_atoms on dim 1, and (for ``atom_to_token``) ``N_tokens_global`` on
    dim 2.  Atom-to-token assignment is constructed by partitioning the
    atom range into contiguous segments, one per token, so the argmax
    yields a known-correct serial reference.

    ``N_atoms_global`` MUST be a multiple of ``cp_size`` so the
    ``(Shard(0), Shard(1))`` distribution shards the atom dim evenly.

    Parameters
    ----------
    B_global, N_atoms_global, N_tokens_global : int
        Global feature dimensions.
    cp_size : int
        Number of CP ranks (used only for divisibility validation).
    device, dtype, generator
        Tensor construction targets and contained RNG source.
    """
    if N_atoms_global % cp_size != 0:
        raise ValueError(f"N_atoms_global={N_atoms_global} must be divisible by cp_size={cp_size}")
    if N_atoms_global < N_tokens_global:
        raise ValueError(
            f"N_atoms_global={N_atoms_global} must be >= N_tokens_global={N_tokens_global} "
            "so every token receives at least one atom."
        )

    # Build a deterministic atom->token assignment by chunking the atom
    # range into N_tokens contiguous segments.  segment_lengths are
    # round-robin assigned to balance shards, then `cumsum` gives
    # per-token atom boundaries.
    base = N_atoms_global // N_tokens_global
    rem = N_atoms_global - base * N_tokens_global
    atom_counts_per_token = torch.full((N_tokens_global,), base, dtype=torch.long, device=device)
    if rem > 0:
        atom_counts_per_token[:rem] += 1
    assert int(atom_counts_per_token.sum().item()) == N_atoms_global

    # token_id_per_atom[a] = which token owns atom a (shape: N_atoms_global)
    token_id_per_atom = torch.repeat_interleave(
        torch.arange(N_tokens_global, dtype=torch.long, device=device),
        atom_counts_per_token,
    )
    assert token_id_per_atom.shape == (N_atoms_global,)

    # Construct atom_to_token as a one-hot of token_id_per_atom (per batch sample).
    atom_to_token_per_sample = torch.nn.functional.one_hot(token_id_per_atom, num_classes=N_tokens_global)
    atom_to_token = atom_to_token_per_sample.unsqueeze(0).expand(B_global, -1, -1).contiguous().to(dtype=dtype)

    # atom_pad_mask: all real atoms (no padding) so every entry of
    # atom_to_token_ids_global is a valid token index.
    atom_pad_mask = torch.ones((B_global, N_atoms_global), dtype=torch.bool, device=device)

    # ref_* features: any plausible-dtype tensor whose first two dims
    # match (B, N_atoms_global).  pack_atom_features only re-packs them
    # — values are not under test here.
    ref_pos = torch.randn((B_global, N_atoms_global, 3), dtype=dtype, device=device, generator=generator)
    ref_charge = torch.randn((B_global, N_atoms_global), dtype=dtype, device=device, generator=generator)
    # ref_element/ref_atom_name_chars/ref_space_uid are integer-coded in production;
    # pack_atom_features treats them as opaque payload, so the exact dtype just needs to round-trip.
    ref_element = torch.randint(
        0, 128, (B_global, N_atoms_global), dtype=torch.long, device=device, generator=generator
    )
    ref_atom_name_chars = torch.randint(
        0, 64, (B_global, N_atoms_global, 4), dtype=torch.long, device=device, generator=generator
    )
    ref_space_uid = torch.randint(
        0, 1024, (B_global, N_atoms_global), dtype=torch.long, device=device, generator=generator
    )

    return {
        "atom_pad_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
    }


def _parallel_assert_pack_atom_features_1d(
    rank: int,
    grid_group_sizes,
    device_type: str,
    backend: str,
    env_per_rank,
    dtype: torch.dtype,
    W: int,
    B_global: int,
    N_atoms_global: int,
    N_tokens_global: int,
    feats_global_host: dict[str, torch.Tensor],
):
    """Per-rank worker for the 1D-CP ``pack_atom_features`` regression test."""
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
    # The 1D-CP signature this test exercises: a 2D ``(dp, cp)`` mesh.
    # (2D-CP has a 3D mesh and is covered by test_dtensor_pack_and_pad_atom_features.)
    assert device_mesh.ndim == 2, f"Expected 2D (dp, cp) mesh, got ndim={device_mesh.ndim}"
    assert device_mesh.mesh_dim_names == (
        "dp",
        "cp",
    ), f"Expected mesh dim names ('dp', 'cp'), got {device_mesh.mesh_dim_names}"
    cp_size = device_mesh.size(1)
    dp_size = device_mesh.size(0)

    # Build 1D-CP DTensors with placements (Shard(0), Shard(1)):
    # - Shard(0) on dp axis -> batch B is split across DP ranks
    # - Shard(1) on cp axis -> N_atoms is split across CP ranks
    # The token dim (dim 2 of atom_to_token) is NOT sharded — this is the
    # core distinction from 2D-CP that the fix at ac8d1c0c addresses.
    placements_single = (Shard(0), Shard(1))
    placements_pair = (Shard(0), Shard(1))  # atom_to_token: shard atom, replicate token

    feats_dt: dict[str, DTensor] = {}
    for key in sorted(feats_global_host.keys()):
        v = feats_global_host[key].to(device=device)
        if v.dtype.is_floating_point and v.dtype != dtype:
            v = v.to(dtype=dtype)
        if key == "atom_to_token":
            # 3D tensor; Shard(1) puts atom dim on cp, token dim (dim 2) replicated.
            placement = placements_pair
        else:
            placement = placements_single
        feats_dt[key] = distribute_tensor(v, device_mesh, placement)

    # Non-vacuous: confirm sharding is active on the atom axis when cp_size > 1.
    atom_to_token_dt = feats_dt["atom_to_token"]
    if cp_size > 1:
        local_atom_dim = atom_to_token_dt.to_local().shape[1]
        assert local_atom_dim < N_atoms_global, (
            f"Rank {rank}: atom_to_token CP shard atom dim {local_atom_dim} not smaller than "
            f"global {N_atoms_global}; sharding inactive."
        )
    # Critical: confirm token dim of the local one-hot is the FULL global size,
    # not a per-shard slice. The fix at ac8d1c0c depends on this property to
    # decide that no per-rank offset is needed.
    assert atom_to_token_dt.to_local().shape[2] == N_tokens_global, (
        f"Rank {rank}: atom_to_token local token-dim size {atom_to_token_dt.to_local().shape[2]} != "
        f"global {N_tokens_global}; 1D-CP layout invariant violated, test setup is incorrect."
    )
    if dp_size > 1:
        local_batch_dim = atom_to_token_dt.to_local().shape[0]
        assert local_batch_dim < B_global, (
            f"Rank {rank}: atom_to_token DP shard batch dim {local_batch_dim} not smaller than "
            f"global {B_global}; DP sharding inactive."
        )

    # =====================================================================
    # Call pack_atom_features.  Under 1D-CP, this must produce
    # atom_to_token_ids_global with every entry in [0, N_tokens_global).
    # =====================================================================
    feats_packed = pack_atom_features(feats_dt, _ATOM_FEATURE_KEYS_PACKED, W)

    assert (
        "atom_to_token_ids_global" in feats_packed
    ), "pack_atom_features must return 'atom_to_token_ids_global' for atom_to_token inputs."
    a2t_ids_global = feats_packed["atom_to_token_ids_global"]

    # =====================================================================
    # Primary regression assertion: every rank's local atom_to_token_ids_global
    # values must lie within the valid token-index range. Under the pre-fix
    # bug, rank=1 of cp=2 produced indices in [N_tokens_global, 2*N_tokens_global)
    # because of the spurious offset.
    # =====================================================================
    a2t_ids_local = a2t_ids_global.to_local()
    atom_mask_local = feats_packed["atom_pad_mask"].to_local().bool()

    # Compare shapes (local pad mask vs ids share atom-dim layout after pack).
    assert a2t_ids_local.shape == atom_mask_local.shape, (
        f"Rank {rank}: a2t_ids_local.shape={a2t_ids_local.shape} != " f"atom_mask_local.shape={atom_mask_local.shape}"
    )

    if atom_mask_local.any():
        valid_ids = a2t_ids_local[atom_mask_local]
        min_id = int(valid_ids.min().item())
        max_id = int(valid_ids.max().item())
        assert min_id >= 0, (
            f"Rank {rank}: pack_atom_features produced atom_to_token_ids_global with min={min_id} "
            f"(< 0). This is the ac8d1c0c regression: 1D-CP applied a 2D-CP shard offset."
        )
        assert max_id < N_tokens_global, (
            f"Rank {rank}: pack_atom_features produced atom_to_token_ids_global with max={max_id} "
            f">= N_tokens_global={N_tokens_global}. This is the ac8d1c0c regression: "
            f"1D-CP unconditionally applied 2D-CP's per-rank token offset, shifting rank>=1 "
            f"indices out of range."
        )

    # =====================================================================
    # Secondary parity assertion: the gathered global indices match the
    # serial argmax of the global one-hot in the valid (non-pad) region.
    # =====================================================================
    a2t_ids_global_full = a2t_ids_global.full_tensor()  # (B_global, N_atoms_packed)
    atom_mask_full = feats_packed["atom_pad_mask"].full_tensor().bool()  # (B_global, N_atoms_packed)

    serial_a2t_ids = feats_global_host["atom_to_token"].to(device=device).argmax(dim=-1)  # (B_global, N_atoms_global)
    serial_atom_pad_mask = feats_global_host["atom_pad_mask"].to(device=device).bool()

    # The packed atom axis is >= N_atoms_global (trailing padding to W * cp).
    # Compare ids only at positions where the packed pad mask is True; the
    # number of True positions equals the number of real atoms.
    assert a2t_ids_global_full.shape[0] == B_global
    assert a2t_ids_global_full.shape[1] >= N_atoms_global
    assert atom_mask_full.shape == a2t_ids_global_full.shape

    for b in range(B_global):
        packed_valid_ids = a2t_ids_global_full[b][atom_mask_full[b]]
        serial_valid_ids = serial_a2t_ids[b][serial_atom_pad_mask[b]]
        assert packed_valid_ids.shape == serial_valid_ids.shape, (
            f"Rank {rank} sample {b}: packed has {packed_valid_ids.shape[0]} valid atoms, "
            f"serial has {serial_valid_ids.shape[0]}"
        )
        torch.testing.assert_close(
            packed_valid_ids,
            serial_valid_ids,
            msg=lambda m, _b=b: (
                f"Rank {rank} sample {_b}: pack_atom_features atom_to_token_ids_global "
                f"disagrees with serial argmax of atom_to_token one-hot: {m}"
            ),
        )

    # =====================================================================
    # Tertiary assertion: round-trip via a serial scatter into a
    # [B_global, N_tokens_global] buffer must match a serial reference.
    # This catches the case where indices happen to land in [0, N_tokens_global)
    # by chance but still permute atoms across the wrong token bins.
    # =====================================================================
    # scatter atoms -> per-token atom counts, using packed (1D-CP) ids
    cnt_packed = torch.zeros((B_global, N_tokens_global), dtype=torch.long, device=device)
    cnt_serial = torch.zeros((B_global, N_tokens_global), dtype=torch.long, device=device)
    for b in range(B_global):
        packed_valid_ids = a2t_ids_global_full[b][atom_mask_full[b]].to(torch.long)
        serial_valid_ids = serial_a2t_ids[b][serial_atom_pad_mask[b]].to(torch.long)
        ones_packed = torch.ones_like(packed_valid_ids)
        ones_serial = torch.ones_like(serial_valid_ids)
        cnt_packed[b].scatter_add_(0, packed_valid_ids, ones_packed)
        cnt_serial[b].scatter_add_(0, serial_valid_ids, ones_serial)
    torch.testing.assert_close(
        cnt_packed,
        cnt_serial,
        msg=lambda m: f"Rank {rank}: per-token atom-count scatter disagrees with serial reference: {m}",
    )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "N_tokens_global",
    [
        # Aligned case: N_tokens_global divisible by every cp_size in the
        # mesh parametrize axis (1, 2, 4). Standard regression check.
        16,
        # Adversarial: N_tokens_global is PRIME, indivisible by cp_size>1.
        # A hypothetical "fix" that replaced the existing
        # ``offset_per_rank = 0`` 1D-CP branch with
        # ``offset_per_rank = N_tokens_global // cp_size``  — a tempting
        # but wrong refactor under the false assumption that the token dim
        # is sharded — would silently truncate to ``17 // 2 = 8`` and shift
        # rank>=1 ids by 8 instead of 0, producing a permutation bug that
        # the aligned (16) case happens to leave invisible because
        # ``offset_per_rank = N_tokens_global // 2 == n_atoms_per_token == 4``
        # accidentally lines up with the chunk-of-4 atom-to-token layout.
        17,
    ],
    ids=["N_tokens=16(aligned)", "N_tokens=17(adversarial-prime)"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        # CUDA dp=1 cp=1: trivial 1D-CP mesh sanity (1 GPU).
        ((1, 1), True, "cuda", "ENV"),
        # CUDA dp=1 cp=2: minimal CP shard count exposing the regression (2 GPUs).
        ((1, 2), True, "cuda", "ENV"),
        # CUDA dp=2 cp=2: DP + 1D-CP combined (4 GPUs).
        ((2, 2), True, "cuda", "ENV"),
        # CUDA dp=1 cp=4: larger CP shard count (4 GPUs).
        ((1, 4), True, "cuda", "ENV"),
        # CPU dp=1 cp=2: gloo-backed CP=2 path (no NCCL all_to_all required by
        # pack_atom_features / distributed_pack_and_pad).  Keeps the suite
        # runnable on CPU-only CI as one cheap parametrization.
        ((1, 2), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp1", "cuda-dp1-cp2", "cuda-dp2-cp2", "cuda-dp1-cp4", "cpu-dp1-cp2"],
)
def test_pack_atom_features_1d(setup_env, N_tokens_global: int):
    """1D-CP regression test for ``pack_atom_features`` (commit ac8d1c0c).

    Asserts that under a 2D ``(dp, cp)`` device mesh:

    * Every rank's local ``atom_to_token_ids_global`` values lie in
      ``[0, N_tokens_global)``. The pre-fix bug produced out-of-range
      indices on every cp rank >= 1.
    * Packed global indices match a serial argmax of the global
      one-hot ``atom_to_token`` in the valid (non-pad) region.
    * A serial scatter of atoms into a ``[B, N_tokens_global]`` count
      buffer matches the serial reference (catches permutation bugs
      that happen to keep indices in range).
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = int(grid_group_sizes["cp"])
    dp_size = int(grid_group_sizes["dp"])

    # Test dimensions: keep small but non-trivial.
    # - N_atoms_global divisible by cp_size for even atom-axis sharding.
    # - N_tokens_global parametrized (16 aligned vs 17 adversarial-prime):
    #   the prime variant catches a hypothetical ``// cp_size`` "fix" that
    #   would happen to line up with the aligned case (see N_tokens_global
    #   parametrize doc above).
    # - N_atoms_global tuned so each token receives multiple atoms (catches
    #   permutation bugs that single-atom-per-token cases would miss).
    B_global = dp_size  # one sample per DP rank
    n_atoms_per_token = 4
    # Round N_atoms_global up to the LCM of (cp_size, N_tokens_global) so
    # the atom dim is always evenly cp-shardable AND every token gets >=1 atom.
    raw_atoms = N_tokens_global * n_atoms_per_token
    if raw_atoms % cp_size != 0:
        raw_atoms += cp_size - (raw_atoms % cp_size)
    N_atoms_global = raw_atoms
    assert N_atoms_global % cp_size == 0, f"N_atoms_global={N_atoms_global} must be divisible by cp_size={cp_size}"
    W = 16  # atoms per window for queries; small to keep packed shape compact

    # Build features on host with a contained generator (avoid global RNG drift).
    gen = torch.Generator(device="cpu").manual_seed(20260515)
    # Build on CPU first, then ship to ranks; spawn workers re-route to their device.
    cpu_device = torch.device("cpu")
    feats_global = _build_1d_cp_feats(
        B_global=B_global,
        N_atoms_global=N_atoms_global,
        N_tokens_global=N_tokens_global,
        cp_size=cp_size,
        device=cpu_device,
        dtype=torch.float32,
        generator=gen,
    )
    feats_global_host = {k: v.detach().clone() for k, v in feats_global.items()}

    spawn_multiprocessing(
        _parallel_assert_pack_atom_features_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        torch.float32,
        W,
        B_global,
        N_atoms_global,
        N_tokens_global,
        feats_global_host,
    )
