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

"""Automated mutation test for the ``pack_atom_features`` 1D-CP fix.

Companion to :mod:`tests/distributed/data/feature/test_dtensor_pack_atom_features_1d`,
which is the regression test that pins the fix from commit ``ac8d1c0c``
(``fix(1d-cp): no atom_to_token offset under 1D-CP topology``).

The regression test's docstring claims that reverting the fix would fail
its in-range invariant. That claim was previously procedural — a human
had to hand-edit ``src/boltz/distributed/data/feature/featurizer.py``,
re-run the test, and revert.  This module **automates** that verification
via runtime monkeypatching so the claim is checked on every test run.

Tracked in ``.claude/todo.md`` lines 64-72 ("Mutation test (procedural)
for tests/distributed/data/feature/test_dtensor_pack_atom_features_1d.py
(ac8d1c0c regression)").

What is mutated
---------------
The fix at ``src/boltz/distributed/data/feature/featurizer.py:597-610``:

.. code-block:: python

    n_tokens_per_shard = atom_to_token_dt.to_local().shape[2]
    atom_to_token_ids_local = shardwise_argmax(atom_to_token_dt, dim=-1, keepdim=False)
    if atom_to_token_dt.device_mesh.ndim == 2:
        offset_per_rank = 0          # 1D-CP path (the fix)
    else:
        offset_per_rank = n_tokens_per_shard  # 2D-CP path
    atom_to_token_ids_global = shardwise_offset(
        atom_to_token_ids_local, dim=1, offset_per_rank=offset_per_rank
    )

Two mutations are injected by monkeypatching the ``shardwise_offset`` symbol
inside :mod:`boltz.distributed.data.feature.featurizer` to override
``offset_per_rank`` at the call site:

* ``unconditional_n_tokens_per_shard`` — the original pre-``ac8d1c0c`` bug:
  ``offset_per_rank = n_tokens_per_shard`` unconditionally. Under 1D-CP
  ``n_tokens_per_shard == N_tokens_global`` (the token axis is replicated),
  so rank>=1 lands in ``[N_tokens_global, k*N_tokens_global)`` — every entry
  out of range.
* ``divide_by_cp_size`` — a tempting-but-wrong refactor of the 1D-CP branch
  that mistakes the replicated token dim for a sharded one:
  ``offset_per_rank = n_tokens_per_shard // cp_size``. For aligned
  ``N_tokens_global = 16`` and ``cp_size = 2`` this offsets rank 1 by 8 and
  produces a permutation that lands in range but wrong; the prime
  parametrization ``N_tokens_global = 17`` exposes the off-by-floor.

Expected behavior
-----------------
Under either mutation, the in-range invariant from the regression test
(``max(atom_to_token_ids_global_local) < N_tokens_global``) must be
violated on at least one rank for at least one parametrization. This module
asserts that violation directly per-rank by collecting the local max id
across ranks and checking that some rank exceeds the bound.

References
----------
* commit ``ac8d1c0c``: ``fix(1d-cp): no atom_to_token offset under 1D-CP topology``
* ``.claude/todo.md`` lines 64-72 (procedural mutation test follow-up)
* regression test:
  :mod:`tests.distributed.data.feature.test_dtensor_pack_atom_features_1d`
"""

from __future__ import annotations

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import boltz.distributed.data.feature.featurizer as featurizer_module
from boltz.distributed.data.feature.featurizer import pack_atom_features
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.shardwise_op import shardwise_offset as _real_shardwise_offset
from boltz.testing.utils import spawn_multiprocessing

# Atom feature keys that pack_atom_features consumes. Kept in sync with
# the regression test's _ATOM_FEATURE_KEYS_PACKED.
_ATOM_FEATURE_KEYS_PACKED = {
    "atom_pad_mask",
    "atom_to_token",
    "ref_pos",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "ref_space_uid",
}

# Mutation identifiers. Kept short so they read well in pytest -v output.
_MUTATION_UNCONDITIONAL = "unconditional_n_tokens_per_shard"
_MUTATION_DIVIDE_BY_CP_SIZE = "divide_by_cp_size"


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
    """Build global feature tensors for the 1D-CP pack_atom_features mutation test.

    Mirrors :func:`tests.distributed.data.feature.test_dtensor_pack_atom_features_1d.
    _build_1d_cp_feats` so that the mutation test exercises the same code paths
    with the same shapes / dtypes the regression test pins. Inlined (not imported)
    because the regression test module is not a package and importing across test
    files is fragile under pytest rootdir conventions.
    """
    if N_atoms_global % cp_size != 0:
        raise ValueError(f"N_atoms_global={N_atoms_global} must be divisible by cp_size={cp_size}")
    if N_atoms_global < N_tokens_global:
        raise ValueError(
            f"N_atoms_global={N_atoms_global} must be >= N_tokens_global={N_tokens_global} "
            "so every token receives at least one atom."
        )

    base = N_atoms_global // N_tokens_global
    rem = N_atoms_global - base * N_tokens_global
    atom_counts_per_token = torch.full((N_tokens_global,), base, dtype=torch.long, device=device)
    if rem > 0:
        atom_counts_per_token[:rem] += 1
    assert int(atom_counts_per_token.sum().item()) == N_atoms_global

    token_id_per_atom = torch.repeat_interleave(
        torch.arange(N_tokens_global, dtype=torch.long, device=device),
        atom_counts_per_token,
    )
    assert token_id_per_atom.shape == (N_atoms_global,)

    atom_to_token_per_sample = torch.nn.functional.one_hot(token_id_per_atom, num_classes=N_tokens_global)
    atom_to_token = atom_to_token_per_sample.unsqueeze(0).expand(B_global, -1, -1).contiguous().to(dtype=dtype)
    atom_pad_mask = torch.ones((B_global, N_atoms_global), dtype=torch.bool, device=device)

    ref_pos = torch.randn((B_global, N_atoms_global, 3), dtype=dtype, device=device, generator=generator)
    ref_charge = torch.randn((B_global, N_atoms_global), dtype=dtype, device=device, generator=generator)
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


def _make_mutated_shardwise_offset(mutation_id: str, n_tokens_global: int, cp_size: int):
    """Return a wrapper around ``shardwise_offset`` that injects a buggy offset.

    The wrapper is installed via monkeypatch on
    :func:`boltz.distributed.data.feature.featurizer.shardwise_offset` (the
    name imported into ``featurizer.py``). The wrapper IGNORES the
    ``offset_per_rank`` argument supplied by ``pack_atom_features`` and
    substitutes a value chosen to emulate one of the two known mutations:

    * ``unconditional_n_tokens_per_shard`` — the pre-``ac8d1c0c`` bug. Under
      1D-CP the local one-hot's token dim equals ``N_tokens_global``, so
      ``n_tokens_per_shard == N_tokens_global`` and the bad offset is
      ``N_tokens_global``.
    * ``divide_by_cp_size`` — a wrong refactor that floor-divides the local
      token dim by the cp-axis size: ``N_tokens_global // cp_size``.

    Parameters
    ----------
    mutation_id : str
        One of ``_MUTATION_UNCONDITIONAL`` or ``_MUTATION_DIVIDE_BY_CP_SIZE``.
    n_tokens_global : int
        The global token count (== local one-hot dim 2 under 1D-CP). Captured
        via closure so the wrapper does not have to introspect the input.
    cp_size : int
        Size of the cp mesh axis (1D-CP). Captured via closure for the
        ``divide_by_cp_size`` mutation.
    """
    if mutation_id == _MUTATION_UNCONDITIONAL:
        bad_offset = n_tokens_global
    elif mutation_id == _MUTATION_DIVIDE_BY_CP_SIZE:
        bad_offset = n_tokens_global // cp_size
    else:
        raise ValueError(f"Unknown mutation_id={mutation_id!r}")

    def _wrapper(x: DTensor, dim: int, offset_per_rank):
        # offset_per_rank from pack_atom_features is discarded; mutation injects
        # a deterministic wrong value derived from closure.
        return _real_shardwise_offset(x, dim, bad_offset)

    return _wrapper


def _parallel_assert_mutation_breaks_in_range(
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
    mutation_id: str,
):
    """Per-rank worker that monkeypatches the mutation and asserts the regression test would catch it.

    Steps:

    1. Initialise the distributed manager exactly as the regression worker
       does (same mesh, same env, same backend).
    2. Build the same DTensor inputs (1D-CP, ``(Shard(0), Shard(1))`` for
       ``atom_to_token``, token dim replicated).
    3. Monkeypatch ``featurizer.shardwise_offset`` to inject the buggy
       ``offset_per_rank`` for the given mutation.
    4. Call ``pack_atom_features``. Under the mutation, this must produce
       at least one rank with ``max(atom_to_token_ids_global.to_local()) >=
       N_tokens_global`` (the regression test's in-range invariant).
    5. All-reduce a boolean "this rank violated the invariant" flag across
       cp + dp so every rank sees the same global witness, then assert at
       least one rank's local max is OOB.
    """
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
    assert device_mesh.ndim == 2, f"Expected 2D (dp, cp) mesh, got ndim={device_mesh.ndim}"
    assert device_mesh.mesh_dim_names == (
        "dp",
        "cp",
    ), f"Expected mesh dim names ('dp', 'cp'), got {device_mesh.mesh_dim_names}"
    cp_size = device_mesh.size(1)

    placements_single = (Shard(0), Shard(1))
    placements_pair = (Shard(0), Shard(1))

    feats_dt: dict[str, DTensor] = {}
    for key in sorted(feats_global_host.keys()):
        v = feats_global_host[key].to(device=device)
        if v.dtype.is_floating_point and v.dtype != dtype:
            v = v.to(dtype=dtype)
        placement = placements_pair if key == "atom_to_token" else placements_single
        feats_dt[key] = distribute_tensor(v, device_mesh, placement)

    # Sanity check that the 1D-CP layout invariant holds (mutation only makes
    # sense under this layout; if the layout broke we'd be testing the wrong thing).
    atom_to_token_dt = feats_dt["atom_to_token"]
    assert atom_to_token_dt.to_local().shape[2] == N_tokens_global, (
        f"Rank {rank}: 1D-CP layout invariant violated — atom_to_token local token-dim "
        f"size {atom_to_token_dt.to_local().shape[2]} != N_tokens_global={N_tokens_global}. "
        "Mutation test cannot run."
    )

    # Install the mutation.
    mutated = _make_mutated_shardwise_offset(mutation_id, N_tokens_global, cp_size)
    monkeypatch.setattr(featurizer_module, "shardwise_offset", mutated)

    feats_packed = pack_atom_features(feats_dt, _ATOM_FEATURE_KEYS_PACKED, W)
    a2t_ids_global = feats_packed["atom_to_token_ids_global"]
    atom_mask_local = feats_packed["atom_pad_mask"].to_local().bool()
    a2t_ids_local = a2t_ids_global.to_local()

    # Per-rank max over valid (non-pad) positions. -1 sentinel when this rank
    # holds no real atoms (cannot reason about OOB).
    if atom_mask_local.any():
        local_max_id = int(a2t_ids_local[atom_mask_local].max().item())
    else:
        local_max_id = -1
    local_violates = local_max_id >= N_tokens_global

    # All-reduce the violation flag over the *flat* cp+dp world so every rank
    # raises consistently. Use a small int tensor over the world group.
    violates_t = torch.tensor([1 if local_violates else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(violates_t, op=torch.distributed.ReduceOp.SUM)
    any_rank_violates = int(violates_t.item()) > 0

    assert any_rank_violates, (
        f"Rank {rank}: mutation '{mutation_id}' did NOT trigger the in-range "
        f"invariant violation that the regression test pins. "
        f"local_max_id={local_max_id}, N_tokens_global={N_tokens_global}. "
        "Either the monkeypatch did not apply at the call site, or the mutation "
        "does not actually break the post-ac8d1c0c invariant — investigate."
    )

    # Restore and clean up.
    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "mutation_id",
    [_MUTATION_UNCONDITIONAL, _MUTATION_DIVIDE_BY_CP_SIZE],
    ids=["mut=unconditional_n_tokens_per_shard", "mut=divide_by_cp_size"],
)
@pytest.mark.parametrize(
    "N_tokens_global",
    [16, 17],
    ids=["N_tokens=16(aligned)", "N_tokens=17(prime)"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        # CUDA dp=1 cp=2: minimum config that actually exposes the mutation
        # (cp=1 would have rank * offset = 0 even under the bad offset, making
        # both mutations vacuously pass the in-range check).
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cuda-dp1-cp2"],
)
def test_pack_atom_features_1d_mutation(setup_env, N_tokens_global: int, mutation_id: str):
    """Assert that the regression-test in-range invariant is violated under each mutation.

    For each of the two mutations defined in the module docstring,
    monkeypatch ``pack_atom_features``'s internal ``shardwise_offset`` so
    the produced ``atom_to_token_ids_global`` reflects the bug, then assert
    that at least one rank's local ids exceed ``N_tokens_global``. This is
    the direct counterpart of the regression test's primary assertion:

    .. code-block:: python

        assert max_id < N_tokens_global, "...ac8d1c0c regression..."

    Together with the regression test, this gives bidirectional coverage:
    the regression test catches the unmutated fix's behaviour, and this
    mutation test catches the (hypothetical) reversion.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    cp_size = int(grid_group_sizes["cp"])
    dp_size = int(grid_group_sizes["dp"])

    B_global = dp_size
    n_atoms_per_token = 4
    raw_atoms = N_tokens_global * n_atoms_per_token
    if raw_atoms % cp_size != 0:
        raw_atoms += cp_size - (raw_atoms % cp_size)
    N_atoms_global = raw_atoms
    assert N_atoms_global % cp_size == 0
    W = 16

    gen = torch.Generator(device="cpu").manual_seed(20260515)
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
        _parallel_assert_mutation_breaks_in_range,
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
        mutation_id,
    )
