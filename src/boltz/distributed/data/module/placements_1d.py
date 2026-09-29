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

"""1D Context Parallelism placement registry.

Placements are 1-element tuples on the CP sub-mesh ``(cp,)``, following the
same convention as the 2D CP registry in ``placements.py`` where placements
target the ``(cp_axis_0, cp_axis_1)`` sub-mesh.  The DP dimension is added
later by ``CollateDTensor``.

Features are distributed **before batching** (per-sample), so tensor shapes
here do NOT include the batch dimension B.

Placement conventions (unbatched tensor layout -> CP sub-mesh placement):
    - Single repr ``[N, ...]``:       ``(Shard(0),)`` — N on cp
    - Pair repr   ``[N, N, ...]``:    ``(Shard(0),)`` — first N on cp (row-slab)
    - MSA         ``[S, N, ...]``:    ``(Shard(1),)`` — N (dim 1) on cp, S replicated
    - Atom        ``[N_atoms, ...]``:  ``(Shard(0),)`` — N_atoms on cp
    - Ensemble    ``[E, N, ...]``:     ``(Shard(1),)`` — N (dim 1) on cp, E replicated
    - Replicate:                       ``(Replicate(),)`` — fully replicated on cp
"""

from torch.distributed.tensor import Replicate, Shard

# ---------------------------------------------------------------------------
# Placement type constants for the 1D CP sub-mesh (cp,)
# ---------------------------------------------------------------------------

# Single representation: [N, ...] -> N sharded on cp
PLACEMENT_1D_SINGLE = (Shard(0),)

# Pair representation: [N, N, ...] -> first N (row) sharded on cp (row-slab)
PLACEMENT_1D_PAIR = (Shard(0),)

# MSA features: [S, N, ...] -> N (tensor dim 1) sharded on cp, S replicated
PLACEMENT_1D_MSA = (Shard(1),)

# Atom features: [N_atoms, ...] -> N_atoms sharded on cp
PLACEMENT_1D_ATOM = (Shard(0),)

# Ensemble-aware features: [E, N, ...] -> N (tensor dim 1) on cp, E replicated
PLACEMENT_1D_ENSEMBLE = (Shard(1),)

# Fully replicated on cp
PLACEMENT_1D_REPLICATE = (Replicate(),)

# ---------------------------------------------------------------------------
# Feature -> placement mapping
# ---------------------------------------------------------------------------

BASE_FEATURE_PLACEMENTS_1D: dict[str, tuple] = {
    # Atom features — [N_atoms, ...]
    "ref_pos": PLACEMENT_1D_ATOM,
    "ref_charge": PLACEMENT_1D_ATOM,
    "atom_resolved_mask": PLACEMENT_1D_ATOM,
    "ref_element": PLACEMENT_1D_ATOM,
    "ref_atom_name_chars": PLACEMENT_1D_ATOM,
    "ref_space_uid": PLACEMENT_1D_ATOM,
    "coords": PLACEMENT_1D_ENSEMBLE,  # [E, N_atoms, 3]
    "atom_counts_per_token": PLACEMENT_1D_ATOM,
    "frame_resolved_mask": PLACEMENT_1D_ENSEMBLE,  # [E, N_tokens]
    "frames_idx": PLACEMENT_1D_ENSEMBLE,  # [E, N_tokens, 3]
    "atom_pad_mask": PLACEMENT_1D_ATOM,
    "atom_to_token": PLACEMENT_1D_ATOM,  # [N_atoms_per_shard, N_tokens]
    "token_to_rep_atom": PLACEMENT_1D_SINGLE,  # [N_tokens_per_shard, N_atoms]
    "r_set_to_rep_atom": PLACEMENT_1D_SINGLE,  # [N_r_set_per_shard, N_atoms]
    "ref_chirality": PLACEMENT_1D_ATOM,
    "atom_backbone_feat": PLACEMENT_1D_ATOM,
    "bfactor": PLACEMENT_1D_ATOM,
    "plddt": PLACEMENT_1D_ATOM,
    # Token features — [N, ...]
    "token_index": PLACEMENT_1D_SINGLE,
    "residue_index": PLACEMENT_1D_SINGLE,
    "asym_id": PLACEMENT_1D_SINGLE,
    "entity_id": PLACEMENT_1D_SINGLE,
    "sym_id": PLACEMENT_1D_SINGLE,
    "mol_type": PLACEMENT_1D_SINGLE,
    "res_type": PLACEMENT_1D_SINGLE,
    "disto_center": PLACEMENT_1D_SINGLE,
    "disto_target": PLACEMENT_1D_PAIR,  # [N, N]
    "disto_coords_ensemble": PLACEMENT_1D_ENSEMBLE,  # [E, N, 3]
    "token_bonds": PLACEMENT_1D_PAIR,  # [N, N]
    "type_bonds": PLACEMENT_1D_PAIR,  # [N, N]
    "token_pad_mask": PLACEMENT_1D_SINGLE,
    "token_resolved_mask": PLACEMENT_1D_SINGLE,
    "token_disto_mask": PLACEMENT_1D_SINGLE,
    "token_pair_pad_mask": PLACEMENT_1D_PAIR,  # [N, N]
    "pair_mask": PLACEMENT_1D_PAIR,
    "contact_conditioning": PLACEMENT_1D_PAIR,  # [N, N]
    "contact_threshold": PLACEMENT_1D_PAIR,  # [N, N]
    "cyclic_period": PLACEMENT_1D_SINGLE,
    "method_feature": PLACEMENT_1D_REPLICATE,  # scalar-like, replicated on cp
    "modified": PLACEMENT_1D_SINGLE,
    # MSA features — [S, N, ...]
    "msa": PLACEMENT_1D_MSA,
    "msa_paired": PLACEMENT_1D_MSA,
    "deletion_value": PLACEMENT_1D_MSA,
    "has_deletion": PLACEMENT_1D_MSA,
    "msa_mask": PLACEMENT_1D_MSA,
    "deletion_mean": PLACEMENT_1D_SINGLE,
    "profile": PLACEMENT_1D_SINGLE,
}

TRAINING_FEATURE_PLACEMENTS_1D: dict[str, tuple] = {
    **BASE_FEATURE_PLACEMENTS_1D,
    "temp_feature": PLACEMENT_1D_REPLICATE,
    "ph_feature": PLACEMENT_1D_REPLICATE,
}

INFERENCE_FEATURE_PLACEMENTS_1D: dict[str, tuple] = {
    **BASE_FEATURE_PLACEMENTS_1D,
    "affinity_token_mask": PLACEMENT_1D_SINGLE,
}

# ---------------------------------------------------------------------------
# Per-feature, per-axis semantic labels.
#
# Each entry is a tuple matching the unbatched tensor rank, where each
# position labels what that axis represents.  Allowed labels:
#   - ``"tokens"`` : axis length equals ``N_tokens``
#   - ``"atoms"``  : axis length equals ``N_atoms``
#   - ``None``     : axis is neither N_tokens nor N_atoms (e.g. MSA depth S,
#                    ensemble E, channel C, coord 3, atom-name 4, etc.)
#
# This map is the authoritative source for cross-axis padding inside the
# CP data path: when CP divisibility forces padding of one axis, every
# other axis that shares the same global length must be padded too,
# otherwise the model forward sees a dim mismatch.  Driving the dispatch
# from these annotations instead of a runtime size-match heuristic avoids
# the degenerate ``N_tokens == N_atoms`` case where the wrong axis would
# silently win (heuristic ambiguity).
# ---------------------------------------------------------------------------

FEATURE_AXIS_SEMANTICS_1D: dict[str, tuple[str | None, ...]] = {
    # Atom features — [N_atoms, ...]
    "ref_pos": ("atoms", None),  # [N_atoms, 3]
    "ref_charge": ("atoms",),
    "atom_resolved_mask": ("atoms",),
    "ref_element": ("atoms", None),  # [N_atoms, num_elements_one_hot]
    "ref_atom_name_chars": ("atoms", None, None),  # [N_atoms, 4, 64]
    "ref_space_uid": ("atoms",),
    "coords": (None, "atoms", None),  # [E, N_atoms, 3]
    "atom_counts_per_token": ("tokens",),  # [N_tokens] — name is historical; axis is tokens
    "frame_resolved_mask": (None, "tokens"),  # [E, N_tokens]
    "frames_idx": (None, "tokens", None),  # [E, N_tokens, 3]
    "atom_pad_mask": ("atoms",),
    "atom_to_token": ("atoms", "tokens"),  # [N_atoms, N_tokens]
    "token_to_rep_atom": ("tokens", "atoms"),  # [N_tokens, N_atoms]
    "r_set_to_rep_atom": (None, "atoms"),  # [N_r_set, N_atoms] — N_r_set is neither N_tokens nor N_atoms
    "ref_chirality": ("atoms", None),  # [N_atoms, num_chirality_classes]
    "atom_backbone_feat": ("atoms", None),  # [N_atoms, num_backbone_feat]
    "bfactor": ("atoms",),
    "plddt": ("atoms",),
    # Token features — [N_tokens, ...]
    "token_index": ("tokens",),
    "residue_index": ("tokens",),
    "asym_id": ("tokens",),
    "entity_id": ("tokens",),
    "sym_id": ("tokens",),
    "mol_type": ("tokens",),
    "res_type": ("tokens", None),  # [N_tokens, num_res_type_classes]
    "disto_center": ("tokens", None),  # [N_tokens, 3]
    "disto_target": ("tokens", "tokens", None),  # [N_tokens, N_tokens, num_disto_bins]
    "disto_coords_ensemble": (None, "tokens", None),  # [E, N_tokens, 3]
    "token_bonds": ("tokens", "tokens"),
    "type_bonds": ("tokens", "tokens"),
    "token_pad_mask": ("tokens",),
    "token_resolved_mask": ("tokens",),
    "token_disto_mask": ("tokens",),
    "token_pair_pad_mask": ("tokens", "tokens"),
    "pair_mask": ("atoms", "atoms"),  # [N_atoms, N_atoms] inference-only pair mask over atoms
    "contact_conditioning": ("tokens", "tokens", None),  # [N_tokens, N_tokens, num_contact_feat]
    "contact_threshold": ("tokens", "tokens"),
    "cyclic_period": ("tokens",),
    "method_feature": (),  # scalar
    "modified": ("tokens",),
    # MSA features — [S, N_tokens, ...]
    "msa": (None, "tokens"),  # [S, N_tokens]
    "msa_paired": (None, "tokens"),
    "deletion_value": (None, "tokens"),
    "has_deletion": (None, "tokens"),
    "msa_mask": (None, "tokens"),
    "deletion_mean": ("tokens",),
    "profile": ("tokens", None),  # [N_tokens, profile_dim]
    # Training-only features
    "temp_feature": (),  # scalar
    "ph_feature": (),  # scalar
    # Inference-only features
    "affinity_token_mask": ("tokens",),
}


def _check_registry_coverage() -> None:
    """Fail fast at import if axis-semantics drift away from placements.

    Both training and inference placement registries must have a
    matching ``FEATURE_AXIS_SEMANTICS_1D`` entry; otherwise the
    cross-axis padding loop in
    :class:`boltz.distributed.data.module.trainingv2_1d._BaseDatasetCP1D`
    has no way to decide whether a non-sharded N-shaped axis is
    ``N_tokens`` or ``N_atoms`` for the new feature.
    """
    placement_keys = set(TRAINING_FEATURE_PLACEMENTS_1D) | set(INFERENCE_FEATURE_PLACEMENTS_1D)
    semantic_keys = set(FEATURE_AXIS_SEMANTICS_1D)
    missing = placement_keys - semantic_keys
    extra = semantic_keys - placement_keys
    if missing or extra:
        raise RuntimeError(
            "FEATURE_AXIS_SEMANTICS_1D and placement registries disagree. "
            f"In placements but missing axis-semantics: {sorted(missing)}. "
            f"In axis-semantics but not in any placement registry: {sorted(extra)}."
        )


_check_registry_coverage()
