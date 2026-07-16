---
name: shard_data_feats
description: >
  Implement distributed data-feature sharding for CP: assign DTensor placements to
  every model feature, build the placement-definition dictionary, implement
  atom-feature pack/pad/scatter, per-shard and cross-axis divisibility padding,
  the DTensor Dataset / DataLoader / DataModule and collate, and the parity tests
  that prove the sharded features reassemble to the serial features. Traces the
  user's featurization from the inference/training drivers and mirrors the Boltz-CP
  data pipeline. Use after learn_context + build_infra, when wiring a custom
  model's features into a context-parallel mesh.
argument-hint: "[2d] [feature group: atom | token | msa | pair | all]"
---

# shard_data_feats — produce DTensor features the CP model can consume

The model's distributed forward reads each feature at a specific placement; the
data pipeline must **produce** it at exactly that placement, or the CP boundary
mismatches (shape error at best, silent wrong-sharding at worst). This skill
builds that producer side. Placement tables, the file map, and the
padding/index rules are in [reference.md](reference.md); read it before
implementing.

## Preconditions

- `docs/current_code_structure.md` §3 (data-feature inventory) exists. If not, run
  `/fold-cp:learn_context` first.
- `docs/cp_infra.md` names the target 2D topology and `world_size`. If
  not, run `/fold-cp:build_infra`.
- `$BOLTZ_CP_REPO` is known; tech guide **§1 Data Feature Sharding** is the
  reference for this skill.

## Step 1 — Lock the placement of every feature

For each row of the feature inventory, assign a placement from the tables in
[reference.md](reference.md), driven by **axis semantics** (tokens / atoms / MSA
depth / ensemble / pair), not by guessing from shape. The CP sub-mesh placement
(without DP) is:

- **2D** `(cp0, cp1)`: token/atom/single `[N,…]` → `(Shard(0), Replicate())`;
  pair `[N,N,…]` → `(Shard(0), Shard(1))`; ensemble `[E,N,…]` → `(Shard(1),
  Replicate())`.

DP is prepended later by the collate step — do not bake it into per-feature
placements. Record the final placement back into the inventory table.

**Co-shard coupled axes — placement of the second axis is not free.** A feature with
two token-like axes must place those axes consistently with the algorithm that
consumes it, or distributed gather/scatter reads off-rank data: pair `[N, N, …]`
(token×token) → 2D `(Shard(0), Shard(1))` (the square tile); and an
atom→token `[N_atoms, N]` feature must have its atom axis and the token axis it indexes
**co-sharded** (same cp axis) or **co-replicated**, never sharded independently. See the
"Co-sharding multi-semantic-axis features" section in [reference.md](reference.md) and
`dtensor_modules/module_map.md`.

## Step 2 — Build the placement-definition dictionary

Create a dict mapping **the user's feature names** to their placement tuples,
mirroring `$BOLTZ_CP_REPO/src/boltz/distributed/data/module/placements.py`.
This dict is the single source of truth shared by the
data pipeline and the model forward — the same names, the same placements. Keep a
`FEATURE_AXIS_SEMANTICS` registry alongside it (per axis: `"tokens"`, `"atoms"`,
`"msa"`, `None`) so padding logic never has to guess when `N_tokens == N_atoms`.

## Step 3 — Atom-feature pack / pad / scatter

Atom features map to tokens and shard in parallel with N. Mirror
`featurizer.py:pad_and_scatter_atom_features_dtensor` and `pack_atom_features`:
pack per-token atom windows, pad the atom axis to a multiple of the mesh size,
and scatter to the owning rank with indices that stay consistent with the token
sharding. See [reference.md](reference.md) "Atom scatter & index remap".

## Step 4 — Divisibility padding (two kinds — both required)

Why pad at all: `torch.distributed` collectives/P2P need **identical per-rank buffer
shapes** (see the fold-cp hard rules), so the sharded axis must divide evenly across
the mesh, and every axis sharing its semantics must match the same padded length.

1. **Per-shard padding:** if `N % cp0 != 0`, pad the sharded axis to the next
   multiple of `cp0` before distribution, and mark the pad positions invalid in the
   propagated mask.
2. **Cross-axis padding:** the second pair axis, and the
   trailing token axis of atom→token index features, must be padded to the same
   padded `N`. Drive this from the `FEATURE_AXIS_SEMANTICS` registry, never from
   raw shape comparison.

## Step 5 — Distribute, broadcast metadata, collate

- Use `distribute_features` + `broadcast_feature_tensors_metadata` patterns from
  `data/utils.py`: rank 0 (or the serial fetcher) produces features, metadata is
  broadcast so all ranks build matching `DTensor.from_local` with explicit
  `shape`/`stride`/`placements` (Rule 9 — never let it infer).
- Add the DP axis with `CollateDTensor`, mapped onto the mesh with
  `map_subgroup_mesh_to_cpu`.
- **Sort keys** in every collective loop so all ranks issue collectives in
  identical order (Rule 7).

## Step 6 — DTensor Dataset / DataLoader / DataModule

Mirror the Boltz data modules for **both** workflows:

- Inference: `data/module/inferencev2.py`.
- Training: `data/module/trainingv2.py`.

Adapt the user's existing `Dataset`/`DataLoader`/`DataModule` (or Lightning
`LightningDataModule`) to emit DTensor batches at the locked placements. Preserve
the serial dataset as ground truth (Rule 2) — wrap or subclass, don't rewrite it.
Wire cross-rank fetch-error propagation (`_error_propagation.py`) so a failed
serial fetch on one rank raises on all ranks instead of deadlocking.

## Step 7 — Index / mask / padding consistency check

Before testing, audit (see [reference.md](reference.md) checklist): local-vs-global
index semantics agree between data and model; mask polarity is consistent; pad
positions are masked; atom-to-token indices remap correctly after repad
(`remap_atom_indices_repad`).

## Step 8 — Test the sharded features

Hand off to `/fold-cp:test` to build the data parity test (mirror
`tests/distributed/data/`): assemble the DTensor features with `full_tensor()` and
assert equality to the serial features; assert sharding is active **when either 2D mesh axis
has size > 1** (local shape < global on the sharded axis; `cp=(1,1)` ⇒ local == global);
assert masks/pads and cross-axis
padding are correct;
include a non-power-of-two per-axis square-mesh boundary case (for example, `cp=(3,3)`).

## Output contract

- A placement dictionary + axis-semantics registry covering every model feature — including
  **auxiliary features** (masks, biases, frame/relpos/atom indices) and the **local↔global index
  mapping** (which indices are local `0..N/cp0−1` vs global `0..N−1`), not just the primary
  token/pair/atom features.
- DTensor Dataset/DataLoader/DataModule for inference and training emitting batches
  at the locked placements, with serial code left intact.
- A passing data parity test (full_tensor == serial features; sharding active;
  padding/mask correct), logged under `/tmp/$USER/`.
- A short report: features sharded, placements used, and any feature that could
  not be mapped (escalate to the user).
