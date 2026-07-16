# shard_data_feats — reference

All paths are relative to `$BOLTZ_CP_REPO`. Tech guide section: **§1 Data Feature
Sharding**.

## Placement tables (CP sub-mesh only; DP added by collate)

### 2D-CP — sub-mesh `(cp0, cp1)`, pair tensor square-tiled

| Feature class | Example shape | Placement | Per-rank local |
|---|---|---|---|
| Token / single / atom | `[N, C]`, `[N_atoms, C]` | `(Shard(0), Replicate())` | `[N/cp0, C]` |
| Pair | `[N, N, C]` | `(Shard(0), Shard(1))` | `[N/cp0, N/cp1, C]` |
| MSA | `[S, N, C]` | `(Shard(1), Replicate())` | `[S, N/cp0, C]` |
| Ensemble-aware | `[E, N, …]` | `(Shard(1), Replicate())` | E replicated, N on cp0 |
| Scalar / global | `[C]` | `(Replicate(), Replicate())` | full |

2D requires `size_cp = cp0*cp1` a perfect square (`cp0 == cp1`).

`Shard(d)` = `Shard`, `Replicate()` = `Replicate`. Indices are **mesh-axis
positional**: `placements[k]` applies to mesh axis `k`; the integer inside
`Shard(dim)` is the **tensor** dimension.

## Data-pipeline file map (serial ↔ CP)

| Concern | 2D-CP file |
|---|---|
| Placement dictionary | `data/module/placements.py` |
| Atom pack/pad/scatter | `data/feature/featurizer.py` (`pad_and_scatter_atom_features_dtensor`, `pack_atom_features`) |
| Distribute + metadata bcast + collate | `data/utils.py` (`distribute_features`, `broadcast_feature_tensors_metadata`, `CollateDTensor`, `map_subgroup_mesh_to_cpu`, `get_flattened_group`) |
| Inference DataModule | `data/module/inferencev2.py` |
| Training DataModule | `data/module/trainingv2.py` |
| Symmetry features | `data/feature/symmetry.py` |
| Cross-rank fetch-error propagation | `data/module/_error_propagation.py` |
| Shared types | `data/module/types.py` |

Tests to mirror: `tests/distributed/data/` —
`test_dtensor_pack_and_pad_atom_features.py`,
`test_dtensor_scatter_features.py` (`remap_atom_indices_repad`),
`module/test_dtensor_distribute_features_error_propagation.py`.

## Axis-semantics registry (resolve N_tokens == N_atoms ambiguity)

Label each feature axis explicitly so padding never guesses:

```python
FEATURE_AXIS_SEMANTICS = {
    "token_single":  (None, "tokens", None),          # [B, N, C]
    "pair":          (None, "tokens", "tokens", None),# [B, N, N, C]
    "msa":           (None, "msa", "tokens", None),    # [B, S, N, C]
    "atom_single":   (None, "atoms", None),            # [B, N_atoms, C]
    "atom_to_token": (None, "atoms", "tokens"),        # [B, N_atoms, N]
}
```

Every `"tokens"`/`"atoms"` axis is subject to the matching divisibility padding below.

## Co-sharding multi-semantic-axis features

A feature with two token-like axes cannot have those axes sharded independently. The
distributed gather/scatter algorithms (tech guide §9 **"Co-Sharding/Co-Replicating
Requirements"**) require the related axes to be **co-located on the mesh**, because
`torch.distributed`'s equal-per-rank-buffer constraint shows up at the feature level:
each rank's atom tile must correspond to its token tile, or the gather indexes off-rank
data.

- **Pair `[N, N, C]` (token×token):** `(Shard(0), Shard(1))` — both token axes on
  the `cp0×cp1` grid (the square tile).
- **atom→token `[N_atoms, N]` (atoms×tokens):** the atom axis and the token axis it
  indexes must be **co-sharded** (same cp axis) or **co-replicated**, never sharded on
  different axes — `distributed_gather` / `distributed_outer_gather` /
  `distributed_scatter_reduce` only produce aligned tiles when they are.

This is *why* the axis-semantics registry labels both axes: the placement of the second
axis is dictated by the first axis and the consuming algorithm, not chosen freely.

## Padding rules

Padding makes shards equal-size so `torch.distributed` collectives/P2P — which require
identical per-rank buffer shapes — don't hang or corrupt memory. Two kinds:

1. **Per-shard (intra-shard):** pad the sharded axis to the next multiple of the
   mesh size *before* distribution; propagate a mask marking the pad positions
   invalid.
2. **Cross-axis (inter-axis):** every additional axis with the **same** semantics
   as the sharded axis (the second pair `N`, the token axis of atom→token indices)
   must be padded to the same padded `N`. Driven by the registry, not raw shape.
3. After padding, **remap atom→token indices** (`remap_atom_indices_repad`) so
   they point at the repadded token positions.

## Index / mask / padding consistency checklist

- [ ] Local vs global indices: a feature the pipeline sends as a *local* index
      (0..N/cp0−1) is read by the model as local, not global — and vice versa.
- [ ] Mask polarity (`1=valid` vs `1=pad`) matches the model's expectation.
- [ ] Pad positions are masked everywhere they are consumed (loss, attention bias).
- [ ] `DTensor.from_local` is called with explicit `shape`, `stride`, `placements`
      (Rule 9); strides via `update_exhaustive_strides`.
- [ ] Collective loops iterate **sorted** keys (Rule 7).
- [ ] Reshape/squeeze that changes tensor rank updates every `Shard(dim)` index.
- [ ] One-hot / label validity in a **padded** layout is `sum ∈ {0, 1}` (a pad row is
      all-zero), **not** `sum == 1` — a `== 1` assert spuriously fails on padding.
- [ ] A `torch.Generator`'s device matches the op's `device=` argument — a CPU generator on a
      CUDA op raises or silently desyncs the stream.

## Atom↔token co-sharding & index semantics

The atom-feature pipeline carries subtleties beyond the placement tables:

- **Block-diagonal co-sharding enables local matmul.** atom↔token maps
  (`atom_to_token`, `token_to_rep_atom`) are sharded so shard *i* holds only the atoms
  whose token lives in shard *i* — a diagonal block. With atoms and tokens co-sharded
  this way, atom↔token coordinate mapping (`r_coords = map @ atom_coords`) is a **local**
  matmul, no cross-shard communication.
- **Keep `atom_to_token` UNPACKED; pack only its global indices.** Window-batching/packing
  moves atoms between shards and breaks the block-diagonal scheme, so the one-hot
  `atom_to_token` matrix is **not** packed. Convert it to **global token indices**
  (shard-local argmax + rank offset) *before* packing; packing the one-hot directly gives
  wrong shard boundaries. The 2D layout is block-diagonal, so the rank offset is
  `n_tokens_per_shard`.
- **Remap atom-index features after padding, before scatter.** `frames_idx` etc. store
  dense unpadded global atom indices; once each shard is padded to `max_atoms_per_shard`,
  remap via `bucketize + offset` *before* scattering, and in collation pre-scan
  `all_reduce(MAX)` of per-sample atom counts to fix the stride.
- **`atom_to_token`'s token axis lags the atom axis after padding** — padding the atom
  (`Shard`) dim doesn't pad the trailing token dim; pad it to the global token count (the
  cross-axis padding of Rule 8).
- **`.clone()` bool/int8 shards before scatter.** Float shards in a `scatter_list` alias
  safely, but bool/int8 duplicates can alias the same storage across `j` and corrupt.

## Cross-rank fetch-error propagation (avoid rank-0 deadlock)

When only rank 0 (or the source rank) does the serial dataset fetch and the others wait in
collectives, a fetch **exception on rank 0** hangs the rest forever. Broadcast a
success/failure object (type, repr, traceback) to all ranks **before** entering the
`distribute_features` collectives, and re-raise on every rank on failure — mirror
`data/module/_error_propagation.py`. Any rank-0-only work preceding a collective needs
this pattern.

This generalizes to **Rule 20 (every rank must reach every collective)**: a worker
`IndexError`, an **infinite cropper/retry loop**, or an op with **no compute kernel** (an int64
`matmul` / one-hot product has no BLAS path and stalls one rank past the NCCL timeout) all
surface as a **hang, not a traceback**. So bound every fallback/retry loop to a finite depth,
prefer `argmax + gather` over an int64 one-hot `matmul`, and broadcast the ok/error sentinel
above before any rank-asymmetric step. Treat data/featurizer correctness as a
distributed-liveness concern, not just a data-quality one.
