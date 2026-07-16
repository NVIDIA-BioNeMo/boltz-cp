# dtensor_modules — serial → CP dictionary

Look up the target here before implementing. All paths are relative to
`$BOLTZ_CP_REPO/src/boltz`. Read the cited tech guide section for the algorithm
and the exact forward/backward schedule.

## How to read placements

- Sub-mesh only (DP omitted). **2D** = `(cp0, cp1)`.
- `S(d)` = `Shard(d)` on **tensor** dim `d`; `R` = `Replicate`.
- Single/token/atom `[N, C]`: `(S0, R)`. Per-rank O(N/cp0).
- Pair `[N, N, C]`: `(S0, S1)` square tile O(N²/(cp0·cp1)).
- MSA `[S, N, C]`: `(S1, R)`.
- **Backward per-rank memory must equal the forward budget** for the same tensor
  class (Rule 11). Saved-for-backward tensors that scale with full N or S are bugs.
- **Every collective needs an equal per-rank buffer shape** (fold-cp Rule 8) — hence
  even shards (`shape[dim] % mesh_size == 0`) and the co-sharding requirement below.

## Substrate primitive catalog (no serial twin — these *are* the CP layer)

Compose these for ≤3-op spans; fuse into one `autograd.Function` for ≥4. Files
under `distributed/model/layers/`.

| Utility (file) | What it does | Placement effect |
|---|---|---|
| `elementwise_op.py` | unary/binary elementwise (`ElementwiseOp.RELU`, `SIGMOID`, add, mul) | preserves placement |
| `sharded_op.py` | reductions over a **sharded** dim (`sharded_sum`) | reduces across shards (collective) |
| `shardwise_op.py` | reductions over a **non-sharded** dim (`shardwise_sum`) | local, placement-preserving |
| `outer_op.py` | outer products of two single reprs | `[N,C]×[N,C] → [N,N,C]` |
| `replicate_op.py` | replicate / broadcast along a mesh axis | changes `R`↔`S` deliberately |
| `gather.py` | `distributed_gather`: token single → atom single, interval comm | `[N,…]→[N_atoms,…]` |
| `outer_gather.py` | `distributed_outer_gather`: token pair → atom pair (2D) | `[N,N,…]→[N_atoms,N_atoms,…]` |
| `scatter.py` | `distributed_scatter_reduce`: atom → token reduce | `[N_atoms,…]→[N,…]` |
| `redistribute_transpose.py` (+ `_without_dtensor.py`) | `[S0,S1]↔[S1,S0]` transpose via explicit P2P/all-to-all | swaps sharded pair axes |
| `cat_and_chunk.py` | concat/chunk across the sharded axis | re-tiles |
| `repeat_interleave.py` | `repeat_interleave` that preserves sharding | preserves placement |
| `linear.py` | param-replicated `Linear`, fp32 grad reduce, `Partial(Avg)` over replicate axis | params `R`, acts preserve |
| `layernorm.py` | param-replicated LayerNorm | same |
| `swiglu.py`, `sigmoid_gate.py`, `dropout.py`, `where.py`, `clip.py`, `squeeze.py`, `embedding.py`, `flatten_and_unflatten.py` | gated/activation/util DTensor ops | placement-preserving (dropout per-rank) |
| `atom_to_token.py` | atom↔token index mapping helpers | — |
| `dtensor_metadata_tools.py` | `update_exhaustive_strides`, global shape/stride helpers for `from_local` | — |
| `distribute_module_tools.py`, `utils.py` | module distribution, `dtensor_zeros`, misc | — |

### Gather-dense-slice (the recipe for non-differentiable targets)

Non-differentiable targets/masks — binned distances, relpos features, frames, and any
`argmin`/`topk`/SVD **selection** — are best built by **gather-dense-slice**: gather the
cheap, low-rank inputs *full* on every rank → run the **unchanged serial math densely** →
slice this rank's block to the working placement. This (a) reuses the serial code verbatim
(no CP re-derivation of non-diff math), and (b) gives Rule-17 determinism for free — every
rank runs the identical selection, so there is no per-rank `argmin`/SVD/`topk` drift. Only
the *differentiable* tensor (logits, coords) needs a real collective; the target is built
locally. Canonical uses: relpos pair init, distogram/PDE binned targets, PAE frame
construction. Cost is `O(N)`/`O(N²)` *dense per rank* on the low-rank input — fine when the
input is cheap relative to the model tensors.

### Co-sharding / co-replicating requirement (multi-axis gather/scatter)

`distributed_gather`, `distributed_outer_gather`, and `distributed_scatter_reduce`
relate two semantic axes (atoms↔tokens, or the two token axes of a pair). Those axes
must be **co-located on the mesh** — co-sharded on the same cp axis, or co-replicated —
so each rank's atom tile and token tile correspond (tech guide §9 **"Co-Sharding/
Co-Replicating Requirements"**). If the atom axis is sharded on cp0 while the token
axis it indexes is sharded on cp1, the gather reads off-rank data. This is a hard
precondition the data pipeline must satisfy (`/fold-cp:shard_data_feats`), and it is
why `torch.distributed`'s equal-per-rank-buffer-shape constraint (Rule 8) surfaces at
the *feature* level, not just the tensor level. The `outer_gather` row below is marked
"(2D)" for this reason — both pair axes are co-sharded on the `cp0×cp1` grid.

## Distributed manager & communication (tech guide §2)

| Concern | File | Notes |
|---|---|---|
| Mesh + process groups | `distributed/manager.py` (`DistributedManager`, `_create_device_mesh_and_groups`, `create_grid_group`, `initialize`) | singleton; flat CP group at `group["cp"]` for control-flow broadcasts |
| Comm primitives | `distributed/comm.py` (`One2OneComm`, `TransposeComm`, `Ring2DComm`, `AttentionPairBiasComm`, `Ring2DCommTriAttn`, `ternary_parity`) | double-buffered overlap; ring/transpose schedules |

## Layer dictionary (tech guide §3–§9)

| Module | Serial role | 2D file | § | In → Out placement | Collectives | Bwd budget |
|---|---|---|---|---|---|---|
| Triangle Mult (out/in) | pair→pair, contract over k | `layers/triangular_mult.py` | §4 | pair `(S0,S1)`→`(S0,S1)` | Cannon skew + `[S0,S1]↔[S1,S0]` | pair |
| Triangle Attn (start/end) | attn over row/col of pair | `layers/triangular_attention.py` | §3 | pair + bias → pair | start: all-gather bias; end: transpose `[I/cp,J]↔[J,I/cp]`; tri-rotation | pair |
| Outer Product Mean | single/MSA → pair | `layers/outer_product_mean.py` (`outer_op.py`) | §6 | MSA `(S1,R)` → pair | skew + reduce-scatter | pair |
| Pair Weighted Averaging (PWA) | pair-weighted single update | `layers/pair_averaging.py` | §5 | single+pair → single `(S0,R)` | transpose + row/column Cannon ring | pair (weights) |
| Attention Pair Bias (Ring) | attn w/ pair bias | `layers/attention.py` (`attention_impl.py`) | §7 | single q/k/v + pair bias → single | ring P2P of k/v+bias (`AttentionPairBiasComm`) | single + bias |
| Attention Pair Bias (Shardwise) | attn w/ pair bias, shardwise bias | `layers/attention.py` (shardwise path) | §8 | single + shardwise bias → single | shardwise gather/reduce | single |
| Transition | SwiGLU MLP | `layers/transition.py` | (within trunk) | single/pair → same | none (param-replicated `linear`) | matches input class |
| Pairformer block | full trunk block | `layers/pairformer.py` | §3–§8 | single+pair → single+pair | composition of the above | pair |
| Window batching / sliding-window gather | atom transformer windows | `layers/gather.py`, `outer_gather.py`, `scatter.py` (`GatherSlidingWindows`) | §9 | token→atom gather, atom→token scatter-reduce | interval-based P2P gather/scatter | O(window·N/cp0) |

## Module & model dictionary (tech guide §9–§11)

| Module | Serial role | 2D file | § | Notes |
|---|---|---|---|---|
| Atom Encoder | token↔atom + window attn | `modules/encoders.py` (`_atom_encoder`) | §9 | canonical **composed** path (~50 utilities); pulls atom→token indices + masks from feats |
| Token/Atom transformer | APB transformer | `modules/transformers.py` | §7–§8 | APB ring/shardwise |
| Trunk (recycling) | recycle single+pair | `modules/trunkv2.py` | §3–§8 | broadcast recycle count across CP group (Rule 7) |
| Diffusion | structure head | `modules/diffusion.py` | — | atom-level; sampling-step count broadcast across CP group |
| Diffusion conditioning | trunk→diffusion conditioning | `modules/diffusion_conditioning.py` | — | — |
| Confidence | pLDDT/PAE head | `modules/confidencev2.py` (`confidence_utils.py`) | §10 | own Pairformer stack; embeds predicted coords as distogram |
| Top-level model | wires all of the above | `models/boltz2.py` | §2 | comm wiring via `DistributedManager` groups |

## Loss dictionary (tech guide §10–§11)

| Loss | 2D file | § | Notes |
|---|---|---|---|
| Distogram | `loss/distogram.py` | §10 | canonical **fused** `autograd.Function` template (docstring documents budget) |
| Confidence / pLDDT / PDE | `loss/confidencev2.py` | §10 | fused Triton `cdist_lddt`, `cdist_pde` |
| Smooth LDDT | (composable) | §11 | — |
| Diffusion | `loss/diffusion.py` | — | — |
| B-factor | `loss/bfactor.py` | — | loss-level mean all-reduce over cp1 (replicate-axis drift mitigation) |
| Validation (LDDT etc.) | `loss/validation.py`, `model/validation/` | — | metrics |

## Implicit feature-container inputs (read this)

Modules pull tensors out of a `feats` dict that the signature does not name —
masks, `atom_to_token` indices, pair biases, ensemble/recycle indices. Each has a
**required placement** that must match what `shard_data_feats` produced. Before
implementing a module:

1. Grep the serial module for `feats[` / `**kwargs` access to enumerate every
   consumed field.
2. For each, find its placement in `current_code_structure.md` §3 + the placement
   dictionary; if absent, it is an unmapped feature — fix that in
   `shard_data_feats` first.
3. Treat a feature consumed at the wrong placement as the prime suspect when a
   parity test fails before blaming the math.

## autograd.Function correctness gotchas (hard-won, from the substrate)

Beyond the conventions above, these recur across the CP layers — each has caused a
hang or a silent wrong result:

- **Never read `ctx.needs_input_grad` in `forward`** — it can hang NCCL. Use
  `tensor.requires_grad` in `forward`; `ctx.needs_input_grad` is fine in `backward`.
- **`.contiguous()` before every collective** (`all_reduce`, `reduce_scatter`,
  `batch_isend_irecv`) — a non-contiguous buffer can corrupt or hang it.
- **Backward mirrors the forward comm in reverse** — the backward P2P plan is the
  forward's with send↔recv swapped; a rank's self-send is a direct `copy_`, not a
  P2POp (a self-P2P deadlocks).
- **Backward grad-reduction mode follows the forward's gather: slice vs reduce-scatter.**
  If the forward **all-gathered** an input, the backward **reduce-scatters** its grad;
  if each rank already holds a **complete replicated** copy of the quantity, the backward
  **slices** its local chunk — do NOT reduce-scatter (that double-counts). Choosing the
  wrong one silently doubles or halves the grad and still passes a single-rank check;
  it surfaces only in a multi-rank per-param/grad parity test.
- **dtype discipline** (Rule 10): cast masks/biases to the operand dtype *before*
  arithmetic (fp32×bf16 silently upcasts and poisons forward+saved+backward); save
  for-backward tensors in the compute dtype or re-promote in `backward` (`custom_bwd`
  doesn't restore autocast on CPU); reduce cross-rank grad/dbias sums in ≥fp32 then cast.
- **Pin the communicated-buffer dtype across ranks** (Rule 19): if a buffer's dtype
  depends on a per-rank branch / autocast, two ranks `send`/`recv` mismatched dtypes and
  **deadlock** — `.to(dtype)` the buffer to one agreed dtype before the collective. And an
  *unintended* up/down-cast is correctness-silent (parity passes after the round-trip) but
  a **transient peak-memory** blowup parity tests miss — keep intermediates in the intended
  dtype and verify peak with `/fold-cp:mem_profile`.
- **Re-gather, don't save the gathered form** (Rule 11): if forward all-gathers a
  bias/row, save only the O(N/cp0) shard and re-gather transiently in `backward` —
  saving the O(N²) gathered tensor blows the per-rank budget.
- **Scatter/gather with `reduce="mean"`**: divide the backward gradient by `count`
  (clamped `min=1`) and clamp masked-out indices into range before `scatter_add_`.
- **Ring init ordering**: when a transpose precedes the ring loop, `wait()` it before
  launching the row/column init sends, or the init recv can fire early and deadlock.
- **Pick the collective by the consumer's axis order.** When two collectives are
  mathematically equivalent (e.g. an `all_gather` vs an all-to-all + axis flip), choose the one
  whose **output axis order already matches the downstream consumer** — the other forces an
  extra transpose. Equal values are not equal layout.
- **A fused Triton loss wraps fwd+bwd in one `autograd.Function`** with
  `ctx.set_materialize_grads(False)` and explicit `None`-grad handling for non-differentiable
  outputs, so an upstream `None` grad neither allocates a zero tensor nor crashes. Gate it on a
  hardware/dtype capability predicate, and assert zero PTX register spilling (see
  `/fold-cp:test` Triton trio).
