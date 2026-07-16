---
name: dtensor_modules
description: >
  Implement DTensor-based context-parallel model modules that mirror a serial
  reference. Maps each serial layer/module to its Boltz-CP counterpart with exact
  input/output shapes, placements, collectives, and backward memory budget; writes
  the CP implementation following the autograd.Function conventions (explicit
  collectives, no implicit DTensor ops, promote_types, explicit from_local
  shape/stride, attribute-name + registration-order mirroring); and verifies it
  against the serial forward+backward via the test skill. Use when porting a
  custom model's trunk / attention / triangle / OPM / diffusion / confidence /
  loss modules to CP.
argument-hint: "[module name] [2d]"
---

# dtensor_modules — port a serial module to context parallelism

The serial module is the mathematical specification. The CP version must produce
numerically equivalent forward+backward while holding only its O(N/cp0) (single) or
O(N²/(cp0·cp1)) (pair) shard. The serial→CP dictionary with all I/O placements,
collectives, and backward budgets is in [module_map.md](module_map.md) — **look up
the target there first.**

## Preconditions

- `docs/current_code_structure.md` §4 (module map) names the target's serial
  block and its CP-candidate status (reuse / adapt / new).
- `docs/cp_infra.md` gives the 2D mesh and `world_size`.
- The features the module consumes are (or will be) produced by
  `/fold-cp:shard_data_feats` at known placements.
- Tech guide section for the target (see [module_map.md](module_map.md)).

## Step 1 — Look up the contract

In [module_map.md](module_map.md) find the target's row: serial file, 2D CP
file, tech guide §, **input placements**, **output placements**, **collectives**,
and **backward memory budget**. If the target is in the substrate-primitive
catalog, prefer composing it from those utilities over writing a new
`autograd.Function`.

**A reference CP layer = an app-agnostic ALGORITHM + a reference-SPECIFIC placement
contract; reconcile the contract before reusing it.** The Boltz file's ring / skew /
gather *algorithm* is reusable, but the *axes it shards vs replicates* are Boltz's data
contract and may differ from your model's (e.g. Boltz shards MSA-depth `S` where your
model replicates it; a different window-edge convention). Blindly vendoring the core
imports fine but **fails parity**. So: check the reference core's placement contract
against YOUR data placements (from `shard_data_feats` / `current_code_structure.md` §3)
first; if they match, reuse it; if they differ, **reuse the algorithm but re-derive the
collective schedule for your contract**. Algorithm-reuse and contract-match are
orthogonal — a row marked "reuse" can still need a contract re-derivation.

## Step 2 — Extract the math spec AND the full input contract

Read the serial module (the user's, and the Boltz serial twin if it maps cleanly).
Record:

- the exact math (so the CP forward mirrors it line-for-line where possible);
- the **complete input contract** — including feature-dict fields the signature
  does not spell out. Modules routinely pull extra tensors out of a `feats` dict
  (masks, atom→token indices, bias tensors, ensemble indices). Each such field has
  a required placement; cross-check it against the data-feature inventory
  (`current_code_structure.md` §3) and the placement dictionary from
  `shard_data_feats`. An undocumented feature input at the wrong placement is the
  most common CP integration bug.

## Step 3 — Decide compose vs fuse

- **≤3 chained DTensor ops:** compose the substrate utilities (`elementwise_op`,
  `sharded_op`, `shardwise_op`, …) — each checks placements at its boundary;
  flexible and auditable. Canonical: `modules/encoders.py:_atom_encoder`.
- **≥4 chained ops:** fuse into one `autograd.Function` — `to_local()` once, plain
  PyTorch math, explicit collectives, one `from_local()`. Canonical:
  `loss/distogram.py`.

## Step 4 — Implement (autograd.Function conventions — these are hard rules)

Forward:
- Assert inputs are DTensors of the expected placements **and global shapes** — accept **only**
  DTensor on the distributed path (no "naked" plain tensors, which hide wrong placements/strides —
  Rule 6); assert all share one `device_mesh`; assert even sharding for every `Shard` (collectives
  need equal per-rank buffer shapes — Rule 8); reject unhandled `Partial`. Do **not** read
  `ctx.needs_input_grad` in forward (use `tensor.requires_grad`).
- Decorate `@torch.amp.custom_fwd(device_type="cuda")` / `custom_bwd`, then **audit the
  dtype path end-to-end** against the app's precision semantics (Rule 19): operands cast
  to the compute dtype before math, every communicated buffer pinned to one dtype across
  ranks (a per-rank/autocast-dependent dtype deadlocks `send`/`recv`), and no unintended
  up/down-cast (correctness-silent but a transient peak-memory hazard — see `mem_profile`).
- Insert collectives explicitly (the exact set from [module_map.md](module_map.md));
  make tensors contiguous before collectives.
- `torch.promote_types(dtype, torch.float32)`, never `.float()`.
- Build outputs with `DTensor.from_local(..., shape=<global>, stride=<global via
  update_exhaustive_strides>, placements=<...>)` — never let it infer (Rule 9).

Backward:
- Match upstream-grad dtype to the saved-tensor dtype before mixed-precision math;
  cast operands to compute dtype for CPU.
- Reduce parameter gradients in ≥fp32 (`_all_reduce_grad_gteqfp32`).
- **Saved-for-backward tensors must obey the forward per-rank budget** — a tensor
  that scales with full N or S is a critical bug (Rule 11).

Module wiring:
- Mirror the serial **attribute names and `__init__` registration order** exactly,
  so `named_parameters()`/`state_dict()` line up in parity tests.
- In the model forward, avoid native DTensor dispatch on a sharded axis
  (`zeros_like`, `reshape`, `repeat_interleave`, `squeeze`) — use `dtensor_zeros`,
  `to_local()/from_local()`, or the substrate utilities. Elementwise `.to(dtype)`
  is safe.
- Broadcast control-flow scalars (recycle/sample counts) across the flat CP group
  so all ranks branch identically (Rule 7).
- Document input/output placements, collectives, and backward budget in the module
  docstring (template: `loss/distogram.py`).

## Step 5 — Wire into the forward and reconcile neighbors

Insert the module into the model's distributed forward mirroring the serial order.
Confirm the **producer→consumer placement contract** with each neighbor: the
upstream module's output placement must equal this module's input placement
(Rule 4). If you are porting several coupled modules, this reconciliation is the
job `/fold-cp:dispatch_work` enforces between paired agents.

## Step 6 — Test against the serial reference

Hand to `/fold-cp:test` at the right level:

- **unit** — the single op/`autograd.Function` vs its serial twin;
- **layer integration** — the layer in isolation;
- **module integration** — the module with its real feature inputs drawn from the **shared
  synthetic DataModule** (one `Dataset`/`DataLoader`/`DataModule` that reuses the production
  distributor; tests subset the feature set), **not** ad-hoc per-feature `randint`, so the test
  also exercises the featurizer→module seam (Rule 4) — see `/fold-cp:test` Step 2.2;
- **workflow** — end-to-end forward (and backward/step) under CP.

Always serial fwd+bwd as ground truth, explicit random `grad_output`, fp64, and
the anti-vacuous checks. Pin each new operator with its own parity test before
moving downstream.

## Output contract

- The CP module implemented per the conventions above, mirroring serial attribute
  names/order, with a docstring documenting placements + collectives + backward
  budget.
- A passing parity test at the appropriate level (serial == CP fwd and bwd),
  logged.
- The module's row in `docs/current_code_structure.md` §4 updated to "done", and
  any newly-resolved feature-input placements written back to §3.
