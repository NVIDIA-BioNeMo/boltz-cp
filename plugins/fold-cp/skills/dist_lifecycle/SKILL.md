---
name: dist_lifecycle
description: >
  Stand up the distributed-model lifecycle for a CP training/inference run: device
  placement before DTensor wrapping, the all-trainable-params-are-DTensors invariant
  (with placeholder/freeze for unimplemented modules), checkpoint save (DTensor →
  plain) and load (realign via the live state_dict template + redistribute optimizer
  state to parameter placements), resume RNG seed offset, and DTensor-safe EMA. Use
  after dtensor_modules + build_infra, when wiring the ported CP modules into a real
  trainer/predictor (Lightning or custom) and you need checkpoints, resume, or EMA to
  work — the gap between "modules pass parity" and "training runs and resumes".
argument-hint: "[wrap | checkpoint | resume | ema | optimizer]"
---

# dist_lifecycle — make a CP model train, checkpoint, and resume correctly

Porting modules (`dtensor_modules`) proves the math; this skill makes the *trainer*
work: device placement, parameter typing, checkpoint I/O, resume, and EMA under
DTensor. These steps are where a parity-passing model still fails to train or resume.
Reference files (all under `$BOLTZ_CP_REPO/src/boltz/distributed`):
`lightning_strategy.py`, `train.py`, `predict.py`, `model/optim/ema.py`, `manager.py`.

## Preconditions

- Modules ported + parity-tested (`dtensor_modules` + `test`).
- `docs/cp_infra.md` (topology, world_size, launch recipe) and
  `docs/current_code_structure.md` (framework: Lightning / DeepSpeed / custom).

## Step 1 — Device placement before DTensor wrapping

Move the serial module to its CUDA device **before** wrapping/distributing it. A CPU
module wrapped against a CUDA mesh is a device mismatch. Order: build serial → `.to(device)`
→ DTensor-wrap (`train.py:_create_distributed_model`). Fail fast if `torch.distributed`
is not initialized — no silent single-GPU fallback (Rule 12).

## Step 2 — Every trainable parameter must be a DTensor

Gradient redistribution (`on_after_backward` → `param.grad.redistribute(...)`) only fires
on DTensor parameters; a plain-tensor trainable parameter silently **bypasses** it and
diverges across ranks. So:

- Validate in `__init__`: every `requires_grad` parameter is a `DTensor` (else raise).
- For modules you have not ported yet, use a **placeholder**: keep the parameters
  (so `state_dict` stays key-compatible with the serial checkpoint), `requires_grad_(False)`,
  and raise `NotImplementedError` on `forward`. This keeps checkpoints portable and lets
  training proceed without the unimplemented submodule (`boltz2.py` `_PlaceholderModule`).
- Mirror serial **attribute names and registration order** so `state_dict` keys line up.

## Step 3 — Checkpoint SAVE: DTensor → plain tensors

Convert every DTensor in the payload to a plain (full) tensor before writing
(`lightning_strategy.py`). Save an identical plain checkpoint (portable across
topologies / loadable by the serial model). Do **not** use a callable `map_location`
with DTensor checkpoints.

## Step 4 — Checkpoint LOAD + optimizer-state redistribution

- Load the plain checkpoint, then **realign placements using the live model's
  `state_dict()` as the template** — `from_local`/`distribute_tensor` each loaded tensor
  to the placement the live parameter expects.
- **Redistribute optimizer buffers to match parameter placements before
  `optimizer.step()`** — plain-tensor optimizer state from the checkpoint mixed with
  DTensor parameters errors (or silently mis-updates) on step.
- Pretrained / transfer load: `strict=False` **plus an explicit architecture check**
  (e.g. assert v2 flags / module presence) — `strict=False` silently tolerates a
  confidence-module or V1/V2 mismatch that then crashes or mis-loads.
- Use **FQN (fully-qualified-name) optimizer-state keys**, and **auto-detect legacy
  integer-indexed keys** on load so an old checkpoint still maps — integer param-group indices
  break the moment module registration order changes.
- Prefer **`from_local`** (with explicit `shape`/`stride`/placements) over `distribute_tensor`
  on every same-process load/restore path, and keep `map_location` a **string** device — a
  callable `map_location` strips the DTensor type.
- After `load_from_checkpoint`, **re-inject any ctor args passed to
  `save_hyperparameters(ignore=[...])`** (validators, callbacks, non-serializable objects):
  they come back `None`, so restore them from the live module/config before use.

## Step 5 — Resume RNG

Offset the seed by **global rank**, and on resume additionally by **epoch +
global_step**, so ranks don't replay identical data/noise and a resume doesn't repeat
the pre-resume stream (`train.py`). This is the entropy side of the single-device RNG
rule; replicate-axis consistency is still by broadcast, and control-flow scalars
(recycling/sampling counts) are broadcast across the flat CP group. (See RULES.md
"single-device RNG entropy equivalence" + Rule 18.)

## Step 6 — EMA under DTensor

EMA updates parameters with **in-place `.data` arithmetic, which bypasses DTensor
dispatch** — so EMA only works correctly when the parameters are **Replicate** on every
mesh axis; `Shard`/`Partial` placements give silently wrong EMA updates. When swapping
EMA weights in/out, back up each tensor's `device_mesh` + `placements` and rewrap, rather
than round-tripping through CPU (which strips the DTensor type) — see `model/optim/ema.py`.
Guard the EMA update with `torch.inference_mode(False)` (an EMA step inside an inference-mode
context errors on the in-place write), keep a CPU backup of the shadow weights for
checkpointing, and on fine-tune resume **backfill any missing EMA keys** from the live
parameters so a newly-added module doesn't crash the EMA load.

## Step 7 — Verify with a stop-and-go test

Hand to `/fold-cp:test`: train N steps → checkpoint → resume → train M more, and assert
(a) the resumed run's weights/optimizer state match a non-interrupted run (within the
fp budget), (b) intermediate checkpoints differ from the final, (c) the saved checkpoint
contains **plain** tensors (no DTensors), (d) it loads into the serial model, and (e) when
you change the checkpoint/serialization layout, a checkpoint written by the **prior format**
still loads (backward-compat) — cover both save and load paths. Mirror
`tests/distributed/test_dtensor_stop_and_go.py` (monkeypatch the distributed cleanup to a
no-op for in-process port reuse).

## Gotchas that pass parity but break training

- **A detached / re-wrapped DTensor loss silently drops its gradient.** Building a loss via
  `from_local(local.detach(), …)`, adding a structurally-zero term, or letting the framework's
  grad-accumulation divide such a tensor creates a **fresh** DTensor from `_local_tensor` with
  `requires_grad=False` — backward then no-ops and weights never move, with **no error**. Keep
  the loss on the autograd graph (no `detach`/native re-dispatch on the differentiable path,
  Rule 6); if you must accumulate manually, pin grad-accumulation to 1 and verify `grad_norm > 0`.
- **Do not "harden" framework checkpoint loads to `weights_only=True`.** Lightning/torch
  checkpoints carry non-tensor objects by design; forcing `weights_only=True` breaks the load —
  it is accepted risk, not a vulnerability to patch.

## Output contract

- The CP model places on device before wrapping; all trainable params are DTensors
  (unimplemented ones placeholdered/frozen).
- Checkpoints save as plain tensors, load with placement realignment + optimizer-state
  redistribution, and pass a stop-and-go resume parity test.
- Resume seeding offsets by rank (+ epoch/global_step); EMA (if used) is Replicate-only.
- A short report: framework, checkpoint format, what is placeholdered, and the
  stop-and-go test result.
