---
name: test
description: >
  Write and run multi-rank pytest parity tests that prove a CP implementation is
  numerically equivalent to its serial reference, using mp.spawn / spawn_multiprocessing
  as in the Boltz-CP test framework. Covers unit, layer-integration,
  module-integration, and workflow-integration levels; handles missing test data by
  asking the user or synthesizing random features in the correct format; and enforces
  the anti-vacuous rules (serial ground truth, explicit random grad_output, fp64
  default tolerances, sharding-active and replicated-identical assertions). Also covers a
  second, non-parity class — property / invariant tests (e.g. test_rng_entropy) that assert
  a structural invariant such as RNG entropy when there is no serial value oracle. Use to
  verify any output of shard_data_feats or dtensor_modules.
argument-hint: "[path to source file under test] [unit|layer|module|workflow] [2d]"
---

# test — prove the CP path is correct

Most CP correctness is **parity**: given identical inputs and weights, the distributed
forward+backward must numerically match the serial reference, which is the **only** value
oracle (Rule 13) — Steps 1–7 below. But some correctness properties have **no serial value
oracle** and need a different, **property / invariant** test (see
[§Property tests](#a-second-test-class-property--invariant-tests-not-serial-parity)) — most
notably anything random (RNG / noise), where no distributed RNG reproduces an all-gathered
single-device draw, so only the *entropy structure* is checkable. Either way: a test that
can pass with a broken implementation is worse than no test — follow the anti-vacuous rules
without exception. The worker skeleton, comparison matrix, and helper inventory are in
[reference.md](reference.md).

## Preconditions

- `docs/cp_infra.md` gives the launch recipe + `world_size` (run `build_infra` if
  not).
- The source under test exists and has a serial twin to compare against.

## Step 1 — Choose the level and the test path

| Level | What it pins | Compare |
|---|---|---|
| unit | one op / `autograd.Function` | op output + grad vs serial op |
| layer | one layer in isolation | layer fwd+bwd vs serial layer |
| module | module with real feature inputs (from the shared synthetic DataModule — Step 2.2) | module fwd+bwd vs serial module |
| workflow | end-to-end inference or train step | outputs / loss / weight deltas vs serial |

**Test path mirrors the source path**:
`tests/distributed/<path>/test_dtensor_<file>.py` for DTensor tests,
`test_<file>.py` otherwise. Write the parity test for a new operator **before**
anything downstream depends on it.

## Step 2 — Get test data (real or synthesized)

1. Prefer **real** inference/training features if the user can provide them — ask
   for a path and the feature format.
2. Otherwise **synthesize through ONE shared synthetic `Dataset` + `DataLoader` + `DataModule`** —
   **not** per-test data-gen, and **not** loose one-off helper functions. Build a single synthetic
   data stack in the project's `distributed/testing/` subpackage (mirror `src/boltz/testing/utils.py`
   and `src/boltz/distributed/testing/`):
   - a **`Dataset`** whose `__getitem__` yields a synthesized `input_feature_dict` matching §3's
     contract (names, shapes, dtypes, axis semantics, masks), with a **`feature_subset`** knob;
   - a **`DataLoader`** (bs=1 + the production collate);
   - a **`DataModule`** that **REUSES the production CP distributor** (`distribute_input_feature_dict`
     / the data module's distributor) — never a re-implementation — to emit, for the requested
     subset, both the **serial** (full) and **CP** (DTensor) views. Every feature is emitted at the
     **data-feature sharding requirements** (the §3 placements / the §4.5 data-feature→consumer
     contract), which the reused distributor satisfies **by construction** — a synthetic
     dataset/dataloader/featurizer must never hand-roll placements that violate that contract.

   **Every test — data, module, and workflow — draws its inputs from this one DataModule and SUBSETS
   the feature set its target consumes.** The common synth + distribution + divisibility-padding +
   atom-index-remap code then lives in exactly one place; tests never duplicate it. Two payoffs:
   - Because inputs flow through the real synthesizer + production distributor at the locked
     placements, a **feature-consuming** module test (relative-position encoder, atom encoder, input
     embedder, MSA module, diffusion conditioning, …) automatically exercises the **featurizer→module
     seam (Rule 4)** and the featurizer's **value semantics** (valid ranges, coupled-feature
     consistency, monotone indices) — drift is caught at the *unit* level, not deferred to the
     workflow test. Do **not** hand-roll data features with ad-hoc `torch.randint`/`randn`.
   - Keep ad-hoc `randn` **only** for genuine intermediate **activations** that are not featurizer
     outputs (an upstream single/pair representation, a diffusion noise level).
3. Use the **same** inputs and weights on the serial and CP sides — load one `state_dict` into both,
   and draw features from the DataModule with one deterministic seed so both sides see identical
   inputs. Convert the module to the test dtype (fp64) **before** `load_state_dict` to avoid an fp32
   round-trip (fold-cp test discipline).

## Step 3 — Pre-check on a 1×1 mesh FIRST, then build the multi-rank worker

**Before any multi-GPU run, verify the fwd+bwd decomposition on a 1×1 mesh** (single
rank / single process, where every collective degenerates to a no-op), in fp64, CP ==
serial. This isolates math / placement-seam / op-order bugs from collective bugs, needs
no GPUs or multi-rank launch, and runs in seconds; a 4-rank 2×2 gloo run is a cheap second
step. Modules that pass this 1×1 pre-check hit multi-GPU parity on the first try —
treat it as routine, not optional. Only then build the multi-rank worker:

- Spawn with `spawn_multiprocessing(worker, world_size, *args)` (Boltz testing
  utils) or `mp.spawn`.
- Init the process group with a **per-collective timeout** (e.g. 60s) so deadlocks
  fail fast; build the `DeviceMesh` for the selected 2D mesh.
- Monkeypatch the distributed cleanup to a no-op when a test reinitializes groups
  in one process (port reuse) — see `tests/distributed/test_dtensor_stop_and_go.py`.
- Wrap the worker body in **`try/finally`** to `destroy_process_group()` and free the GPU on exit;
  **do not `try/except` to swallow a rank's error and continue** — the throwing rank skips its
  remaining collectives and deadlocks the others (broadcast an ok/error sentinel instead — Rule 20).
- `seed_by_rank(rank)` for per-rank inputs that must differ; use a contained
  `torch.Generator` rather than global `manual_seed`.

## Step 4 — Compare against serial (the right reassembly)

| Comparison | Use |
|---|---|
| DTensor vs serial reference | `full_tensor()` to reassemble, then `assert_close` |
| DTensor vs DTensor (same placements) | `to_local()` on both — no communication |
| Gradients (replicated params) | `grad.full_tensor()` — `Partial(Sum)` needs the reduce; `to_local()` is only this rank's contribution |

Run serial fwd+bwd on full tensors; run CP fwd+bwd on sharded DTensors; compare.

## Step 5 — Anti-vacuous assertions (all required)

- **Explicit random `grad_output`** for backward — never `.sum().backward()`
  (uniform grads mask sign/permutation bugs).
- **Sharding is active when either 2D mesh axis has size > 1:**
  `local.shape[sharded_dim] < global.shape[sharded_dim]` — skip for `cp=(1,1)`
  (a valid debugging mesh where local == global).
- **Replicated values identical across ranks** (`assert_all_identical`).
- **Gradients non-zero** (and finite).
- **Perturb/randomize zero-initialized params first** — AF/residual "final" projections
  are zero-init, so output and all grads are vacuously zero (`0 == 0` passes).
- **Per-PARAMETER grad parity, not only input-grad** — for any module with parameters
  replicated over a sharded axis, assert `param.grad.full_tensor()` vs serial. A replicated
  param whose grad is a per-rank partial (missing all-reduce) passes forward + input-grad
  parity but is wrong; only the param-grad check catches it.
- **fp64 + `assert_close` default tolerances.** If it only passes after loosening
  `atol`, that is a bug signal — derive the tolerance from the error budget, do not
  tune it (Rule 15).
- At least one **adversarial/boundary** case (a non-power-of-two per-axis 2D mesh such as
  `cp=(3,3)`, a padded axis, a single-element shard).

## Step 6 — Parametrize and prefer CUDA

One test function parametrized over a tuple of mesh configs (do not copy-paste per
size). Prefer CUDA parametrizations; keep at most one CPU param (e.g. a 3×3 mesh) for
CI path coverage.

## Step 7 — Run, with timeout and logs

```bash
timeout 120 python -m pytest <test path> -x -q |& tee /tmp/$USER/<name>.log
```

**Make the log self-verifying (Rule 22).** Print the evidence on **success**, not only on
failure: the measured max abs/rel difference **and** the tolerance, the mesh / `world_size`, and a
one-line confirmation of each non-vacuous check (sharding-active, replicated-identical, grads
non-zero, per-param-grad). An orchestrator or reviewer then verifies the gate by **reading this
log** instead of re-running the GPU test (Rule 22) — a silent `assert_close` that prints nothing on
green forces a needless re-run. Emit the numbers.

Inspect the log even on green (catch silent skips / warnings). When a test fails,
assume the implementation is wrong (Rule 14) — diagnose the CP code; do not weaken
the test.

## A second test class: property / invariant tests (not serial parity)

Some correctness properties are **not** numerical equality to the serial reference — the
serial run is not a value oracle for them. The oracle is a **structural invariant** the
distributed run must satisfy. Write a *property test* when:

- the quantity is **random / not reproducible across mesh configurations** — RNG and noise: no
  distributed RNG numerically reproduces an all-gathered single-device draw today, so you
  can assert only its *entropy structure*, not its values; or
- the property is about **layout / determinism / consistency** rather than numbers — e.g.
  replicated values agree across ranks, control-flow counts match, sharding is active.

**Canonical example — `test_rng_entropy`** (`tests/distributed/test_dtensor_predict.py`):
asserts every per-rank RNG draw obeys **single-device RNG *entropy* equivalence** (the
fold-cp RNG rule) — sharded-axis draws carry **independent** entropy (the cp0 diffusion noise
must not tile), replicate-axis draws are **identical** (via broadcast) — **without** comparing
to any serial value. Mechanism (skeleton in [reference.md](reference.md)):

1. run the real workflow (e.g. DP=1, CP=(2,2), smallest sample) under `spawn_multiprocessing`;
2. monkeypatch the python `random` / numpy / torch RNG entry points to fingerprint every
   drawn sample per rank, and save the per-rank fingerprints to the test context (one JSON
   per rank);
3. **after the join**, in the parent process, assert the invariant across ranks: sizable
   sharded draws are **pairwise disjoint** across ranks (no tiling); replicated draws agree.

**Anti-vacuous rules for property tests** (these replace parity's grad_output/fp64/tolerance):

- **Negative control is mandatory.** Prove the test catches the bug: run it once against the
  known-bad implementation (e.g. `git stash` the fix) and confirm it **FAILS**, then restore.
  For `test_rng_entropy`, the pre-fix shared seed fails on the tiled coordinate noise.
- **Exercise threshold.** Assert the property was actually exercised — `>= N` ranks drew
  sizable samples (numel above a threshold so scalar/degenerate draws don't make it vacuous),
  never zero.
- **Capture must not perturb the run** — wrap fingerprinting in try/except, skip un-patchable
  C-extension entry points, never raise from the capture path.
- **Assert in the parent post-join** — the per-rank files are the test context; the workflow
  may have already torn down the process group, so do not rely on a live PG in the worker.
- Same infra as parity: `timeout`-wrap, per-collective PG timeout, tee logs, CUDA-preferred.

## More test classes & hygiene (details in [reference.md](reference.md))

- **Dtype-path unit test** per `autograd.Function`, parametrized over (activation dtype × aux/
  mask dtype × autocast on/off) — end-to-end parity is *vacuous* against a cast inserted and
  reverted inside one Function (Rule 19). Make intermediate dtypes observable with a forward-
  **pre**-hook profiler (post-hooks miss the recompute pass under `use_reentrant=False`).
- **Triton-kernel trio:** parity · edge cases · a **register-spill gate** (dump PTX, `ptxas -v`,
  assert `0 bytes spill`). Triton kernels are often **fp32-only** (raise on fp64), so do the fp64
  CP-vs-serial **parity via the REFERENCE path on both sides**, and gate the kernel *separately*
  (compiles + matches its own fp32 reference + 0 spill). To force the reference path, override the
  has-kernel flag in **every module that imported it into its own namespace** (grep the flag name) —
  setting it only on the defining module is insufficient; run eager + kernel in one process.
- **Stochastic outputs:** compare **distributions** (energy distance + Hungarian-matched lDDT/
  RMSD), never point values — sampled ensembles have no per-sample oracle.
- **Hygiene:** no global grad/RNG state in `setUp` (use `fork_rng`/local `no_grad`); an OOM
  poisons the shared CUDA context and cascades (must-fix); classify a failing dist test in a
  **fresh process**; log `grad_norm` in `on_after_backward` (it is 0 before backward).

## Test workflow & post-landing

- **Identify the proving test cases *before* implementing**; after, confirm the new code
  paths are actually exercised — if not, extend an existing test (preferred) or add one.
- **Checkpoint / serialization changes:** cover **both save and load**, including
  **backward-compatibility with the prior format** (load an old-format checkpoint), not just
  a round-trip of the new format. (See `/fold-cp:dist_lifecycle` for the resume test.)
- **Sweep `TODO`/`FIXME` after a feature lands:** grep the distributed tree + tests for
  markers referencing the new capability and resolve stale ones (remove if done, or convert
  to a concrete follow-up) so they don't rot.

## Output contract

- A parity test at the chosen level that passes against the serial reference and
  contains every anti-vacuous assertion above, parametrized over the selected 2D
  mesh configs.
- The test log on disk; a one-line statement of the tolerance used and why. The log is
  **self-verifying** — it prints the measured difference vs the tolerance and a one-line pass of
  each non-vacuous check on success, so the gate is checkable by reading it (Rule 22), not by a
  re-run.
- If the new code path is not exercised by the test, extend the test or add one —
  do not declare done on an untested path.
- For a property / invariant test (non-parity): the across-rank invariant assertion, an
  exercise-threshold check, and a documented **negative-control** result (the test fails on
  the known-bad implementation, passes after the fix).
