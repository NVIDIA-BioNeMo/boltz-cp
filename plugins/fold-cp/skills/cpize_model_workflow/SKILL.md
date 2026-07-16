---
name: cpize_model_workflow
description: >
  Orchestrate the END-TO-END integration of context parallelism into a custom co-folding /
  structure-prediction model: turn the whole effort into a prioritized, dependency-sorted
  worklist, then drive it phase by phase — map the model (learn_context), verify infra
  (build_infra), shard data features (shard_data_feats), port modules (dtensor_modules,
  fanned out via dispatch_work), prove correctness (test), wire the trainer/predictor
  (dist_lifecycle), and finally profile memory and compute (mem_profile, nsys_profile) and
  benchmark. Use as the front door when the user wants to "CP-ify my model" as a whole program
  of work — it prioritizes, sequences, and gates the long task list and delegates each task to
  the right fold-cp skill. Not for a single module (use dtensor_modules + test directly).
argument-hint: "[scope] [focus] [dp:N cp:(A,B)] [model:path] [ref:path] [--local|--slurm] [data:path|--random] [--automatic|--manual-approve] [resume]"
---

# cpize_model_workflow — conduct the whole CP integration, prioritized and gated

This is the **conductor**. It does **not** re-implement any step — it decomposes the
integration into a (potentially long) task list, **prioritizes and sorts** it by dependency /
critical path / risk / quick-win, delegates each phase to the matching fold-cp skill, and
enforces the **gate** between phases — above all *correctness before profiling*. All work obeys
the hard rules in `RULES.md` (injected at SessionStart). The task schema, priority rubric, worked
sort, per-phase gate definitions, execution-mode detail, and ledger template are in
[reference.md](reference.md).

## Phases and what each delegates to

| # | Phase | Delegate to | Gate to advance |
|---|---|---|---|
| 0 | Map the model | `/fold-cp:learn_context` | `docs/current_code_structure.md` filled (module map + feature inventory) |
| 1 | Verify infra | `/fold-cp:build_infra` | `docs/cp_infra.md`; resource + hardware/runtime **self-consistency** recorded; collective + **DTensor/`device_mesh`** smoke tests pass |
| 2 | Shard data features | `/fold-cp:shard_data_feats` | data-feature parity test green |
| 3 | Port modules | `/fold-cp:dtensor_modules` (+ `/fold-cp:dispatch_work` to fan out) | each module's parity test green |
| 4 | Wire trainer/predictor | `/fold-cp:dist_lifecycle` | stop-and-go resume test green |
| 5 | Prove correctness | `/fold-cp:test` | end-to-end CP==serial parity + property tests green, non-vacuous |
| 6 | Profile | `/fold-cp:mem_profile`, `/fold-cp:nsys_profile` | peak ≤ budget (Rule 11); compute/comm attributed |
| 7 | Benchmark | `/fold-cp:benchmark` | max-token / walltime recorded |

The orchestrator **owns only** the plan, the ordering, the gates, and the ledger
(`docs/cp_integration_plan.md`).

## Step 1 — Establish ground state

**Parse `$ARGUMENTS` by keyword (order-independent), then resolve the inputs below:**

| token | input | default |
|---|---|---|
| `inference` / `training` / `all` | **scope** (which workflow) | `all` |
| `dp:<d>` `cp:(<a>,<b>)` | **2D mesh config** (see below) | from `cp_infra.md` |
| bare `2d` | **topology** hint only (sizes deferred to `build_infra`) | from the `cp:` shape |
| `model:<path>` | the **user's model code** | resolve by tracing |
| `ref:<path>` | the **reference repo** (`$BOLTZ_CP_REPO`) | `$BOLTZ_CP_REPO` env / search |
| `--local` / `--slurm` | **launch environment** | `build_infra` decides |
| `data:<path>` / `--random` | **parity/benchmark data source** | ask, else synthesize |
| `resume` | resume mode → [Resume](#re-prioritize-on-failure--resume) | — |
| `--automatic` / `--manual-approve` | **execution mode** — continuous vs approve-each-wave (aliases: `--auto` / `--manual`) | `automatic` |
| any other token(s) | **focus area** | whole model |

- **Execution mode — `--automatic` (default) vs `--manual-approve`.** Fix it once at the front
  door: honor an explicit token without asking; else, if interactive, ask **once**
  (`AskUserQuestion`, default automatic) and record it in the ledger (`--auto`/`--manual` alias).
  `--automatic` drives the ENTIRE plan continuously to the final report — never handing off or
  proposing a fresh session — verifying every gate on disk (Rule 22) and handling setbacks
  autonomously (a red gate → [re-prioritize](#re-prioritize-on-failure--resume), never stop,
  Rule 3; a design fork → sensible default, recorded). `--manual-approve` pauses at every wave
  boundary for approval. **Full semantics — automatic vs manual behavior, the critical-blocker
  list (the only halts allowed under `--automatic`), and the per-call permission caveat — are in
  [reference.md](reference.md#execution-mode-detail).**
- **Reference repo (`$BOLTZ_CP_REPO`)** — the Boltz-CP implementation every mapping cites. Resolve
  it first (env var → `ref:<path>` → search); phase-0 `learn_context` prompts if unset. Record it
  in the ledger — *the whole workflow is undefined without it.* The **model path** (`model:<path>`)
  and **scope** feed phase-0 `learn_context`; the **launch** flag feeds phase-1 `build_infra`.
- **Mesh config — the concrete 2D device mesh used to TEST, BENCHMARK, and PROFILE.**
  `dp:<d> cp:(<cp0>,<cp1>)`; `dp` defaults to 1,
  `world_size = dp · cp0 · cp1` (mechanics + threading in
  [reference.md](reference.md#run-inputs-front-door-arguments)). It is the single mesh threaded
  to every parity test (phases 2–5), the benchmark (phase 7), and the profiles (phase 6), and it
  sets the Step-4 slot bound (`slots = floor(free_GPUs / world_size)`). Absent ⇒ `build_infra`
  picks a feasible 2D mesh.
- **Data source** (`data:<path>` | `--random`) decides whether the parity/e2e tests and the
  benchmark use real features or synthesized ones (the test skill synthesizes when none is given).
- **Focus area** (optional) names a model **subsystem or module to CP-ify first**
  (`trunk`/`pairformer`, `diffusion`, `confidence`, `data`, `losses`, or a module name); it
  **filters** the task list (Step 2) and **seeds** the sort (Step 3). **Absent ⇒ the DATA
  module/pipeline is the default ordering seed** — *not* a filter (the whole model stays in
  scope): its feature-sharding is integrated first, then propagated to every consumer (Rule 4).
  Orthogonal to `scope` and `mesh`. Full behavior in
  [reference.md](reference.md#focus-area-optional-narrowing).
- If `docs/current_code_structure.md` is missing/stale, run `/fold-cp:learn_context` first
  (phase 0) — never plan module work against an un-mapped model (Rule 1). If `docs/cp_infra.md` is
  missing, run `/fold-cp:build_infra` (phase 1).
- Read the **§4 module map**, the **§3 data-feature inventory**, and the **§4.5
  data-feature→consumer placement contract** (Rule 4) — the raw material for the task list; if
  §4.5 is missing/stale, re-run `learn_context`. **Thread each module task's input-feature
  placement contract from §4.5** so a `dtensor_modules` port wires against a *known* contract, and
  make the **synthetic test data** (`/fold-cp:test`) emit at those same placements by reusing the
  data-layer distributor (never a hand-rolled source). Why propagate once:
  [reference.md](reference.md#data-feature--consumer-contract-why-propagate-once).

## Step 2 — Build the master task list

Enumerate one task per unit of work and tag each with the attributes in
[reference.md](reference.md) (phase · delegated skill · dependencies · status reuse/adapt/new ·
GPU need · risk). Include every data-feature task, every **adapt**/**new** module, the
lifecycle items, the tests, and the two profiling passes. Persist as
`docs/cp_integration_plan.md` (template in reference.md) — the living ledger you update every
wave.

**If a focus area was given,** restrict the master list to the tasks in that subsystem **plus
their transitive dependencies** (a focused module still needs its upstream data features, infra,
and producers — Rule 4), and mark every other task **`out-of-scope`** in the ledger rather than
deleting it — so a later run can widen the focus without re-deriving the plan. With no focus
area, include everything.

## Step 3 — Prioritize and sort (the core)

**Focus area is the seed (if given):** the focus subsystem and its dependency chain are scheduled
**first**, and the subsystem is the natural scope to hand to `/fold-cp:dispatch_work`. This does
**not** override rules 1–2 — you still cannot run a focus module before its infra/data/producers,
so the dependencies come first and the focus subsystem leads as soon as it is ready.

Order the list by these rules, in precedence (full rubric + worked example in reference.md):

1. **Phase DAG is a hard barrier.** map → infra → data → modules → lifecycle → correctness →
   profile → benchmark. You cannot multi-rank-test without infra, cannot parity-test a module
   without its data features, and **cannot profile before correctness is green** (a memory or
   timing number on a wrong model is noise).
2. **Producer before consumer** within the module phase (data-flow order): data features → atom
   encoder → trunk (OPM → tri-mult/attn → PWA → attention-pair-bias → transition → pairformer) →
   diffusion → confidence → losses. Each edge is a placement contract (Rule 4).
3. **Reuse before new.** Schedule "reuse-as-is" mappings first (cheap green, shrinks the board),
   then "adapt", then "new".
4. **Risk/uncertainty first inside a ready wave.** Spike the scariest unknown early so it does
   not ambush you late: a custom kernel, the diffusion/sampling RNG (Rules 7/20 — entropy +
   liveness), a coupled multi-axis feature (Rule 8 co-sharding), a 2D-only collective. A
   1-rank "does it import / do shapes line up" spike de-risks before the full parity test.
5. **Blockers / fan-in first.** A task many others depend on — the data-feature placement
   dictionary, the mesh/manager setup, a shared substrate primitive — outranks a small leaf.
6. **Critical path.** Among independent ready tasks, start the longest dependency chain first to
   minimize wall-clock.

Then choose **fan-out vs direct** per wave: a wave with **≥3 independent** module/test items →
hand the wave to `/fold-cp:dispatch_work` (coder+reviewer pairs, GPU-slot-bounded). 1–2 items →
`/fold-cp:dtensor_modules` + `/fold-cp:test` directly. (`dispatch_work` sorts *within* the
module phase; this skill sorts *across* phases and feeds it the wave.)

## Step 4 — Drive the plan: one wave at a time, gate, then advance

Loop until the list is done:

1. Pick the next **ready** wave — dependencies met, fits GPU slots
   (`slots = floor(free_GPUs / world_size)` from `cp_infra.md`). On a shared **local (non-SLURM)**
   box, give each fan-out pair a **contiguous GPU chunk** of `world_size` devices and HOLD if busy —
   parallel tests otherwise overload the GPUs and stall (see `/fold-cp:dispatch_work`); under SLURM
   the scheduler sets per-job device visibility.
2. Delegate to the phase's skill (or `dispatch_work` for a fan-out wave).
3. **Verify the gate from durable evidence, not a prose claim (Rule 22)** — inspect the committed
   **test source** (non-vacuous — Rule 14), the **diff** (touches only the distributed tree —
   Rule 2; pre-commit clean), and the tee'd **log** (right mesh; measured-difference-vs-tolerance
   numbers for a run against the *current* tree). When all three inspect clean, that is the
   verified gate. **Re-run the test yourself only if the evidence is missing, stale, ambiguous, or
   fails inspection** — a re-run cannot make a vacuous test non-vacuous, so reading the assertions
   is the rigorous check, not the re-run. A delegated agent's "PASS" is a claim, not a gate.
4. Update the ledger **once per wave** (batched writes — Rule 23) and emit a **one-line** progress
   note to the user (done / in-flight / blocked / next) — don't re-narrate the plan each turn.
5. **Never advance past a red gate** — Rule 3: a failing test means the implementation is wrong;
   fix it, do not loosen the test. A red gate triggers [re-prioritize](#re-prioritize-on-failure--resume).

**Per-module delegation loop (what scales).** Delegate each module to **one coder agent** that
writes → **1×1-mesh fp64 pre-check** (`/fold-cp:test` Step 3) → multi-rank parity → iterates to
green. **Serialize on the GPU test, not the whole agent**: with `slots=1`, agents may draft code
in parallel but their multi-rank parity tests must not share GPUs. Reuse a **warm agent** for a
closely-related next module, or spawn a **fresh agent with a distilled "playbook"** (the
accumulated quirks) when its context bloats. Distinguish **leaf ports** (a new collective — high
risk, needs its own fresh parity test) from **compositions** (wiring already-green modules — lower
risk; the test focus is Rule-4 seam reconciliation + op-order, not new collective math). Commit at each wave boundary **only if the user has authorized committing** (otherwise leave
the distributed tree uncommitted and report it in the final summary — committing is an outward action,
never taken unprompted). Keep the loop cheap at scale: launch the wave's
independent module drafts **together** (Rule 24), have each coder **write its report/log to a file
and return only a terse verdict + artifact paths** (Rule 23), and verify each gate by **inspecting
that evidence** (Rule 22) rather than re-running by default.

## Step 5 — Correctness gate (hard firewall before profiling)

Route to `/fold-cp:test` for the **end-to-end workflow parity** (CP fwd+bwd == serial) on top of
the per-module parity from phase 3, plus the **property/invariant** tests where there is no value
oracle (RNG entropy, determinism, layout). All must be green **and non-vacuous** (Rule 14 —
explicit random `grad_output`, sharding-active, replicated-identical, non-zero grads; property
tests carry a negative control). A stochastic head (diffusion) is validated by **distribution**,
not point value (test skill). This gate is the firewall: a profiling result on an incorrect
model is meaningless, so phase 6 does not start until phase 5 is green.

## Step 6 — Profile (only on a green model)

Run `/fold-cp:mem_profile`, then `/fold-cp:nsys_profile` (they may overlap if GPUs allow). Read
them **against the rules**, not just for numbers:

- **Memory:** per-rank peak must match the forward+backward budget (Rule 11 — single O(N/cp0),
  pair O(N²/(cp0·cp1))). Watch the **transient recompute peak** (activation checkpointing moves
  the ceiling into recompute) and an **unintended dtype up-cast** inside an `autograd.Function`
  (Rule 19) — both pass parity but surface here.
- **Compute/comm:** attribute GPU time to modules (the custom python-functions trace names
  TriMul / Pairformer / trunk / diffusion); flag comm-bound stages (ring/transpose P2P) and
  kernels with register spilling (cross-ref the test skill's Triton spill gate).
- **Feedback loop:** a finding here can **reopen a phase-3 task** (a memory blowup → revisit that
  module's saved-for-backward set or dtype path). Mark it + its downstream **dirty** and re-run
  Step 3 over only the dirty set.

## Step 7 — Benchmark and report

`/fold-cp:benchmark` for max-token-at-CP-size and end-to-end walltime → `docs/cp_benchmark.md`.
Then deliver the final summary (Output contract).

## Re-prioritize on failure / resume

- **On a red gate or a profiling regression:** mark that task **and everything downstream**
  dirty, record *why* in the ledger, and re-run Step 3 over the dirty set only (do not re-sort
  the whole board). A failed seam reconciliation reopens both coupled modules (Rule 4).
- **`resume`:** read `docs/cp_integration_plan.md`, skip `done` tasks, and continue from the
  first not-done ready wave — the ledger is the source of truth across sessions.
- **Bounding context on long runs:** the ledger is the durable state, so the integration *can* be
  split across **several sessions** — but splitting is a **`--manual-approve`** convenience or a
  response to an **explicit user stop**, **never** an `--automatic` default. **Under `--automatic`
  you never hand off or propose a fresh session** — you drive to the final report, bounding context
  with lean delegation + per-wave ledger writes + terse file-based agent verdicts (Rules 23/24), not
  by stopping. When the user *does* stop (or between waves under `--manual-approve`), `resume`
  re-enters from the ledger: skip `done` tasks and re-read only the ledger + the files the current
  wave needs, not the whole accumulated transcript (Rule 23).
- **Widening the focus:** a later run with a broader (or no) focus area re-includes the tasks
  previously marked `out-of-scope`; re-run Step 3 over the newly-included set.

## Output contract

- `docs/cp_integration_plan.md`: the prioritized ledger — every task with
  phase/skill/deps/status/gate/artifact — kept current through the run.
- Each phase advanced **only** through its verified gate; end-to-end CP==serial parity green;
  property tests green with documented negative controls.
- Memory + nsys reports on disk, read against Rules 11/19, with any reopened tasks resolved.
- A final summary: what was reused vs newly built, the correctness evidence, the memory/compute
  headline numbers, and any remaining `TODO(learn_context)` / follow-ups.
