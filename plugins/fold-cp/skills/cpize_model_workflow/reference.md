# cpize_model_workflow — reference

Detail behind the conductor: the task schema, the priority rubric, a worked sort, the
per-phase gate definitions, the fan-out thresholds, and the ledger template. SKILL.md is the
loop; this is the bookkeeping.

## Run inputs (front-door arguments)

Resolved once in SKILL.md Step 1 and threaded to the phase that needs them:

| input (`$ARGUMENTS`) | form | threads to |
|---|---|---|
| reference repo | `ref:<path>` or `$BOLTZ_CP_REPO` env | phase 0 `learn_context` — every mapping cites it |
| model code path | `model:<path>` | phase 0 `learn_context` |
| scope | `inference` / `training` / `all` | phase 0 `learn_context`; gates which phases run |
| **mesh config** | `dp:<d> cp:(<a>,<b>)` (2D) | phase 1 `build_infra` + the `cp size` of every test / benchmark / profile + Step-4 slots |
| launch env | `--local` / `--slurm` | phase 1 `build_infra` |
| data source | `data:<path>` / `--random` | parity tests (phases 2–5) + benchmark (phase 7) |
| focus area | subsystem / module | Steps 2–3 filter + sort (below) |
| resume | `resume` | re-enter from the ledger |

The **mesh config is the primary device mesh for testing, benchmarking, and profiling**:
`cp:(a,b)` ⇒ 2D (`cp0×cp1`), `dp` defaults to 1, `world_size = dp·cp0·cp1`. Pin it at the
front door so the parity tests, the benchmark, and the mem/nsys profiles all run on the **same**
mesh; it is the single source for each downstream skill's `cp size` argument and for the GPU-slot
bound.

## Execution mode (detail)

The front-door summary is in SKILL.md Step 1; this is the full behavior of `--automatic`
(default) vs `--manual-approve`. Fix the mode once: honor an explicit `--automatic`/`--auto` or
`--manual-approve`/`--manual` token without asking; if none was given and the session is
interactive, ask **once** (`AskUserQuestion`, default automatic). Record it in the ledger.

- **`--automatic` (default) — execute the ENTIRE plan continuously**, wave after wave, through to
  the final report, **without pausing for confirmation and without ever proposing to stop or to
  resume in a fresh session.** Bound context with lean delegation + per-wave ledger writes + terse
  file-based agent verdicts (Rules 23/24) — **never** by handing off. Gates are still verified on
  disk every wave (Rule 22). Handle setbacks autonomously: a **red gate** triggers the re-prioritize
  loop (diagnose and fix the implementation — do **not** stop, Rule 3); a **design fork** is resolved
  by the sensible default and recorded in the ledger. **Halt only on a confirmed _critical blocker_**
  (below) — and then report exactly what is blocking and what input is needed, rather than silently
  stopping.
- **`--manual-approve` — pause at every wave boundary.** Before starting the next wave, give the
  user (a) a one-screen **summary of the phases/waves finished** and their verified gates, (b) the
  **next wave/phase plan** (its tasks, fan-out-vs-direct, GPU-slot bound), and (c) an explicit
  **request to continue**; proceed only on approval. Re-prioritize/resume work the same way; the
  pauses are the only difference from automatic.
- **Critical blockers — the ONLY conditions that may halt `--automatic`:** no usable GPU, or a
  device mesh that cannot satisfy the required `world_size`; a required software dependency that
  cannot be installed in any reachable environment; the reference repo or another required input
  that cannot be located. A large/long transcript, a red gate, and a resolvable design choice are
  **not** blockers — keep driving.
- **Permission note:** if the agentic runtime is **not** known to be in a standing
  "don't-ask"/bypass-permissions mode, **warn once** that individual actions (shell commands, file
  writes, multi-rank launches) may still require per-call approval even under `--automatic`, then
  proceed.

## Task attribute schema

Every task in the master list carries:

| Field | Values | Drives |
|---|---|---|
| `id` | short slug (`feat:atom_to_token`, `mod:tri_mult`, `test:trunk`) | references in the ledger |
| `phase` | 0–7 (see SKILL.md table) | the hard phase barrier |
| `skill` | the `/fold-cp:*` skill that executes it | delegation |
| `deps` | ids of upstream tasks (data-flow producers, infra) | topo-sort + ready-wave test |
| `status_kind` | `reuse` / `adapt` / `new` | reuse-before-new ordering (rule 3) |
| `state` | `todo` / `in_flight` / `done` / `blocked` / `dirty` / `out-of-scope` | the drive loop + resume; `out-of-scope` = excluded by the focus area (Step 2), re-includable on widening |
| `gpu` | `world_size` it needs to test (1 for spikes, ≥4 for 2D) | slot bounding |
| `risk` | `low` / `med` / `high` (+ one-line reason) | risk-first ordering (rule 4) |
| `gate` | the phase gate that closes it (below) | "done" definition |
| `artifact` | path of the proof (test log, doc, profile report) | gate verification |

High-risk flags worth setting explicitly: a **custom/Triton kernel**, the **diffusion/sampling
RNG** (Rules 7/20), a **coupled multi-axis feature** (atom↔token, pair — Rule 8 co-sharding), a
**2D-only collective** (Cannon skew, transpose), anything **fp64-fragile**, or a module with a
**heavy backward budget** (Rule 11).

## Focus area (optional narrowing)

A **focus area** (passed in `$ARGUMENTS`) scopes the conductor to part of the model instead of
the whole program. It is a model **subsystem or module** — `trunk`/`pairformer`, `diffusion`,
`confidence`, `data` (featurizer/pipeline), `losses`, or a named module — resolved against the §4
module map / §3 feature inventory. It is **orthogonal** to `scope` (inference/training/all —
*which workflow*) and the 2D mesh config; focus is *which part of the model*.

- **Filter (Step 2):** keep the focus subsystem + its **transitive dependencies**; mark the rest
  `out-of-scope` (don't delete — a later run can widen).
- **Sort (Step 3):** the focus chain is the seed — scheduled first, subject to the phase /
  producer barriers (rules 1–2).
- **Hand-off:** the focus subsystem is the natural `/fold-cp:dispatch_work` wave scope.

Absent ⇒ no narrowing (the whole model stays in scope), but the **data module / pipeline is the
default ordering seed**: its feature-sharding requirements are integrated first and **propagated to
every consumer** (Rule 4) and the synthetic test data, with the rest following producer→consumer order.
Pass an explicit focus to override.

## Data-feature → consumer contract (why propagate once)

Phase-0 `learn_context` must produce the **§4.5 data-feature→consumer placement contract** (which
CP module consumes each feature and the placement it must arrive at — Rule 4), not just the §3
inventory and §4 module map. Thread each module task's input-feature placement contract from §4.5
so a `dtensor_modules` port wires against a *known* contract instead of re-deriving it. In a recent
run the shared MSA-feature placement was re-reconciled **three times** — OPM, PWA, and the MSA
module — for exactly this missing-propagation reason; deriving it once here avoids that. The
synthetic test data (`/fold-cp:test`) must emit features at these same placements by reusing the
data layer's distributor — never a hand-rolled source that could violate the contract. If §4.5 is
missing or stale, re-run `learn_context` before planning module work.

## Priority rubric (tie-break ladder)

Apply the six SKILL.md ordering rules as a **strict ladder** — earlier rules dominate later ones.
When two ready tasks are still tied after the ladder, prefer the one that **unblocks more
downstream tasks** (highest out-degree in the dependency graph), then the smaller one (faster
green). Concretely, to order a set of *ready* tasks (deps already done):

1. lower `phase` first (hard);
2. within a phase, **producer before consumer** (topo order of the data-flow edges);
3. `reuse` before `adapt` before `new`;
4. `risk=high` before `med`/`low` (spike the unknown early);
5. higher fan-in/blocker score (more tasks depend on it);
6. longer remaining dependency chain (critical path);
7. tie-break: higher out-degree, then smaller task.

You do not need a numeric score — the ladder resolves almost every pair. Reach for a weighted
score only when sequencing a very large module phase; even then, `dispatch_work` re-sorts within
that phase, so this skill only needs the *wave* order right.

## Worked sort (illustrative ~18-task list)

Given a model mapped to: data features (token_single, pair, msa, atom_single, atom_to_token,
frames_idx); modules atom_encoder, OPM, tri_mult, tri_attn, PWA, attn_pair_bias, transition,
pairformer, diffusion, confidence; losses distogram, diffusion, smooth_lddt; trainer wiring;
end-to-end test; mem + nsys profile. A correct wave ordering:

- **Wave A (phase 0–1):** `learn_context`, then `build_infra`. Everything blocks on these.
- **Wave B (phase 2, blockers first):** the **placement dictionary** + `token_single`/`pair`/`msa`
  features (fan-in for all modules), then `atom_single`; **`atom_to_token` + `frames_idx` flagged
  high-risk** (Rule 8 co-sharding, Rule 20 index remap) → spike on 1 rank first, then full data
  parity test.
- **Wave C (phase 3, producer→consumer, reuse first):** `atom_encoder` (consumes atom_to_token) →
  then the trunk substrate that the pairformer needs. Independent trunk leaves (`OPM`, `tri_mult`,
  `tri_attn`, `PWA`, `transition`, `attn_pair_bias`) are a **fan-out wave → `dispatch_work`**
  (coder+reviewer, GPU-slot-bounded). `pairformer` waits on all of them (highest fan-in →
  scheduled as the wave's closing item).
- **Wave D (phase 3 cont.):** `diffusion` (**high-risk: sampling RNG** — do the RNG-entropy
  property spike here, Rule 20) → `confidence` → losses (`distogram` is the canonical fused
  autograd.Function; `smooth_lddt`/`diffusion` losses follow).
- **Wave E (phase 4):** `dist_lifecycle` wiring → stop-and-go resume test.
- **Wave F (phase 5):** end-to-end CP==serial parity + property tests (the firewall).
- **Wave G (phase 6–7):** `mem_profile` ∥ `nsys_profile` → `benchmark`.

Note how risk-first pulls `atom_to_token` and `diffusion` RNG spikes *earlier within their
phase* without violating the phase barrier, and blockers (`placement dictionary`, `pairformer`
fan-in) are sequenced by dependency, not size.

## Per-phase gate definitions (what "done" means)

A phase advances only when its gate is verified **on disk** (Rule 3 / Rule 14):

- **0 Map:** `docs/current_code_structure.md` exists; §3 feature inventory and §4 module map
  filled (or explicit `TODO(learn_context)` markers, not silent gaps).
- **1 Infra:** `docs/cp_infra.md` records topology + `world_size` + launch recipe; the shipped
  P2P / all-gather / all-reduce / DTensor smoke tests pass (build_infra).
- **2 Data:** the data-feature parity test is green; sharded local shape < global on sharded
  dims (when either 2D mesh axis has size > 1); masks/indices consistent (Rules 8/9);
  collective loops use sorted keys (Rule 7).
- **3 Module (per module):** parity test green vs serial fwd+bwd, non-vacuous (Rule 14);
  backward budget matches forward (Rule 11); coupled seams reconciled (Rule 4); no serial file
  edited (Rule 2); pre-commit clean.
- **4 Lifecycle:** stop-and-go resume test green — resumed weights/optimizer match an
  uninterrupted run; checkpoint is plain tensors and loads into the serial model; prior-format
  checkpoint still loads (back-compat).
- **5 Correctness:** end-to-end CP==serial parity green; property/invariant tests green with a
  documented **negative control**; stochastic outputs compared by distribution.
- **6 Profile:** per-rank peak within the Rule 11 budget (no O(N)/O(N²) backward surprise, no
  transient recompute/dtype blowup — Rules 11/19); compute/comm attributed to modules; no
  unexpected register-spilling kernel.
- **7 Benchmark:** max-token-at-CP-size and end-to-end walltime recorded in
  `docs/cp_benchmark.md`.

## Fan-out vs direct (per wave)

- **≥3 independent** ready module/test items → `/fold-cp:dispatch_work` (persistent coder+reviewer
  team, cross-review of coupled seams, GPU-slot-bounded waves).
- **1–2 items, or a deep serial chain** → `/fold-cp:dtensor_modules` + `/fold-cp:test` directly.
- **Independent, uncoupled batch** (e.g. several unrelated substrate primitives) → a stateless
  `Workflow` fan-out is lighter than a team (dispatch_work "Alternative" section).
- Never let two CP test jobs share GPUs; bound concurrency by `floor(free_GPUs / world_size)`.

## Ledger template — `docs/cp_integration_plan.md`

```markdown
# CP integration plan (cpize_model_workflow)

- ref ($BOLTZ_CP_REPO): <path>   model: <path>   launch: <local|slurm>   data: <path|random>
- scope: <inference|training|all>   focus: <subsystem|—>   mesh: dp=<d> cp=(<a>,<b>)   (world_size=<N>)
- Updated: <date>   |   legend: state = todo / in_flight / done / blocked / dirty / out-of-scope

| id | phase | skill | deps | kind | risk | state | gate / artifact |
|---|---|---|---|---|---|---|---|
| learn_context | 0 | learn_context | — | — | low | done | docs/current_code_structure.md |
| build_infra | 1 | build_infra | learn_context | — | low | done | docs/cp_infra.md |
| feat:placements | 2 | shard_data_feats | build_infra | new | med | in_flight | tests/.../test_dtensor_distribute.py |
| feat:atom_to_token | 2 | shard_data_feats | feat:placements | new | high | todo | (co-shard; 1-rank spike first) |
| mod:atom_encoder | 3 | dtensor_modules | feat:atom_to_token | adapt | med | todo | tests/.../test_dtensor_encoders.py |
| mod:tri_mult | 3 | dispatch_work | feat:pair | adapt | med | todo | ... |
| ... | | | | | | | |
| test:e2e | 5 | test | <all mod> | new | med | todo | end-to-end CP==serial |
| prof:mem | 6 | mem_profile | test:e2e | — | low | todo | docs/mem_report.md |
| prof:nsys | 6 | nsys_profile | test:e2e | — | low | todo | nsys report |
| bench | 7 | benchmark | prof:* | — | low | todo | docs/cp_benchmark.md |

## Decisions / reopen log
- <date>: <task> reopened dirty because <profiling/parity finding>; downstream <ids> re-queued.
```

Keep the table sorted by the rubric so the next ready wave is always near the top; append the
reopen log rather than rewriting history, so the re-prioritization decisions stay auditable.
