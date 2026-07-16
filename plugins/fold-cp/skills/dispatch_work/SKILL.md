---
name: dispatch_work
description: >
  Orchestrate a team of agents to integrate many CP modules and tests in parallel.
  Builds the work-list from the module map, derives a data-flow dependency graph,
  topologically sorts it into GPU-slot-bounded waves, and spawns a coder+reviewer
  pair per work item. Reviewers enforce serial-as-ground-truth, no-vacuous-tests,
  and the fold-cp hard rules; coupled up/down-stream items cross-review each other's
  input/output placement contracts before finalizing. Use when the scope is several
  CP modules and their tests at once — not for a single module (use dtensor_modules
  + test directly for that).
argument-hint: "[module/test scope, or 'all new modules in current_code_structure.md']"
---

# dispatch_work — parallelize CP integration with coder+reviewer pairs

Use this only when there are **multiple** modules/tests to port at once. For one
module, call `/fold-cp:dtensor_modules` + `/fold-cp:test` directly. Roles, the
spawn-prompt template, the interface-reconciliation protocol, the review gates, and
GPU partitioning are in [coordination.md](coordination.md).

This skill assumes the agent-team tooling (`Agent`/`TeamCreate`/`SendMessage`/
`Task*`). If `team_swarm:intercomm` is installed, load it as the messaging protocol
(verifiable artifact reports, lead health-checks). `code_quality:write_test` /
`run_test` are the test-discipline references for reviewers.

## Preconditions

- `docs/current_code_structure.md` (§4 module map, with reuse/adapt/new status) and
  `docs/cp_infra.md` (topology, `world_size`, GPU slots, launch recipe) exist.
- The fold-cp hard rules (`RULES.md`, injected at SessionStart) are in scope; every
  spawn prompt repeats them.

## Step 1 — Build the work-list and the dependency graph

1. Collect every module/test marked **adapt** or **new** in §4 (plus any data-feature
   work still open from `shard_data_feats`).
2. Draw edges by **data flow**: producer → consumer where the producer's output
   feeds the consumer's input. Example chain: data features → atom encoder → trunk
   (OPM, triangle mult/attn, PWA, APB, transition, pairformer) → diffusion →
   confidence → losses. Each edge is a **placement contract** (producer output
   placement == consumer input placement).
3. Two items joined by an edge are **coupled** and must cross-review (Step 4).

## Step 2 — Topologically sort into GPU-bounded waves

- Topo-sort the graph; items with no unmet dependency form the current wave.
- Bound each wave by GPU slots: `slots = floor(free_GPUs / world_size)` from
  `cp_infra.md`. Run at most `slots` CP test jobs concurrently; queue the rest.
- Independent leaves (e.g. several substrate primitives) can run fully parallel;
  a deep chain serializes by dependency, not by slot count.

## Step 3 — Spawn a coder+reviewer pair per work item

For each item in the wave, spawn two agents (template in
[coordination.md](coordination.md)):

- **coder** — **owns the CP module *and* its test**; implements per `/fold-cp:dtensor_modules` /
  `/fold-cp:test`, loads the module's row from `module_map.md`, the serial twin, and the relevant
  tech-guide §. **Verifies the fwd+bwd decomposition on a 1×1 mesh (fp64) before the multi-rank
  parity run** (`/fold-cp:test` Step 3) and **runs the test green before handoff** — this makes the
  GPU run pass first try and keeps the slot-bounded queue moving.
- **reviewer** — spawn as a **read-only `Explore` agent** (no `Edit`/`Write`): it reviews the
  coder's code **and** test against the gates in Step 5 and returns a verdict; it **never writes**
  (the coder owns all authoring + testing).

On a shared local (non-SLURM) box, give each pair a **contiguous GPU chunk** of `world_size`
devices (`CUDA_VISIBLE_DEVICES`); never let two CP jobs share GPUs; probe occupancy and HOLD if
busy (parallel tests otherwise overload the GPUs and stall). Under SLURM the scheduler handles
per-job device visibility. Track each item as a `Task` (coder + reviewer subtasks).

## Step 4 — Reconcile coupled up/down-stream contracts (the key step)

Before a coupled item is declared done, its coder and the neighbor's coder must
**cross-review the shared placement contract**:

- the upstream coder states its module's **output** placements (and feature-dict
  outputs);
- the downstream coder states its module's **expected input** placements;
- they must be **identical**; if not, reconcile (one side adjusts, or insert an
  explicit redistribute) and update both modules and `current_code_structure.md` §3.

This catches the integration bug that unit tests miss: each module passes its own
parity test but they disagree at the seam. The lead routes this exchange via
`SendMessage` and does not advance the wave until every coupled seam is reconciled.

## Step 5 — Review gates (reviewer enforces; lead verifies)

A work item is **done** only when its reviewer confirms all of:

1. **Serial is ground truth** — parity test compares against the serial fwd+bwd; no
   serial file was edited for distributed purposes (Rule 2).
2. **No vacuous test** — explicit random `grad_output`, sharding-active assert,
   replicated-identical assert, non-zero grads, fp64 default tolerances, >=1
   adversarial case (Rule 14).
3. **Hard rules** — autograd.Function conventions, explicit `from_local`
   shape/stride, promote_types, backward budget matches forward, sorted collective
   keys, control-flow scalars broadcast (Rules 6–12).
4. **Placement contract** with neighbors reconciled (Step 4).
5. **Tests pass** under `timeout`, logged; pre-commit clean.

The lead verifies the reviewer's claim by **inspecting the durable evidence** (Rule 22) — the
committed test is non-vacuous (Rule 14), the tee'd log shows measured-vs-tolerance for a run
against the current tree, and the diff touches only the distributed tree (Rule 2) — re-running
only if inspection leaves doubt, before marking the Task done.

## Step 6 — Merge, verify landing, advance

- **Squash-merge** each approved item to the target branch (worktree isolation if coders edit in
  parallel — see `isolation: "worktree"`). After merge, re-run the item's parity test on the merged
  tree to confirm it still passes (landing verification — the merge changed the tree; Rule 22), then
  **remove the item's worktree** and **shut down its coder + reviewer agents with an explicit
  `shutdown_request`** (finished, no further task; completed background teammates do not auto-dismiss
  and `TaskStop` won't remove an idle one) — freeing their GPU chunk (destroy groups, kill orphaned
  ranks). See the [coordination.md](coordination.md) **Shutdown & cleanup protocol**.
- Update `current_code_structure.md` §4 status to done; advance to the next wave.
- When all waves complete, run the workflow-level parity test (end-to-end CP vs
  serial) as the final gate.

## Alternative: stateless fan-out (prefer this for independent items)

If the items are **independent** (no coupling — e.g. porting a batch of unrelated
substrate primitives), **prefer a deterministic `Workflow` pipeline or foreground agents over a
persistent background team.** A `Workflow` pipeline (implement → parity-test → adversarial-verify
per item) and foreground `Agent` calls (omit `run_in_background`) **leave no lingering teammate** —
they return inline and never accumulate as idle rows in the agent view, whereas a messageable
background team persists as idle teammates that must each be torn down with a `shutdown_request`.
They're also lighter and minimise orchestrator round-trips (Rule 24). **Reserve the persistent
background team strictly for coupled seams that need mid-run lead↔coder messaging;** use
`Workflow` / foreground for embarrassingly-parallel ports.

## Output contract

- Every scoped module/test implemented, parity-tested green, and reviewer-approved.
- Every coupled seam reconciled and recorded in `current_code_structure.md` §3/§4.
- A final end-to-end CP-vs-serial workflow test passing.
- A summary table: item → status → test-log path → reviewer verdict.
