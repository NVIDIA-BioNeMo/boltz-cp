# dispatch_work — coordination reference

## Roles per work item

| Role | Does | Loads |
|---|---|---|
| **coder** | **owns** the CP module **and** its test; runs the parity test **green** before handing off | `fold-cp:dtensor_modules` or `fold-cp:test`; `module_map.md` row; serial twin; tech-guide § |
| **reviewer** (**read-only** — spawn as an `Explore` agent; no `Edit`/`Write`) | reviews code + test against the Step-5 gates; returns `APPROVED`/`REVISE`/`BLOCK`; **never writes** (coder owns all authoring + testing) | `code_quality:write_test`, `fold-cp` RULES.md |
| **lead** (you) | builds the graph, spawns pairs, routes cross-review, verifies artifacts, merges, advances waves | this skill |

One coder+reviewer pair per item. The reviewer is a **separate, read-only `Explore` agent** so
review is adversarial and cannot self-certify — it inspects and verdicts, never edits.

## Spawn-prompt template (task-first, rules always included)

```
TASK: <implement CP module X / write parity test for X>
LEVEL: <unit | layer | module | workflow>
TOPOLOGY: 2d, world_size=<N>
CONTRACT (from module_map.md): serial=<file>; cp=<file>; tech §=<n>;
          input placements=<...>; output placements=<...>;
          collectives=<...>; backward budget=<single|pair>.
COUPLED WITH: <neighbor item(s)> — you MUST reconcile the shared placement
          contract with their coder before declaring done (see Reconciliation).
FILES: <serial twin path:lines>, <feature inventory rows>, <tech guide §>
LOAD SKILL: <fold-cp:dtensor_modules | fold-cp:test> (+ team_swarm:intercomm if present)
DELIVERABLE: <cp file path> + <test path> + tee'd test log path
RETURN: a terse verdict + the artifact paths above (numbers, not narrative) — Rule 23
RULES: point at fold-cp RULES.md by PATH (the SessionStart hook injects it; don't paste all
       rules inline — Rule 23) and name the few most load-bearing for THIS task.
ENVIRONMENT: <conda/module activation>, working dir, BOLTZ_CP_REPO=<path>
GPU ASSIGNMENT: CUDA_VISIBLE_DEVICES=<slot>
```

Never paste another agent's message history; point at the on-disk doc/contract (Rule 23). Each
agent writes its report to a file and returns only a terse verdict + artifact paths, so the lead's
context stays small (Rule 23).

## Reconciliation protocol (coupled items)

For an edge `upstream → downstream`:

1. Upstream coder posts its **output contract**: for each output tensor and each
   feature-dict field it produces, the `(placement tuple, shape semantics, dtype,
   mask convention)`.
2. Downstream coder posts its **input contract**: the same fields it expects.
3. Compare field-by-field. They must be **identical**. On mismatch, the lead picks
   the fix (usually: the side that diverges from the serial data semantics adjusts;
   or insert an explicit `redistribute_transpose` / placement change at the seam).
4. Both coders update their modules and the shared rows in
   `current_code_structure.md` §3/§4; both re-run their parity tests.
5. The seam is reconciled only when both sides' tests pass with the agreed contract.

The lead does not advance the wave while any coupled seam is unreconciled.

## Review gates (reviewer checklist)

```
[ ] serial fwd+bwd is the comparison ground truth; no serial file edited (Rule 2)
[ ] explicit random grad_output; NO .sum().backward() (Rule 14)
[ ] sharding active when either 2D mesh axis has size >1: local.shape[shard_dim] < global on sharded dims (skip at cp=(1,1))
[ ] distributed signatures accept only DTensor (placements + shapes checked); no naked plain-tensor distributed code (Rule 6)
[ ] replicated values identical across ranks
[ ] gradients non-zero and finite
[ ] fp64 + assert_close default tolerances; tolerance derived not tuned (Rule 15)
[ ] autograd.Function: asserts placements/mesh/even-shard; rejects Partial (Rule 8)
[ ] from_local has explicit shape/stride/placements (Rule 9)
[ ] promote_types not .float() (Rule 10)
[ ] backward saved tensors obey forward per-rank budget (Rule 11)
[ ] sorted keys in collective loops; control-flow scalars broadcast (Rule 7)
[ ] serial attribute names + registration order mirrored
[ ] placement contract with neighbors reconciled (coupled items)
[ ] test command timeout-wrapped; log shows pass with measured-vs-tolerance numbers (non-vacuous); pre-commit clean (Rules 16/22)
```

Verdicts: `APPROVED` (lead marks Task done after artifact verification) / `REVISE`
(specific items; coder resubmits) / `BLOCK` (fundamental — re-plan).

## GPU partitioning (local / non-SLURM only)

> Under **SLURM** the scheduler sets each job's device visibility at submission (one job = its
> allocated GPUs), so the lead does **not** hand-assign `CUDA_VISIBLE_DEVICES` — submit one
> `srun`/`sbatch` per item with the right `--gpus`. The rules below apply only when the lead manages
> a **shared local box**.

- `slots = floor(free_GPUs / world_size)` from `cp_infra.md`. Each active CP job (a coder running a
  parity test) gets a **contiguous chunk of `world_size` GPUs** via `CUDA_VISIBLE_DEVICES` — e.g.
  8 GPUs, `world_size=4`: pair 1 → `0,1,2,3`; pair 2 → `4,5,6,7`. **Never share GPUs across two CP
  jobs.**
- **Parallel-test overload is a real stall.** Pairs run concurrently (test util `find_free_port`
  enables it); without chunked assignment + the probe below, jobs pile onto the same GPUs and
  development halts. Cap concurrency at `slots`.
- Before launching, check occupancy
  (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`); HOLD if the target chunk is
  busy. Priority: correctness/parity tests > implementation iteration > benchmarks.
- Round-robin queued items as slots free. Coders may edit in parallel git worktrees
  (`isolation: "worktree"`) to avoid file conflicts; a single merger lands them.

## Shutdown & cleanup protocol

- **When a teammate finishes its assigned task and has no further assignment, the lead shuts that
  agent down with an explicit `shutdown_request`** — a `SendMessage` of
  `{type: "shutdown_request"}` to the teammate, which it approves and then exits. Don't keep idle
  coder/reviewer agents alive. **Why it must be explicit:** completed background teammates do **not**
  auto-dismiss, and `TaskStop` only stops a *running* task (it will **not** remove an idle one), so
  the `shutdown_request` is the only teardown the lead can drive — skip it and the teammate lingers
  as an idle row. Before exiting, the agent destroys its process group(s) and **kills any orphaned
  multi-rank processes so its assigned GPU chunk is freed** (no leaked NCCL/python process holding
  GPU memory).
- **After the reviewer approves an item: squash-merge** it to the target branch, then **remove its
  worktree** (`isolation: "worktree"` auto-cleans if unchanged; otherwise remove it explicitly).
- **End of run:** re-probe `nvidia-smi` to confirm no orphaned GPU processes, no leftover worktrees,
  and all process groups destroyed before the lead exits. Any teammate that wasn't torn down stays
  as an idle row in the agent view — the user can clear leftovers there with **`Ctrl+X` twice**
  (or `Ctrl+X` on the group header to clear the whole group).

## Lead operating loop

1. Build graph + waves; `TeamCreate`; `TaskCreate` per item (coder+reviewer subtasks).
2. Spawn the current wave's pairs **together** (one batch of launches — Rule 24), each with the
   template above + a slot.
3. Route reconciliation messages for coupled seams; verify every artifact claim by **inspecting the
   durable evidence** (Rule 22) — `Read` the committed test (non-vacuous, Rule 14), the diff
   (Rule 2), and the tee'd log (mesh + measured-vs-tolerance) — re-running only if inspection
   leaves doubt; never trust a prose claim.
4. On reviewer `APPROVED` + verified artifacts → mark Task done, merge, landing-verify.
5. When the wave drains, advance to the next; at the end, run the end-to-end test.
