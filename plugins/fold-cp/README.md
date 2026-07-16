# fold-cp

Integrate **Boltz-style context parallelism (CP)** into a custom co-folding /
structure-prediction model. The skills assume the user's model is *not* shaped like
the Boltz serial code — every skill begins by exploring the actual usage site and
mapping it onto the Boltz-CP reference implementation. The plugin **bundles** the
high-level CP technical guide ([`reference/cp_tech_guide.md`](reference/cp_tech_guide.md)); the
reference repo provides the distributed code to mirror (`src/boltz/distributed/`).

Point the skills at the reference repo by setting `BOLTZ_CP_REPO` (or answering the
prompt in `learn_context`).

## Skills

| Skill | Purpose |
|---|---|
| **`/fold-cp:cpize_model_workflow`** | **The conductor / front door.** Turns the whole CP integration into a prioritized, dependency-sorted, *gated* worklist (`docs/cp_integration_plan.md`) and drives it phase by phase, delegating each task to the skills below — enforcing *correctness before profiling*. Use to CP-ify a model end-to-end; run the individual skills directly for a single step. |
| **`/fold-cp:learn_context`** | Explore the user's model — inference & training entry points, framework (Lightning/DeepSpeed), data-feature format, featurization — and map components onto Boltz-CP. Writes `docs/current_code_structure.md`. **Run first.** |
| **`/fold-cp:build_infra`** | Probe GPUs (count/topology) and software (torch+CUDA/NCCL), decide whether 2D-CP is testable (≥4 GPUs in a perfect-square mesh; integration 8), run shipped P2P/all-gather/all-reduce/DTensor smoke tests, or escalate to SLURM. Writes `docs/cp_infra.md`. |
| **`/fold-cp:shard_data_feats`** | Implement DTensor data-feature sharding: placement dictionary, atom pack/pad/scatter, per-shard + cross-axis padding, DTensor Dataset/DataLoader/DataModule + collate, and the data parity test. |
| **`/fold-cp:dtensor_modules`** | Port serial modules to CP via the serial→CP dictionary ([`module_map.md`](skills/dtensor_modules/module_map.md)) with exact placements/collectives/backward budgets, following the `autograd.Function` conventions. |
| **`/fold-cp:test`** | Write multi-rank `mp.spawn` parity tests proving CP == serial (fwd+bwd), with the anti-vacuous rules; plus a non-parity **property/invariant** test class (e.g. `test_rng_entropy`). Synthesizes random features when no real data exists. |
| **`/fold-cp:dist_lifecycle`** | Wire ported CP modules into a real trainer/predictor: device-placement-before-wrapping, all-trainable-params-are-DTensors (placeholder/freeze unimplemented), checkpoint save (DTensor→plain) / load (realign + redistribute optimizer state), resume seed offset, and DTensor-safe EMA; verified by a stop-and-go resume test. |
| **`/fold-cp:dispatch_work`** | Orchestrate coder+reviewer agent pairs across a data-flow dependency graph to integrate many modules/tests at once, with up/down-stream placement-contract reconciliation. |
| **`/fold-cp:benchmark`** | Find the max token count that fits at a CP size and record end-to-end walltime; emits a results table to `docs/cp_benchmark.md`. |
| **`/fold-cp:nsys_profile`** | Profile where the CP workflow spends GPU time and communication with NVIDIA Nsight Systems — torchrun launcher, the right `nsys` CLI options, and a custom `--python-functions-trace` JSON that names high-level modules (TriMul / Pairformer / trunk / diffusion). |
| **`/fold-cp:mem_profile`** | Memory-profile the CP workflow with the PyTorch allocator history (`_record_memory_history`/`_dump_snapshot`) under torchrun, then attribute the top-N memory peaks to modules/lines via a stdlib analyzer that emits a clickable markdown report ([`mem_profile_analysis.py`](skills/mem_profile/scripts/mem_profile_analysis.py)). |

## Hook (CP hard rules)

A `SessionStart` hook injects [`hooks/RULES.md`](hooks/RULES.md) — the high-level
hard requirements (serial-as-ground-truth, no implicit DTensor ops, explicit
collectives + P2P parity ordering, fp32 grad reduction + aux-tensor dtype, backward
budget, reduction-axis discipline, distributed lifecycle/checkpoint, no-vacuous-tests,
distributed-test timeouts) plus the agentic workflow rules — into every session, like a
CLAUDE.md.

An **opt-in** `PreToolUse` guard
([`hooks/guard_serial_and_tests.py`](hooks/guard_serial_and_tests.py)) mechanically
enforces two of them once the project contains `.fold-cp/config.json` (written by
`learn_context`): it blocks edits to designated serial-reference files (Rule 2) and
blocks distributed-test commands missing a `timeout` prefix (Rule 16). With no
config file the guard is inert.

## Typical flow

`/fold-cp:cpize_model_workflow` orchestrates the sequence below — prioritizing, sequencing,
and gating the task list — and delegates each phase to these skills (run any directly for a
single step):

```
cpize_model_workflow   (conductor: prioritize · sequence · gate · ledger)
   └─ learn_context → build_infra → shard_data_feats → dtensor_modules → test → dist_lifecycle
                                         │                    │            │
                                         └──────── dispatch_work (many at once) ┘
                                           → [correctness gate] → mem_profile / nsys_profile → benchmark
```

## Testing the plugin

```bash
claude --plugin-dir ./fold-cp        # load locally
claude plugin validate ./fold-cp     # validate manifest + frontmatter + hooks
```

Then run `/manage-plugins:review` to check quality and security.
