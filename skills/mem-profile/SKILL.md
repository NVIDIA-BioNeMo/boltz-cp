---
name: mem-profile
description: >
  Attribute CUDA allocator memory peaks to modules and source lines in a working CP workflow. Use to locate memory bottlenecks; use benchmark for capacity sweeps and nsys-profile for GPU timing.
argument-hint: "[e2e | trunk] [N_token] [cp size]"
license: MIT
metadata:
  author: "Dejun Lin <dejunl@nvidia.com>"
  tags: [context-parallelism, dtensor, biomolecular-modeling]
---

# mem-profile — CUDA memory profiling + peak attribution for the CP workflow

Read the shared [CP integration rules](../learn-context/references/RULES.md) before starting.

Finds **which modules and lines hold memory at each peak** of a CP workflow. Two pieces, both model-agnostic:

1. a torchrun driver that records the allocator history around the real forward
   and dumps a per-rank snapshot pickle —
   [`scripts/mem_profile.py`](scripts/mem_profile.py);
2. a pure-stdlib analyzer that turns a snapshot into a ranked, clickable markdown
   report — [`scripts/mem_profile_analysis.py`](scripts/mem_profile_analysis.py).

The skeletons are complete except for two model hooks (`build_model` /
`run_forward`) in the driver. The four load-bearing steps below are where people
get stuck; follow them in order.

## Preconditions
- A CP (or single-GPU) workflow that runs end-to-end, verified by `test`;
  `docs/cp_infra.md` for topology + `world_size`.
- `nvidia-smi` available; for CP, a multi-GPU node (square grid). **Before claiming GPUs, probe
  occupancy** (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`) and **HOLD if the
  target devices are busy — never oversubscribe (one job per slot)**; the `manage_gpu` plugin
  automates this claim/hygiene if installed.
- Know the real inference entry point (e.g. `model.infer_protein(seq)`) and how to
  build/load + CP-wrap the model — the same wiring the nsys driver uses.

## Step 1 — Record allocation history around the forward (the profiler mechanism)
PyTorch's CUDA caching allocator can log every block alloc/free with the Python+C++
stack that requested it. This is exactly what Boltz2's `CUDAMemoryProfile`
(`pl.Callback`) does; reproduce it without Lightning around one forward:
```python
torch.cuda.memory._record_memory_history(max_entries=400_000)  # start (ring buffer)
run_forward(model, args)                                        # the REAL workflow
torch.cuda.synchronize(device)
torch.cuda.memory._dump_snapshot(path)                          # write the .pickle
torch.cuda.memory._record_memory_history(enabled=None)          # stop + free buffer
```
What matters (these bite):
- **Warm up first, then `reset_peak_memory_stats`, *then* start recording.** The
  first call allocates cuBLAS/cuDNN workspaces and grows the allocator pool;
  without a warmup the snapshot is full of one-off allocations and the peak is
  wrong. One warmup forward under `torch.no_grad()` is enough.
- **`max_entries` is a ring buffer.** A deep workflow (recycles × diffusion steps)
  emits *millions* of events; if it's too small you silently lose the early
  timeline. Start at `200_000–400_000` and bump if the analyzer's event count
  looks truncated.
- **Record-around-forward vs record-from-start.** Recording only the forward keeps
  the pickle small but the timeline peak *excludes the resident baseline* (params
  /buffers allocated before recording) — it will read ~param-bytes below
  `max_memory_allocated`. That gap is usually negligible (note it in the report);
  if you need the timeline peak to match `max_memory_allocated` exactly, start
  recording before model build (`--record-from-start`, larger pickle).
- **Inference → wrap in `torch.no_grad()`.** Otherwise saved-for-backward tensors
  inflate the peak and mislead the attribution.
- Also report the allocator's own headline numbers per rank
  (`max_memory_allocated` / `max_memory_reserved`, reduced MAX across ranks); the
  reserved−allocated gap is fragmentation. The `.pickle` opens at
  **https://pytorch.org/memory_viz** for an interactive timeline + the flame graph
  of stacks live at the peak — the analyzer in Step 3 automates the same
  attribution headlessly.

## Step 2 — Launch under torchrun (one snapshot per rank)
Use **one process per CP rank with torchrun** (not `mp.spawn`) — torchrun owns the
rendezvous and gives a clean process tree:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nnodes=1 --nproc_per_node=<world> \
  -m your.mem_profile --n-token <N> --cp0 <a> --cp1 <a>
```
- The driver reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_*` from torchrun, sets
  the device from `LOCAL_RANK`, inits the process group, builds + CP-wraps the
  model, and dumps `…_rank{0..world-1}.pickle`. **Fail fast** if `RANK` is unset
  (Rule 12) — no silent single-GPU fallback.
- **`expandable_segments:True` is usually required at the ceiling** — it fights the
  fragmentation that makes `reserved ≫ allocated` and triggers `cudaMalloc`
  retries / spurious OOMs.
- fold-cp Rule 7: **seed every rank identically** and broadcast control-flow counts
  (recycles, sampling steps) so all ranks issue the same collectives in the same
  order — divergent control flow deadlocks NCCL. Per-rank snapshots will differ
  slightly; profile/attribute **rank 0** and spot-check another rank.
- Use a token count already shown to fit by `benchmark`. A short run
  can check that capture works, but take the final snapshot with the full workflow
  defaults so the reported allocations match the workload being optimized.

The driver skeleton ([`scripts/mem_profile.py`](scripts/mem_profile.py)) has the
torchrun harness, warmup, record/dump, and the per-rank MAX reduction wired up —
fill in `build_model(device, args)` (construct/load + `.eval()` + CP-wrap) and
`run_forward(model, args)` (one real, `no_grad` forward).

## Step 3 — Use the bundled analyzer

Use [`scripts/mem_profile_analysis.py`](scripts/mem_profile_analysis.py) directly.
It replays allocator events, selects distinct peaks, attributes live allocations to
project source frames, and emits a ranked markdown report. Its module docstring
and functions describe the algorithm if adaptation is needed.

The analyzer accepts PyTorch allocator snapshots containing data only. It rejects
Python globals and persistent references during loading; it does not import objects
from a snapshot. Keep snapshots from the run being profiled and check the reported
trace against the allocator headline numbers from Step 1. Empty device traces
produce a report with zero peaks; missing or malformed traces fail with a diagnostic.

CLI contract (keep these flags — they're what makes the report usable):
`snapshot` positional; `--top-n 6`, `--top-contributors 15`, `--dedup-pct 3.0`,
`--min-sep-ms 0.0`, `--project-root <repo root>` (frames under it are "project"),
`--link-style vscode|cursor|github|file|none`, `--repo-url`/`--commit` (for github
links; commit defaults to `git rev-parse HEAD`), `--out`. Defaults are
project-agnostic (`--project-root` = cwd, `--repo-url` = none).

## Step 4 — Analyze the results
```bash
python scripts/mem_profile_analysis.py <snapshot>_rank0.pickle \
  --project-root <repo-root> --top-n 6 --link-style vscode
# writes MEM_PEAK_ANALYSIS_<snapshot>.md next to the pickle
```
Read the report top-down:
- **Global peak first.** The tagged peak is the OOM ceiling. Its top contributors —
  by `% of peak` — name the module/line holding the most memory. A single
  `Class.method` dominating (e.g. a confidence head or a diffusion pair-bias at
  `[samples, heads, N, N]`) is the bottleneck to attack.
- **Compare peaks across phases.** Distinct peaks usually map to workflow stages
  (trunk recycle vs diffusion sampling vs confidence). If the largest peaks are all
  one module, optimizing it moves the ceiling; if they're spread, the budget is
  structural.
- **Cross-check the headline numbers.** Timeline peak should track
  `max_memory_allocated` (minus the baseline gap if you recorded around the
  forward). A large `reserved − allocated` means fragmentation → confirm
  `expandable_segments:True` and consider it part of the ceiling.
- **For CP specifically:** if per-rank peak does *not* drop as you grow the grid,
  something is gathering a full tensor (e.g. the trunk gathering the full pair at
  CP boundaries, or a serial/replicated diffusion+confidence running at full N on
  every rank). That's the cross-cutting limit to note — the fix is sharding that
  stage, not more GPUs.
- Write a short `MEM_PROFILE_SUMMARY.md`: the profiled workload and token count,
  dominant module per peak (with clickable links), reserved/fragmentation note,
  and concrete next optimization. Cite an existing capacity sweep if available.

## Discipline
- `timeout`-wrap every run; set a process-group timeout; tee logs under `/tmp/$USER/`.
- The big snapshot `.pickle`s (tens of MB each × ranks) are regenerable —
  **gitignore them**; commit the analyzer, the driver, and the markdown report.
- Warmup + `reset_peak_memory_stats` before recording, or the peak is wrong.
- The profiling hooks must not change the GPU work — sanity-check the peak matches a
  prior plain run.

## Output contract
- `scripts/mem_profile.py` filled in for the model + per-rank `…_rank*.pickle`
  (gitignored), with `peak_alloc`/`peak_reserved` reported.
- `MEM_PEAK_ANALYSIS_<snapshot>.md` (peaks → contributors, clickable links) and a
  `MEM_PROFILE_SUMMARY.md` (profiled workload, bottleneck module, next
  optimization).
- A one-paragraph readout: profiled N, the dominant module at the global
  peak, fragmentation/headroom, and whether per-rank memory scales with the CP grid.
