---
name: benchmark
description: >
  Benchmark a CP inference or training workflow to find the maximum token count
  that fits at a given CP size and to record end-to-end walltime. Sweeps input size
  (tokens N, atoms, MSA depth S) over real or synthesized data, ramps N until OOM to
  report the max, times fwd (inference) or fwd+bwd+step (training) with warmup and
  median-of-repeats, captures peak memory, and writes a results table to
  docs/cp_benchmark.md. Use once a CP workflow runs end-to-end, to quantify the
  memory headroom CP buys and the walltime cost.
argument-hint: "[inference | training] [data path | --random] [cp size]"
---

# benchmark — max tokens and walltime for the CP workflow

CP exists to push the maximum sequence length past what one GPU holds; the headline
number is **the largest N that fits at a given CP size**, alongside the walltime to
process representative inputs. The driver skeleton is
[`scripts/bench_driver.py`](scripts/bench_driver.py); adapt its three hooks to the
user's model.

## Preconditions

- A CP workflow that runs end-to-end (forward for inference; fwd+bwd+step for
  training), verified by `/fold-cp:test`.
- `docs/cp_infra.md` gives topology, `world_size`, GPU slots, launch recipe.

## Step 1 — Fix the workflow and CP config

- Workflow: `inference` (time forward + sampling) or `training` (time
  fwd+bwd+optimizer step). Clarify with the user if ambiguous — they measure
  different things.
- CP config: topology + `cp` size from `cp_infra.md`. Hold it fixed within one
  benchmark so numbers are comparable; sweep CP size only as an explicit second axis.

## Step 2 — Choose the data and sweep axes

- **Real data:** use the user's inputs; record their `(N_tokens, N_atoms, S)`.
- **Synthesized:** generate random features per the inventory contract
  (`current_code_structure.md` §3), sweeping `N_tokens` (the headline axis), and
  `N_atoms`, `S` as secondary. Keep masks valid and `atom_to_token` consistent.

## Step 3 — Max-token probe (the headline)

For the fixed CP size, ramp `N` upward (e.g. geometric: 512, 1024, 2048, …; then
bisect the last gap) running a single forward each time, until it OOMs. Report the
**largest N that fits** and its peak memory. Reset CUDA memory stats
(`torch.cuda.reset_peak_memory_stats()`) between points and catch
`torch.cuda.OutOfMemoryError` to ramp gracefully. Report this per CP size if a CP
sweep is requested — that curve is the CP value proposition.

## Step 4 — Walltime measurement (disciplined)

For each `(N, atoms, S)` point that fits:

- **Warm up** (≥3 iters, discard) so cuDNN/autotune/allocator settle.
- Time `repeats` (≥5) iterations with `torch.cuda.synchronize()` around each;
  report the **median** (robust to jitter), not the mean.
- Record **peak memory** (`max_memory_allocated`) for the point.
- One job per GPU slot; note GPU clocks are not locked (or lock them for tighter
  numbers) and the node is/ isn't shared.

## Step 5 — Emit the results table

Write `docs/cp_benchmark.md` with a table and the run metadata (date, GPU model,
CP config, torch/cuda versions, clocks-locked y/n):

```markdown
| N_tokens | N_atoms | MSA_seq | CP config | peak_mem (GiB) | walltime (ms) | <user col> |
|---------:|--------:|--------:|-----------|---------------:|--------------:|------------|
| 1024     | 8192    | 128     | 2d cp=4   | 31.2           | 842           | …          |
| max-fit  | …       | …       | 2d cp=4   | 78.9 (OOM@N+δ) | …             | …          |
```

Add any user-requested columns (e.g. tokens/sec, per-module time, energy). Save the
raw CSV next to it for re-plotting.

## Discipline

- `timeout`-wrap every run; tee logs under `/tmp/$USER/`.
- Compare only like-for-like (same CP size, dtype, batch) across rows.
- If `high_performance_computing:benchmark-design` is installed, follow its
  warmup/repeat/drift-cancellation guidance for publishable numbers.

## Output contract

- `docs/cp_benchmark.md` with the max-token result per CP size and a walltime table
  over the swept points, plus run metadata; raw CSV saved alongside.
- A one-paragraph readout: max N at the tested CP size vs the single-GPU baseline (if
  measurable), and the dominant cost (compute vs communication) if known.
