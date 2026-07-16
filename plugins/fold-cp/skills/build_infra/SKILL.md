---
name: build_infra
description: >
  Probe and establish the distributed test infrastructure for CP development.
  Inventories local GPUs (count, model, memory, NVLink topology), checks the
  software stack (Python, PyTorch+CUDA, NCCL, torch.distributed), and runs shipped
  smoke tests for batch_isend_irecv P2P, all_gather, all_reduce, reduce_scatter,
  and a DTensor round-trip. Maps the GPU count to whether 2D CP is testable
  (2D needs >=4 GPUs; full integration wants 8). When insufficient local GPUs
  are available, generates a SLURM submission template and guides the user to
  provide cluster access. Records docs/cp_infra.md. Use after
  learn_context, before shard_data_feats / dtensor_modules / test.
argument-hint: "[--local | --slurm] [requested world size]"
---

# build_infra — stand up and verify the CP test environment

CP code cannot be developed or trusted without multi-GPU execution. This skill
proves the environment can run the collectives CP depends on, and decides which CP
configurations are testable here. Run the shipped scripts; do not hand-wave the
checks. All scripts live in [`scripts/`](scripts) under this skill
(`${CLAUDE_SKILL_DIR}/scripts`).

## Step 1 — Inventory the hardware

```bash
nvidia-smi -L                                   # count + model
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv
nvidia-smi topo -m                              # NVLink / PCIe topology
```

Note which GPUs are **free** (other users may occupy a shared node — respect
`manage_gpu` hygiene if available: never use a GPU you did not claim).

## Step 2 — Map free GPUs to testable 2D-CP configurations

| Free GPUs | 2D-CP | Integration |
|---|---|---|
| **>=8** | yes — `cp0=cp1=2` (+ `dp>=2`) | full end-to-end |
| **4–7** | yes — `cp0=cp1=2` | limited |
| **0–3** | no (needs a perfect square >=4) | escalate to SLURM (Step 5) |

2D-CP requires `size_cp` to be a perfect square (`cp0=cp1`); the smallest is 4.

Record the chosen config(s).

## Step 3 — Probe the software stack

```bash
python "${CLAUDE_SKILL_DIR}/scripts/probe_env.py"
```

It reports Python / PyTorch / CUDA / NCCL versions, `torch.cuda.is_available()`,
per-GPU compute capability and memory, **free disk space**, the **NVIDIA driver
version**, and a **self-consistency verdict** (see the next section), plus a
recommended CP-config JSON derived from the live GPU count. If PyTorch lacks a CUDA
backend, or versions are incompatible with the target model, stop and surface it to
the user. (Point the disk check at your real work dir with
`FOLD_CP_WORKDIR=<path>`; it defaults to the cwd.)

**Resolve the ONE environment that satisfies BOTH constraints**: a torch built for the
GPUs' arch (e.g. sm_120/Blackwell needs a recent CUDA) AND the target model importable
in it. The interpreter may not be on `PATH` (pixi/conda) and the model may not be
pip-installed — find the env where both `import torch` (right arch) and
`python -c "import <model_pkg>"` succeed, and record its **absolute interpreter path**
in `docs/cp_infra.md`. Also probe the **multi-rank launch prerequisites** and bake them
into the recorded recipe:
- **`PYTHONPATH=<repo root>`** when the model is not pip-installed — `mp.spawn` (spawn
  start-method) workers don't inherit the parent's cwd, so they hit `ModuleNotFoundError`
  without it.
- **Sandbox / seccomp can block NCCL.** If the agent runtime sandboxes syscalls, the
  AF_UNIX sockets used by the NCCL / process-group bootstrap may be blocked → a silent
  deadlock; run multi-rank jobs with the sandbox disabled (or the runtime's escape hatch).
  Record the working invocation (interpreter, env vars, sandbox flag) so every downstream
  multi-rank run reuses it.

## Resource prerequisites (info-only, model-agnostic) — R4 / R10

fold-cp's skills are **model-agnostic**: they do **not** prescribe a fixed GPU SKU or
count, because a custom model may ship kernels that only run on certain GPUs (custom
CUDA/Triton, an arch-gated attention) — unknowable ahead of time. So the requirement is
**not** "have an H100"; it is a **self-consistent stack**. Treat this as a high-level guide,
not a hard gate — the real gates are the smoke tests (Step 4).

- **Self-consistency (the actual requirement).** The user's **hardware ⇄ their PyTorch
  runtime ⇄ their model** must be mutually compatible: a torch built for the GPUs' arch, an
  NVIDIA driver new enough for that CUDA build, and the target model importable in that
  interpreter. `probe_env.py` gives the authoritative signal — **`torch.cuda.is_available()`
  is `True` only when driver + CUDA-build + GPU-arch agree** — and reports the NVIDIA driver
  vs the CUDA build so a mismatch is legible. The two halves probe_env can't settle on its
  own: **model import** (Step 3 — `python -c "import <model_pkg>"` in the chosen interpreter)
  and **DTensor / `device_mesh` support** (Step 4 smoke tests). The one hard software floor is
  that the runtime supports `torch.distributed.tensor` DTensor + `init_device_mesh` — which is
  exactly what Step 4 proves.
- **Disk:** recommend **≥ ~150 GB free** for checkpoints + model containers + datasets + build
  artifacts (a large model or many checkpoints can need more). `probe_env.py` reports free/total
  and warns under the threshold; it is a recommendation, not a hard stop.
- **Per-step hardware (R10) is runtime-derived, not a fixed table.** GPU **count → testable
  configuration** comes from Step 2 (2D needs a perfect square ≥4); **per-module memory
  budgets** come from `/fold-cp:dtensor_modules` (the O(N/cp0) / O(N²/(cp0·cp1)) backward budget)
  and are measured by `/fold-cp:mem_profile`; **training vs inference** differ only by fwd vs
  fwd+bwd+step, captured by `/fold-cp:benchmark` on the *same* mesh. Record the SKU you actually
  ran on in `cp_infra.md` as provenance — do not prescribe one.

## Step 4 — Run the communication smoke tests

These self-launch with `mp.spawn` (no `torchrun` needed) and must pass before any
CP work. Always `timeout`-wrap and tee the logs (Rule 16):

```bash
WS=<world size from Step 2>
timeout 120 python "${CLAUDE_SKILL_DIR}/scripts/test_collectives.py" --world-size $WS |& tee /tmp/$USER/cp_collectives.log
timeout 120 python "${CLAUDE_SKILL_DIR}/scripts/test_dtensor_smoke.py" --world-size $WS |& tee /tmp/$USER/cp_dtensor.log
```

- `test_collectives.py` — verifies `batch_isend_irecv` (ring P2P), `all_gather`,
  `all_reduce`, and `reduce_scatter` against analytic expected values on every
  rank. A hang here means NCCL/IB misconfiguration — diagnose before proceeding.
- `test_dtensor_smoke.py` — builds a `DeviceMesh`, `distribute_tensor`, checks
  `to_local()` shapes and a `full_tensor()` round-trip for a 2D square-tile
  placement. This proves the DTensor substrate works here.

If a test hangs to the timeout, treat it as a hard failure (likely a deadlock or
transport problem), capture `NCCL_DEBUG=INFO` output, and resolve before moving on.

## Step 5 — No local GPUs: escalate to a cluster (SLURM)

If Step 2 yields fewer than 4 GPUs, CP cannot be tested locally. Then:

1. Ask the user for cluster access details: login/submission host, scheduler
   (SLURM assumed), partition/queue, account, time limit, the module/conda
   activation line, and how the repo is reached on compute nodes (shared FS vs
   rsync). Ask whether the agent may submit jobs on their behalf.
2. Fill [`scripts/slurm_probe.sbatch`](scripts/slurm_probe.sbatch) with those
   values — it requests one node + N GPUs and runs the Step 3–4 probes under
   `srun`. Submit with `sbatch`, then poll `squeue`/`sacct` and read the captured
   log. Do not assume success — read the log.
3. Record the working launch recipe (sbatch header + activation) in
   `docs/cp_infra.md` so downstream skills reuse it for every multi-rank run.

## Step 6 — Record the deliverable

Write `docs/cp_infra.md` with: the GPU inventory, the chosen testable 2D-CP config(s)
and `world_size`, software versions, the **resource prerequisites** (free disk vs the
~150 GB recommendation, NVIDIA driver vs torch's CUDA build, and the
self-consistency verdict from the section above), smoke-test results (paths to the
tee'd logs), and the canonical launch command (local `mp.spawn`/`torchrun` or the
SLURM recipe). Downstream skills read this file to know how to launch tests and
benchmarks.

## Deadlock-hardening (set these once, before any multi-rank run)

Distributed CP runs hang silently without a few guards — bake them into the launch
recipe in `docs/cp_infra.md`:

- **Per-collective timeouts** on `init_process_group` (separate NCCL/cuda vs Gloo/cpu
  `timedelta`s) so a deadlock fails fast instead of hanging the session (also Rule 16).
- **`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`** set *before* `init_process_group`, so a rank
  that errors tears the group down instead of leaving peers blocked.
- **Free-port selection + `MASTER_PORT` propagation**: pick an OS-assigned free port with
  `SO_REUSEADDR` on rank 0 and broadcast it; concurrent worktrees/CI that hardcode a port
  collide and rebind-race.
- **P2P parity ordering** in any ring/transpose (Rule 7) — schedule send/recv by rank
  parity so paired ranks don't both block on send.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** at the memory ceiling to fight
  fragmentation-driven `cudaMalloc` retries / spurious OOMs.

## Output contract

- `docs/cp_infra.md` exists and names the testable topology + world size and the
  exact launch recipe.
- Both smoke tests passed (logs on disk) **or** a SLURM path is established and a
  probe job has been verified to pass.
- If neither local GPUs nor cluster access can be obtained, stop and tell the user
  exactly what is blocking CP development.
