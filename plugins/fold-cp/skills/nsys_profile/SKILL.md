---
name: nsys_profile
description: >
  Profile a context-parallel (CP) inference (or training) workflow with NVIDIA
  Nsight Systems (nsys). Locates or installs the nsys binary, mocks random
  features + small Glorot-init weights (or loads a real checkpoint), wraps the
  end-to-end forward in a torchrun launcher, runs nsys with the right CLI options
  (cuda/nvtx trace, --pytorch functions-trace/autograd, cudabacktrace,
  python-backtrace), and auto-annotates high-level module names (TriMul,
  Pairformer, trunk, diffusion, MSA) via a custom --python-functions-trace JSON —
  then reads the kernel/NVTX breakdown (Cannon-ring SendRecv vs GEMM, trunk vs
  diffusion vs confidence). Use once a CP workflow runs end-to-end and you need to
  see where GPU time and communication go.
argument-hint: "[e2e | trunk] [N_token] [cp size]"
---

# nsys_profile — Nsight Systems profiling for the CP workflow

Profiles where a CP workflow spends GPU time and communication, with the module
hierarchy named (trunk / TriMul / Pairformer / diffusion / confidence) so the
timeline is readable. The five load-bearing steps below are where people get
stuck; follow them in order.

## Preconditions
- A CP workflow that runs end-to-end (forward for inference), verified by
  `/fold-cp:test`; `docs/cp_infra.md` for topology + `world_size`.
- `nvidia-smi` available; multi-GPU node (CP grid). **Before claiming GPUs, probe occupancy**
  (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`) and **HOLD if the target
  devices are busy — never oversubscribe (one job per slot)**; the `manage_gpu` plugin automates
  this claim/hygiene if installed.

## Step 1 — Find or install nsys (don't trust the system copy)
`nsys` usually ships inside the **cuda-toolkit / nsight-systems conda package**,
and Nsight Compute installs bundle a matching `nsys`:
```bash
which nsys; find "$CONDA_PREFIX" -name nsys -type f 2>/dev/null   # env-local
find ~/miniconda3 ~/anaconda3 -name nsys -type f 2>/dev/null | head # any env
nsys --version    # and the bundled one's --version
```
Pick a version **new enough for the GPU arch** — a stale `/usr/bin/nsys` (e.g.
2022.x) will not recognize Blackwell (sm_120); prefer the conda 2024.5+/2025.x
binary. If none exists: `conda install -c nvidia nsight-systems` (or
`cuda-nsight-systems`) into the env, or `pip install nvidia-nsight-systems`.
Record the absolute path you use.

## Step 2 — Mock inputs + weights (or load the real checkpoint)
Two regimes:
- **Synthetic (fast, scale sweeps):** generate random features per the inventory
  contract (`current_code_structure.md` §3) — correct shapes/dtypes/axis
  semantics, valid masks, consistent `atom_to_token`. Initialize weights with
  **Glorot-uniform × small gain** (e.g. 0.1) so deep stacks stay numerically tame
  (also what `/fold-cp:test` parity uses). Good for the CP-relevant subset (e.g.
  the trunk) where you control the inputs.
- **Real checkpoint (full e2e):** random weights make the **diffusion sampler /
  Kabsch-SVD / confidence** produce NaNs — load the real checkpoint instead
  (`from_pretrained(<local_dir>, load_esmc=False)` to skip the frozen LM) and
  drive the real inference entry point (e.g. `model.infer_protein(seq)` /
  `processor.fold`) with a random sequence so featurization is real and valid.
Pick **N_token** so the full default workflow fits — the diffusion pair-bias
`[samples, heads, N, N]` and a deep trunk OOM at large N; a realistic length
(256–1024) runs the default sampler, large N is for the trunk-only sweep.

## Step 3 — Identify the e2e workflow and build a torchrun wrapper
Trace the real inference call to enumerate every stage (front-end → recycling
trunk → heads → diffusion → confidence) and which parts are CP-distributed vs
serial. Then wrap **one process per CP rank with torchrun** (don't use
`mp.spawn` for the profiled run — one launcher, clean process tree for nsys):
```python
# module run under torchrun: torchrun sets RANK/LOCAL_RANK/WORLD_SIZE/MASTER_*
def run(args):
    rank = int(os.environ["RANK"])
    torch.manual_seed(args.seed)              # identical RNG on every rank
    init_distributed(...)                      # DistributedManager / init_process_group (ENV reads torchrun env)
    model = build_or_load(...).to(device).eval()
    wrap_model_with_cp(...)                     # CP-distribute trunks (transparent: full-tensor in/out)
    with torch.no_grad():
        run_e2e_forward(model, inputs)          # the REAL workflow
```
Rules that matter (fold-cp Rule 7): **seed all ranks identically** and broadcast
control-flow counts (recycles, sampling steps) so every rank issues the same
collectives in the same order — divergent control flow deadlocks NCCL. The
launcher and CP-wrap live on your side, not in the upstream model.

## Step 4 — Run nsys with the right CLI options
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nsys profile -o profiling/results/inference/<name> --export sqlite \
  --trace cuda,nvtx,osrt --force-overwrite true --cuda-memory-usage true \
  --pytorch=autograd-shapes-nvtx \
  --python-functions-trace=<custom.json> \   # Step 5; gives module names
  --cudabacktrace=all --python-backtrace=cuda \
  torchrun --standalone --nnodes=1 --nproc_per_node=<cp> \
    -m <your.profile_module> --n-token <N> --cp0 <a> --cp1 <a>
nsys stats --report cuda_gpu_kern_sum --report nvtx_gpu_proj_sum <name>.nsys-rep
```
Know what each does and its caveats (these bite):
- `--trace cuda,nvtx,osrt` — kernels + NVTX domains + OS runtime. nsys traces
  `torchrun` **and** its spawned ranks (one report, all ranks).
- `--pytorch=autograd-nvtx | autograd-shapes-nvtx` — wraps the run in
  `emit_nvtx()`. Annotates **autograd** ops only → a **no-op for the parts under
  `torch.no_grad()`**, EXCEPT custom `autograd.Function`s (e.g. a distributed
  TriMul) still get ranges (with shapes under `-shapes-`). `functions-trace-shapes`
  is **not** a valid token; combine `autograd-shapes-nvtx` with `functions-trace`.
- `--python-functions-trace=<json>` — Python-function tracing (needs `pip install
  nvtx`); works under `no_grad`. This is the one that names modules (Step 5).
- `--cudabacktrace=all` / `--python-backtrace=cuda` — CPU/Python backtraces for
  CUDA calls. **Require CPU sampling**; in restricted containers nsys warns "CPU
  IP/backtrace sampling not supported, disabling" and they are inert (need
  `perf_event_paranoid` ≤ 2 / privileges).
- The big `.nsys-rep`/`.sqlite` are regenerable — **gitignore them**.

## Step 5 — Build a custom pytorch function-trace JSON (key for module names)
`--pytorch=functions-trace` uses the shipped `pytorch.json`, which annotates only
**`torch.*`** (generic `Module.__call__`, `Linear.forward`, …) — it will **not**
name your model's classes (TriMul / Pairformer / trunk / diffusion). To get those
without any in-model NVTX, pass a **custom JSON** that lists your module forwards,
same schema as the shipped file:
```json
[ { "domain": "MyModel", "module": "pkg.path.to.module",
    "functions": ["TriangleMultiplicativeBlockDistributed.forward",
                  "FoldingTrunkDistributed.forward", "DiffusionModule.forward"] } ]
```
Build it (merging the torch.* entries so you keep `F.linear` etc.) with
[`scripts/make_pytorch_functions_trace.py`](scripts/make_pytorch_functions_trace.py)
— it imports each `module`, validates `Class.method` exists, and appends. Notes:
- `module` is the **import path where the class is defined** and importable;
  `functions` are `ClassName.method` (`forward`, or any method like `sample`).
- If the model uses absolute imports under a vendored root (e.g.
  `projects.x.y...`), install that namespace alias before importing so the class
  objects resolve (and `isinstance` identity holds).
- Pass via `--python-functions-trace=<merged.json>` (not `--pytorch=functions-trace`)
  and keep `--pytorch=autograd-shapes-nvtx` for the autograd-Function shapes.

## Step 6 — Read the breakdown
```bash
nsys stats --report cuda_gpu_kern_sum --format table <rep>   # kernels by GPU time
nsys stats --report nvtx_gpu_proj_sum --format table <rep>   # module/op ranges (GPU-projected)
```
Look for: `ncclDevKernel_SendRecv` (Cannon-ring p2p) vs `cutlass …gemm` (local
matmul) — comm vs compute; `AllGather` (full-tensor gather at CP boundaries); the
per-module ranges (`*.forward`) to attribute time to trunk vs diffusion vs
confidence. Cross-check the interconnect (`nvidia-smi topo -m`): a no-NVLink
(PCIe) box makes the ring dominate; NVLink shifts it toward compute-bound.

## Discipline
- `timeout`-wrap every run; set a process-group timeout; tee logs under `/tmp/$USER/`.
- Probe feasibility first (1 loop / few diffusion steps, small N) to get peak
  memory + per-iter time before the full capture; `expandable_segments:True`
  fights fragmentation.
- The launcher change (mp.spawn→torchrun) and annotation must not alter the GPU
  work — sanity-check peak memory and headline kernel mix match a prior run.

## Output contract
- `profiling/results/.../<name>.{nsys-rep,sqlite}` (gitignored) + the custom
  functions-trace JSON committed.
- A short readout: per-module time split (CP vs serial), comm-vs-compute share,
  peak memory/rank, the nsys binary + option string used, and any inert options
  (with the reason, e.g. CPU sampling unsupported).
