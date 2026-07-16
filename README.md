# Fold-CP: A Context Parallelism Framework for Biomolecular Modeling

Context parallelism (CP) for distributed inference and training for
biomolecular folding models across multiple GPUs using a 2D CP mesh combined
with data parallelism, demonstrated with the Boltz model.

⚠️ **Note**<br>
This repository demonstrates a proof-of-concept implementation of Fold-CP with Boltz-2. <br>
We are actively working to upstream this context parallelism capability to the official Boltz source code. You can view the [Draft PR here](https://github.com/jwohlwend/boltz/pull/658). <br>
Learn more about Fold-CP here: https://research.nvidia.com/labs/dbr/assets/data/manuscripts/fold_cp.pdf

For an introduction to the Boltz family of biomolecular interaction models,
see the [public Boltz repository](https://github.com/jwohlwend/boltz).

## Copyright and License Compliance

- The context parallel code is licensed under the terms and conditions as written in [the license file](licenses/LICENSE)

- The original Boltz code is licensed under their respective MIT license (See the [third-party-attr.txt](licenses/third-party-attr.txt))

- This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use

## Key Capabilities

- **Distributed inference** with DTensor context parallelism
- **Distributed training** with DTensor context parallelism
- Combined data parallelism (DP) and context parallelism (CP)
- Multiple attention kernel backends: cuEquivariance, trifast, FlexAttention
- Support for BF16, BF16-mixed, TF32, and FP32 precision modes

## Requirements

- Python 3.10+
- PyTorch 2.9+ with CUDA support
- Multiple NVIDIA GPUs (CP requires at least 4 GPUs; CP size must be a
  perfect square)
- `torchrun` or SLURM `srun` for multi-process launching

## Distributed Inference

Distributed inference uses `src/boltz/distributed/main.py predict` to run
structure prediction with DTensor context parallelism.

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 4 \
  src/boltz/distributed/main.py predict \
  /path/to/preprocessed_data \
  --out_dir ./predictions \
  --size_dp 1 \
  --size_cp 4 \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 5
```

For full documentation of all options, the inference pipeline stages, and
differences from serial prediction, see the
[Distributed Inference Guide](docs/boltz2_cp_prediction.md).

## Distributed Training

Distributed training uses `src/boltz/distributed/train.py` with a YAML
config file to run training with DTensor context parallelism.

```bash
torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  src/boltz/distributed/train.py \
  scripts/train/configs/structurev2_small_cp.yaml \
  parallel_size.size_dp=2 \
  parallel_size.size_cp=4 \
  output=<output_dir>
```

For full documentation of the configuration hierarchy, CP-specific settings,
CLI overrides, and differences from serial training, see the
[Distributed Training Guide](docs/boltz2_cp_training.md).

## Integrate Fold-CP Into Your Model Using Agentic Workflow

The sections above run context parallelism on the **demonstrated Boltz model**. If
you want to add the same DTensor-based CP to **your own** co-folding /
structure-prediction model, this repository also ships **fold-cp**, an agentic plugin
that guides the integration end to end and uses this repository's CP code as its
reference implementation. It works with **both** [Claude Code](https://code.claude.com)
and the [OpenAI Codex CLI](https://developers.openai.com/codex) — the same skills, hard
rules, and CP guide ship for both runtimes.

fold-cp is project/model-agnostic. It contributes a set of CP **hard-rules** (injected
every session via a SessionStart hook, plus an opt-in guard that protects your serial
ground-truth files), **11 skills** that cover the integration from exploration through
profiling, and a bundled high-level **CP technical guide**. A top-level *conductor*
skill sequences and gates the whole effort and delegates each phase to the focused
skills; you can also invoke any skill on its own. Once the plugin is installed and
enabled, invoke a skill as **`/fold-cp:<skill>`** in Claude Code, or as **`$<skill>`**
(type `$`, or run `/skills` to browse) in Codex.

### Step 1 — Add the marketplace and install the plugin

**Claude Code.** Register this repository as a plugin marketplace, then refresh its catalog:

```
/plugin marketplace add https://github.com/NVIDIA-BioNeMo/boltz-cp
/plugin marketplace update boltz-cp
```

(You can also point `marketplace add` at a local clone path instead of the URL.)
Then install the plugin and choose an **install scope**. Interactively, open the
plugin manager:

```
/plugin
```

Go to the **Discover** tab, select **fold-cp**, press Enter, and pick a scope —
**User** (all your projects), **Project** (shared with collaborators on this repo), or
**Local** (this repo, only you). Or install non-interactively:

```
/plugin install fold-cp@boltz-cp --scope user      # or: project | local
```

Activate it in the current session:

```
/reload-plugins
```

After this, the CP hard-rules are injected each session and the `/fold-cp:*` skills are
available.

**Codex CLI.** Add the same repository as a Codex marketplace (GitHub shorthand, a Git
URL, or a local clone path all work):

```
codex plugin marketplace add NVIDIA-BioNeMo/boltz-cp
```

Then open the plugin directory, switch to the **boltz-cp** marketplace, select **fold-cp**,
and install it:

```
/plugins
```

Codex reads the repo's `.agents/plugins/marketplace.json` catalog and caches the plugin
under `~/.codex/plugins/cache/`. The SessionStart hook then injects the CP hard-rules and
the skills become available as `$<skill>`. (To disable it later, set `enabled = false`
under `[plugins."fold-cp@boltz-cp"]` in `config.toml`.)

### Step 2 — Drive the whole integration with the conductor

`cpize_model_workflow` is the top-level entry point. It maps your model, verifies
infrastructure, shards the data features, ports the DTensor modules, proves
serial-vs-CP numerical parity, wires the trainer/predictor lifecycle, and
profiles/benchmarks — phase by phase, gating on each.

```
/fold-cp:cpize_model_workflow training trunk dp:1 cp:(2,2) model:~/code/myfold ref:~/code/boltz-cp --local --random --automatic
```

In **Codex**, invoke the same conductor with the `$` prefix and identical arguments:

```
$cpize_model_workflow training trunk dp:1 cp:(2,2) model:~/code/myfold ref:~/code/boltz-cp --local --random --automatic
```

Arguments are parsed **by keyword and are order-independent**; every one is optional and
has a default:

| token | argument | meaning |
|---|---|---|
| `training` | **scope** | which workflow to CP-ify: `inference` / `training` / `all` (default `all`). |
| `trunk` | **focus** | subsystem to integrate **first** — an ordering seed, not a filter (the whole model stays in scope). e.g. `trunk` / `diffusion` / `confidence` / `data`. Absent ⇒ data pipeline first. |
| `dp:1 cp:(2,2)` | **mesh** | device mesh used for every parity test, benchmark, and profile: `dp` data-parallel replicas × a `cp0×cp1` CP grid. `cp:(2,2)` ⇒ 2D-CP, `world_size = 4`. Absent ⇒ chosen by `build_infra`. |
| `model:~/code/myfold` | **model path** | your serial model code to CP-ify. Absent ⇒ resolved by tracing. |
| `ref:~/code/boltz-cp` | **reference repo** | the CP reference implementation every mapping cites (this repo). Absent ⇒ `$BOLTZ_CP_REPO` env var / filesystem search. |
| `--local` | **launch env** | run multi-rank jobs on local GPUs; `--slurm` uses a cluster. |
| `--random` | **data source** | synthesize random features; `data:<path>` uses real inputs. |
| `--automatic` | **execution mode** | how the conductor paces itself — **automatic is the default**: pass *neither* flag and it runs the entire plan unattended (see below). `--automatic` only states that default explicitly; `--manual-approve` instead pauses at each wave boundary. |
| `resume` | **resume** | continue an interrupted run from the plan ledger (`docs/cp_integration_plan.md`). |

With **`--automatic`** the conductor runs every phase continuously, wave after wave,
through the final report — **fixing a failing implementation itself** (a red parity test
means the implementation is wrong, never the test), resolving design forks by sensible
defaults, and halting only on a true blocker (no usable GPU, an unsatisfiable mesh, a
missing reference repo). Because automatic is the default, **`--automatic` is optional** —
the minimal run below behaves identically whether or not you pass it:

```
/fold-cp:cpize_model_workflow model:~/code/myfold ref:~/code/boltz-cp --automatic
```

### Step 3 — Or run individual skills

The conductor invokes the skills below; you can also run any of them directly. Each is
called as **`/fold-cp:<skill> <args>`** in Claude Code or **`$<skill> <args>`** in Codex
(same arguments either way, optional and order-tolerant).

- **`learn_context`** — explore your model and map it onto the CP reference.
  `[scope: inference|training|data|all] [path to model code]`
  ```
  /fold-cp:learn_context all ~/code/myfold
  ```
- **`build_infra`** — probe GPUs / software stack and run distributed smoke tests.
  `[--local | --slurm] [requested world size]`
  ```
  /fold-cp:build_infra --local 4
  ```
- **`shard_data_feats`** — assign DTensor placements and shard the data features.
  `[2d] [feature group: atom|token|msa|pair|all]`
  ```
  /fold-cp:shard_data_feats 2d all
  ```
- **`dtensor_modules`** — port a serial layer/module to a DTensor CP module.
  `[module name] [2d]`
  ```
  /fold-cp:dtensor_modules TriangleMultiplication 2d
  ```
- **`test`** — write/run multi-rank parity tests (CP vs serial ground truth).
  `[source file under test] [unit|layer|module|workflow] [2d]`
  ```
  /fold-cp:test src/boltz/distributed/model/trimul_cp.py module 2d
  ```
- **`dist_lifecycle`** — wire the distributed training/inference lifecycle.
  `[wrap | checkpoint | resume | ema | optimizer]`
  ```
  /fold-cp:dist_lifecycle checkpoint
  ```
- **`dispatch_work`** — orchestrate a coder+reviewer agent team to port many modules in
  parallel. `[module/test scope, or 'all new modules in current_code_structure.md']`
  ```
  /fold-cp:dispatch_work all new modules in current_code_structure.md
  ```
- **`benchmark`** — find the max token count at a CP size and record walltime.
  `[inference | training] [data path | --random] [cp size]`
  ```
  /fold-cp:benchmark inference --random 4
  ```
- **`nsys_profile`** — profile with NVIDIA Nsight Systems.
  `[e2e | trunk] [N_token] [cp size]`
  ```
  /fold-cp:nsys_profile e2e 512 4
  ```
- **`mem_profile`** — attribute the top-N memory peaks to modules/lines via the PyTorch
  allocator history. `[e2e | trunk] [N_token] [cp size]`
  ```
  /fold-cp:mem_profile trunk 768 4
  ```

## Contributing

This project is currently not accepting contributions.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting instructions.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
