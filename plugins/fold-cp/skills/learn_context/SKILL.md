---
name: learn_context
description: >
  Explore a custom co-folding / structure-prediction model to understand its
  inference and training workflows, entry points, training framework (PyTorch
  Lightning vs DeepSpeed), data-feature format, and featurization pipeline — then
  map those components onto the existing Boltz-CP context-parallel reference so
  downstream CP integration reuses what already exists. Produces
  docs/current_code_structure.md. Use FIRST, before any other fold-cp skill, or
  whenever that doc is missing/stale. Runs interactively (Q&A with the user) or
  autonomously on request.
argument-hint: "[scope: inference|training|data|all] [optional path to model code]"
---

# learn_context — map the user's model and bind it to the Boltz-CP reference

Every other fold-cp skill reads `docs/current_code_structure.md`. This skill
produces it. The user's model is almost never shaped like the Boltz serial code,
so do not assume Boltz layouts — **trace the actual code from its entry points**
and record what you find.

The deliverable answers one question: **which parts of this model can reuse an
existing Boltz-CP component, and which need new work?**

## Step 0 — Locate the Boltz-CP reference repo (required)

The reference implementation and its docs are the ground truth for all mapping.
Resolve `BOLTZ_CP_REPO` in this order, then record the absolute path:

1. `$BOLTZ_CP_REPO` env var, if set.
2. A path the user gives when asked.
3. Search: `fd -t d -d 6 'boltz' ~ 2>/dev/null` or look for a dir containing
   `src/boltz/distributed/` (the CP package).

The skills' primary CP reference is **bundled in this plugin**:

- **`${CLAUDE_PLUGIN_ROOT}/reference/cp_tech_guide.md`** — per-module CP algorithms, I/O
  shapes, placements, collectives, complexity (11 sections; see its Table of Contents).
  Always present — no need to locate it.

The reference repo provides the **distributed code to mirror**
(`$BOLTZ_CP_REPO/src/boltz/distributed/`); the governing CP rules are this plugin's
`hooks/RULES.md` (injected at SessionStart), not a file in the reference repo.

**Substrate-boundary scan (do once, up front).** Grep the reference's distributed tree
for imports of its OWN non-distributed package, e.g.
`grep -rE "from boltz\.(model|data|main)" $BOLTZ_CP_REPO/src/boltz/distributed`. This
splits it into the **app-agnostic substrate** (manager, comm, DTensor op-layers — files
with no such imports, vendorable as-is by a namespace rewrite) and the **app-coupled**
files (CP layers that import their serial twin — these get rewired to YOUR serial model,
not vendored). Recording this boundary now prevents rediscovering it per-module later
(it directly seeds the reuse/adapt/new column of §4).

If none can be found, ask the user for the repo location before continuing — the
mapping cannot be done without it.

## Step 1 — Choose mode and scope

- **Scope** comes from `$ARGUMENTS` (`inference`, `training`, `data`, `all`, or a
  path). Default `all`.
- **Mode:** if the user asked to "just explore" / gave a path, run **autonomously**
  (read-only tracing). Otherwise run **interactively** and use `AskUserQuestion`
  for anything you cannot determine from the code (see Step 6). Prefer answering
  from the code first; ask only what tracing cannot resolve.

## Step 2 — Trace the inference workflow

Find the inference entry point and follow the call chain to `model.forward`.
Look for: a CLI (`argparse`/`click`/`typer`), a `predict`/`infer`/`sample`
function, a Lightning `predict_step`, or a notebook. Record:

- entry file + function, the config/CLI surface, and the **ordered call chain**
  from entry → data load → featurize → `forward` → output writing.
- where structure coordinates are produced (diffusion/sampling head) and any
  recycling/sampling loops (these carry control-flow scalars — Rule 7).

Reference comparison: Boltz inference enters at
`$BOLTZ_CP_REPO/src/boltz/distributed/main.py` (`cli`, `predict`) →
`predict.py` (`run_predict`).

## Step 3 — Trace the training workflow and detect the framework

Find the training entry point and **identify the framework** — this changes how
CP wraps the model:

- **PyTorch Lightning:** a `LightningModule` (`training_step`, `configure_optimizers`),
  a `Trainer`, a `LightningDataModule`, a custom `Strategy`. Boltz-CP uses a custom
  DTensor strategy — see `$BOLTZ_CP_REPO/src/boltz/distributed/lightning_strategy.py`
  and `train.py` (`train`, `_create_dist_manager`, `_create_distributed_model`,
  `_create_distributed_data_module`).
- **DeepSpeed:** a `deepspeed.initialize`, ZeRO config JSON, `engine.backward`/`step`.
- **Raw `torchrun` / custom loop:** an explicit `dist.init_process_group` + manual
  optimizer loop.

Record: entry file + function, framework + version, optimizer/scheduler, EMA,
checkpoint format and save/load paths, and how the device/process group is created
today (if at all).

## Step 4 — Build the data-feature inventory

This is the highest-value section. Trace dataset → featurizer → collate →
dataloader → datamodule → the dict that reaches `forward`. For **every feature**
the model consumes, record a row: name, tensor shape, dtype, and **axis
semantics** — which axis is tokens (N), atoms (N_atoms), MSA depth (S),
ensemble (E), or a pair axis `[N, N, …]`. Axis semantics drive every sharding
decision and padding rule downstream, so resolve `N_tokens == N_atoms`
ambiguities explicitly (ask if the code is silent).

Note local-vs-global index features and mask/padding conventions — these must stay
consistent between data and model (Rule on index/mask consistency).

## Step 5 — Map modules: serial → Boltz-CP candidate

Identify the model's major blocks and match each to a Boltz-CP component using the
tech guide. Typical correspondences (confirm against the actual code, names will
differ):

- Pairformer/Evoformer trunk → triangle attention/mult, OPM, PWA, transition,
  attention-pair-bias (tech guide §3–§8).
- Atom encoder / window attention → window batching + distributed gather (§9).
- Diffusion / structure head → diffusion module + conditioning.
- Confidence head, losses (distogram, pLDDT, PDE, smooth-LDDT) → §10–§11.

Produce a mapping table: serial block → candidate 2D CP file(s) → tech
guide § → "reuse as-is / adapt / new". Flag any block with **no** CP counterpart
as new work for `dtensor_modules`.

**Then propagate the data-feature sharding requirements to consumers (the Rule-4 contract) — do not
leave §3 and §5 unlinked.** §3 gives each feature its placement; §5 gives each module. Cross-link them:
for **every module**, list the **§3 data features it consumes** and the **placement each must arrive
at**; and annotate **every §3 feature** with the modules that consume it. This data-feature→consumer
placement contract is the single sharding requirement that **all** consuming CP modules **and** the
synthetic test data/featurizer must honor — deriving it **once** here prevents each module port from
re-reconciling the same seam later (a shared feature like MSA, consumed by OPM + PWA + the MSA module,
is the classic case: propagate `(Shard(0)=depth, Shard(1)=tokens)` once instead of three times). Record
it as **§4.5 of the deliverable** (the placement-contract table in [template.md](template.md)).

## Step 6 — Resolve gaps with the user (interactive mode)

Use `AskUserQuestion` only for what tracing cannot settle. High-value questions:

- Which components does the user want CP'd first (scope the effort)?
- Where is the **serial ground truth** (the read-only reference paths)?
- What perfect-square 2D CP mesh should be targeted (minimum 4 GPUs)?
- Is real inference/training data available for parity tests, or should tests use
  synthesized random features?

## Step 7 — Write the deliverable

1. Write `docs/current_code_structure.md` from
   [template.md](template.md) — fill every section; leave explicit
   `TODO(learn_context): <question>` markers for unresolved gaps rather than
   guessing.
2. If the user designated serial-reference paths, offer to write
   `.fold-cp/config.json` to arm the PreToolUse guard (Rules 2 & 16):
   ```json
   {
     "enforce_serial_protection": true,
     "serial_paths": ["<glob>", "..."],
     "enforce_test_timeout": true,
     "distributed_test_markers": ["tests/distributed"]
   }
   ```

## Output contract

- `docs/current_code_structure.md` exists, every template section is filled or
  TODO-marked, and it records: `BOLTZ_CP_REPO` path, inference + training
  workflows, framework, the data-feature inventory table, the serial→CP module
  map, designated serial paths, and target CP topology.
- Report a one-paragraph summary: what reuses existing CP components vs what is new
  work, and the recommended next skill (`build_infra` if infra is unverified, else
  `shard_data_feats`).
