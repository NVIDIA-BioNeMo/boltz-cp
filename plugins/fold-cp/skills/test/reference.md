# test — reference

Helpers are in `$BOLTZ_CP_REPO/src/boltz/testing/utils.py` and
`src/boltz/distributed/testing/utils.py`. Reuse them rather than re-implementing.

## Parity-test skeleton (multi-rank, serial = ground truth)

```python
import datetime, pytest, torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor
# project helpers (mirror boltz.testing.utils):
from myproj.testing.utils import (
    spawn_multiprocessing, seed_by_rank, assert_all_identical,
)

def _worker(rank, world_size, mesh_shape, dtype, port):
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=60),          # fail fast on deadlock
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=("cp0", "cp1"))

    gen = torch.Generator(device="cuda").manual_seed(0)  # identical global inputs
    x_global = torch.randn(N, N, C, generator=gen, dtype=dtype, device="cuda")
    go_global = torch.randn_like(x_global)               # explicit random grad_output

    # --- serial reference (full tensors) ---
    serial = SerialModule(...).cuda().to(dtype)
    sd = serial.state_dict()
    x_s = x_global.clone().requires_grad_(True)
    y_s = serial(x_s); y_s.backward(go_global)

    # --- CP path (sharded DTensors) ---
    cp = CPModule(...).cuda().to(dtype)                  # to(dtype) BEFORE load_state_dict
    cp.load_state_dict(sd)
    placements = [Shard(0), Shard(1)]
    x_d = distribute_tensor(x_global.clone(), mesh, placements).requires_grad_(True)
    go_d = distribute_tensor(go_global, mesh, placements)
    y_d = cp(x_d); y_d.backward(go_d)

    # --- sharding is active (anti-vacuous) ---
    assert y_d.to_local().shape[0] < y_s.shape[0]

    # --- forward parity: reassemble then compare ---
    torch.testing.assert_close(y_d.full_tensor(), y_s)   # default tolerances

    # --- gradient parity: full_tensor for replicated-param grads ---
    torch.testing.assert_close(x_d.grad.full_tensor(), x_s.grad)
    for (ns, ps), (nc, pc) in zip(serial.named_parameters(), cp.named_parameters()):
        assert ns == nc, f"name/order drift: {ns} vs {nc}"
        torch.testing.assert_close(pc.grad.full_tensor(), ps.grad)
        assert ps.grad.abs().sum() > 0                   # non-zero grads

    # --- replicated values identical across ranks ---
    assert_all_identical(y_d.to_local(), dist.group.WORLD)
    dist.destroy_process_group()

@pytest.mark.parametrize("mesh_shape", [(2, 2), (3, 3)])   # one fn, many configs
def test_dtensor_mymodule(mesh_shape):
    ws = mesh_shape[0] * mesh_shape[1]
    spawn_multiprocessing(_worker, ws, mesh_shape, torch.float64, 29570)
```

## Comparison matrix

| Compare | Method | Why |
|---|---|---|
| CP result vs serial | `dt.full_tensor()` then `assert_close` | reassemble the global tensor |
| CP vs CP (same placements) | `dt.to_local()` both sides | zero communication |
| Replicated-param grads | `p.grad.full_tensor()` | grads accrue as `Partial(Sum)`; needs reduce |
| Across-rank identity | `assert_all_identical(local, group)` | catch silent replicate drift |

## Anti-vacuous checklist

- [ ] explicit random `grad_output` (never `.sum().backward()`)
- [ ] `local.shape[sharded_dim] < global.shape[sharded_dim]` asserted (skip for a `1×1` mesh, where local == global)
- [ ] replicated values identical across ranks
- [ ] gradients non-zero and finite
- [ ] fp64 + `assert_close` default tolerances; tolerance derived, not tuned
- [ ] serial attribute name + registration order match (zip `named_parameters`)
- [ ] >=1 adversarial/boundary case (non-power-of-two per-axis 2D mesh such as `3×3`, padded axis, 1-elem shard)
- [ ] test features are non-trivial (realistic ranges; mask not all-on/all-off; coupled features consistent)
- [ ] command `timeout`-wrapped; PG timeout set; log tee'd and inspected
- [ ] parity-green ≠ memory-correct: a transient dtype up-cast inside an `autograd.Function`
      passes parity but inflates peak — cross-check peak memory with `/fold-cp:mem_profile`

## Reusable helpers (src/boltz/testing/utils.py)

| Helper | Use |
|---|---|
| `spawn_multiprocessing(fn, world_size, *args)` | launch ranks |
| `seed_by_rank(rank, seed=42)` | per-rank seeding (offset on resume) |
| `skip_if_cuda_not_avail_or_device_count_less_than_word_size(...)` | guard CUDA tests |
| `assert_tensors_identical`, `assert_all_identical` | cross-rank identity |
| `assert_tensors_close_with_pad`, `repad_tensor` | compare padded tensors |
| `all_gather_tensors_along_dim`, `all_gather_pair_repr_along_dims` | manual reassembly |
| `save_gradients`, `detach_and_clone_tensors` | grad capture |
| `init_module_params_uniform/glorot`, `init_tensors_uniform/normal` | deterministic init |
| `assert_no_percentile_upshift`, `assert_close_statistics` | distributional checks |
| `try_assert_and_collect` | collect multiple failures before raising |

`src/boltz/distributed/testing/utils.py`: `create_atom_to_token_dtensor`,
`setup_mock_training_datamodule_config` — build DTensor feature fixtures.

## Synthesizing random features

When no real data exists, generate each feature from the inventory contract:
right shape, dtype, axis semantics, and a **valid mask** (don't mask everything —
that hides bugs). For atom features, generate a consistent `atom_to_token` index
map so token/atom counts agree. Keep the generator seeded and contained. Use **non-trivial values**
(realistic ranges, not constant/degenerate). **Initialize module weights with
`init_module_params_glorot` / xavier (× small gain), not the model's defaults** — Glorot controls
the fan-in/out statistics that drive most fp64 parity noise, enabling tolerances well below torch's
`1e-7` default (≤`1e-12` in practice); still perturb zero-init "final" projections (Rule 14).

## Property / invariant test skeleton (non-parity): RNG-entropy class

For properties with **no serial value oracle** (RNG entropy, determinism, layout). The
oracle is a structural invariant checked **across ranks**. Real impl:
`tests/distributed/test_dtensor_predict.py::test_rng_entropy`
(`_install_rng_capture`, `_assert_rng_entropy_rule`).

```python
import hashlib, json
from pathlib import Path

def _fp(t):  # stable, dtype-agnostic content fingerprint
    b = t.detach().to(torch.float64).cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(b).hexdigest(), int(t.numel())

def _install_rng_capture(monkeypatch, records):
    import random as _random
    def wrap(obj, attr, src):
        orig = getattr(obj, attr, None)
        if orig is None:
            return
        def w(*a, **k):
            out = orig(*a, **k)
            try:
                fp, n = _fp(out) if isinstance(out, torch.Tensor) else (f"s:{out!r}", 1)
            except Exception:               # capture must never perturb the run
                fp, n = ("ERR", 0)
            records.append({"src": src, "fp": fp, "numel": n})
            return out
        try:
            monkeypatch.setattr(obj, attr, w, raising=False)
        except (TypeError, AttributeError):  # C-extension types (e.g. some np.random.Generator): skip
            pass
    for nm in ("randn", "rand", "randint", "randn_like", "normal", "multinomial"): wrap(torch, nm, "torch")
    for nm in ("randn", "rand", "normal", "standard_normal", "choice", "permutation"): wrap(np.random, nm, "numpy")
    for nm in ("standard_normal", "normal", "random", "integers", "choice", "permutation"): wrap(np.random.Generator, nm, "numpy.Generator")
    for nm in ("random", "randint", "uniform", "gauss", "choice", "sample"): wrap(_random, nm, "random")

def _worker(rank, env, kwargs, capture_dir):
    mp = pytest.MonkeyPatch(); set_env_per_rank(mp, env, rank)
    records = []; _install_rng_capture(mp, records)
    try:
        run_workflow(**kwargs)               # the REAL run_predict / forward
    finally:                                 # save even on error
        (Path(capture_dir) / f"rng_rank{rank}.json").write_text(json.dumps(records))

# parent, AFTER spawn_multiprocessing joins all ranks (per-rank files = test context):
per = {r: json.loads((capture_dir / f"rng_rank{r}.json").read_text()) for r in range(world)}
fps = {r: {x["fp"] for x in recs if x["src"] == "torch" and x["numel"] >= 8} for r, recs in per.items()}
drawing = [r for r, s in fps.items() if s]
assert len(drawing) >= 2                                   # exercise threshold (non-vacuous)
for i in range(len(drawing)):                              # sharded noise must NOT tile
    for j in range(i + 1, len(drawing)):
        assert not (fps[drawing[i]] & fps[drawing[j]]), "tiling: identical sharded noise across ranks"
```

Key points:
- Capture in the **worker**; assert in the **parent after join** — the workflow may have torn
  down the process group, so there is no live PG to `all_gather` on. The per-rank JSON files
  carry the state across the join.
- **Negative control (mandatory):** re-run against the known-bad code (e.g. `git stash` the
  fix) and confirm the assertion FAILS — a property test with no negative control is vacuous.
- Replicate-axis draws are made identical by broadcast (not by RNG), so under a per-rank seed
  offset they simply don't collide; the disjointness check is the load-bearing one. Add a
  "replicated draws agree across ranks" check for sources that feed replicated data.

## Property-test anti-vacuous checklist

- [ ] negative control: test FAILS on the known-bad implementation, PASSES after the fix
- [ ] exercise threshold: `>= N` ranks actually drew sizable samples (numel >= threshold)
- [ ] capture is best-effort (try/except; skip un-patchable entry points); never perturbs the run
- [ ] assert in the parent post-join (per-rank files = test context); no reliance on a live PG
- [ ] timeout-wrap, per-collective PG timeout, tee logs, CUDA-preferred (same as parity)

## Dtype-path unit test (per `autograd.Function`)

End-to-end parity is **vacuous** against a dtype bug inserted and reverted *inside* one
`autograd.Function`: the round-trip cancels in the output, so the test stays green while peak
memory (or a low-precision `send`/`recv` deadlock) is wrong (Rule 19). The dtype path is jointly
determined by `(upstream activation dtype) × (weight dtype) × (autocast on/off) × impl`, so pin
it with a **dedicated unit test parametrized over that grid**:

```python
@pytest.mark.parametrize("act_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("aux_dtype", [torch.bfloat16, torch.float32])   # mask/bias dtype
@pytest.mark.parametrize("autocast",  [True, False])
def test_fn_dtype_path(act_dtype, aux_dtype, autocast): ...
```

Make intermediate dtypes **observable** with a forward-hook profiler that records each module's
output dtype (and a recompute profiler for activation checkpointing). Use forward **pre-hooks**,
not post-hooks: under non-reentrant checkpointing (`use_reentrant=False`) the recompute pass
raises an internal stop-recomputation signal that makes **post-hooks miss the recomputed
modules**. Assert the recorded dtype equals the intended compute dtype at every boundary, and
that no buffer is silently fp32 when the mode is true-low-precision.

## Triton-kernel test trio (parity · edge · register-spill gate)

A custom Triton (or otherwise fused) kernel needs three tests, not one:
1. **parity** vs the serial/eager reference (the usual ~fp64 `assert_close`);
2. **edge cases** — boundary sizes, the masked/padded tail, a single-element tile;
3. **register-spill gate** — dump the kernel PTX (`TRITON_KERNEL_DUMP=1`), run
   `ptxas -v --gpu-name=<arch>` and assert **`0 bytes spill stores, 0 bytes spill loads`**.
   A spilling kernel is correct but silently slow; treat zero-spill as a merge gate.

To run the eager reference and the kernel in the *same* process, **monkeypatch the
has-kernel capability flag** to force each path — do not rely on a CPU fallback.

## Stochastic-output test (distributional, not point-value)

A stochastic output (diffusion samples, any sampled ensemble) has **no per-sample oracle** —
never assert sample-equals-reference. Compare **distributions**: an **energy distance** between
the CP and serial sample sets, plus a **Hungarian-matched** per-item metric (matched lDDT /
RMSD) so a permuted-but-correct ensemble doesn't fail. This is the stochastic cousin of the
property/invariant class above — it *has* an oracle (the serial *distribution*), just not a
point-value one.

## Distributed-test hygiene

- **No global grad/RNG state in `setUp`/module scope.** `torch.set_grad_enabled(False)` or a
  module-level `manual_seed` **leaks into later tests** and only bites under full-suite ordering
  (a single-process run masks it). Use a local `with torch.no_grad():` and
  `torch.random.fork_rng()` / a contained `Generator`.
- **An OOM corrupts the shared CUDA context and cascades** into unrelated downstream failures —
  an OOM-prone parity test is a **must-fix**, not retry-flaky (cap `max_seqs`/MSA depth; mind the
  recompute peak, Rule 11).
- **Classify a failing distributed test in a fresh process.** A large combined sweep is mostly
  cross-test pollution; before trusting a failure, re-run that one parametrization in a clean
  process (and a clean CUDA context).
- **Log `grad_norm` in `on_after_backward`, not before backward** — pre-backward it is
  identically 0, making a "grad > 0" assertion vacuous.
- **A green source-branch suite does not catch a stale-base overwrite.** When a change
  incidentally rewrites a file from an older base, diff every incidentally-touched file against
  the *target* branch and run the **dependent** tests (those importing the clobbered symbols),
  not just the branch's own suite.
