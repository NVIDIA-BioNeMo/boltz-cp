# fold-cp — hard rules (always in effect when this plugin is enabled)

These are the non-negotiable invariants for integrating context parallelism (CP)
into a co-folding / structure-prediction model with the `fold-cp` skills — the
high-level requirements that gate every code change and the agentic workflow, and
the governing CP rules for the Boltz-CP framework (this file is that authority; it
is not a copy of any reference-repo doc). When a task touches CP code, data, or
tests, obey these even if the local project has no `CLAUDE.md` of its own.

## Agentic workflow (how to work)

1. **Explore before integrating.** Every fold-cp task begins from
   `docs/current_code_structure.md` (the `/fold-cp:learn_context` output) and the
   located Boltz-CP reference repo plus this plugin's bundled CP technical guide
   (`reference/cp_tech_guide.md`). If `current_code_structure.md` is
   missing or stale, run `learn_context` first. Never write CP code against an
   un-mapped model.
2. **Serial code is ground truth — never edit it for distributed purposes.** Treat
   the user's reference (non-distributed) model and data code as read-only. When
   divergence is needed, copy the file into a distributed subtree and change the
   copy. **Two narrow, *benign* exceptions are sanctioned** — both leave the serial
   model's production behaviour bit-identical, and both are used by the Boltz-CP
   reference tree itself:
   - **dtype-parametrising fix in serial:** replace a hardcoded down-cast
     (`.float()` / `.to(torch.float32)`) with
     `torch.promote_types(<serial-context>.dtype, torch.float32)`, so the op runs at
     `max(input-dtype, fp32)`. This is a **no-op on the production path** (fp32/bf16
     inputs still compute in fp32) but lets **fp64 inputs flow through as fp64** — which
     is what makes exact fp64 CP==serial parity tests possible. The Boltz-CP serial tree
     applies this widely (e.g. `model/loss/diffusionv2.py`,
     `model/modules/{transformersv2,encodersv2}.py`, `model/layers/triangular_mult.py`).
     Keep the edit *purely* dtype — never change shapes, control flow, or production
     numerics.
   - **test-scoped monkeypatch:** a test may temporarily patch a serial fp32 down-cast
     (or swap a kernel for its reference) to enable fp64, **inside a context manager /
     fixture that reverts on exit** so the patch never persists beyond the test process.
   No other serial edit is permitted. The opt-in PreToolUse guard blocks serial writes by
   default; set `allow_serial_dtype_promotion: true` in `.fold-cp/config.json` to let the
   sanctioned `promote_types` edit through (the monkeypatch path needs nothing — test
   files are not serial paths).
3. **When a parity test fails, assume the implementation is wrong.** Diagnose the
   distributed code; do not weaken assertions, loosen tolerances, or delete checks
   to make a test pass.
4. **Reconcile up/downstream contracts.** When integrating coupled layers, the
   producer's output placements must equal the consumer's expected input
   placements. Verify both sides against `docs/current_code_structure.md` before
   wiring them together.
5. **Batch shell commands; run pre-commit after edits.** Minimise approval
   round-trips; after editing, run the project's `pre-commit` (or linters) on the
   changed files and fix until clean. At orchestration scale, Rule 24 generalises this
   to delegated work (batch independent launches, collect results together).

## CP correctness (what the code must guarantee)

> **The constraint behind Rules 8–9 (and the data-side padding).** `torch.distributed`
> P2P (`send`/`recv`/`batch_isend_irecv`) and collectives (`all_gather`/`all_reduce`/
> `reduce_scatter`/ring P2P) require **every rank's communicated buffer to have the
> same shape and dtype** — the receiver pre-sizes its buffer, so a mismatch hangs NCCL
> or corrupts memory. Three corollaries follow: shards must be **equal-size** across
> ranks (Rule 8); features are **padded to a multiple of the mesh size before
> distribution** so the sharded axis splits evenly (`/fold-cp:shard_data_feats`); and a
> feature with two coupled semantic axes (atoms×tokens, token×token) must keep those
> axes **co-sharded / co-replicated** so per-rank tiles line up for the algorithm that
> consumes it (Rule 8). **Dtype is as load-bearing as shape here:** if a communicated
> buffer's dtype depends on a per-rank branch or autocast state inside an
> `autograd.Function`, two ranks can enter a `send`/`recv` with mismatched dtypes and
> **deadlock** — pin the buffer to one agreed dtype before the collective (Rule 19).

> **Single-device RNG *entropy* equivalence (the rule for every random draw under CP).**
> A distributed random draw must match the *entropy structure* of the single-device
> reference — which axes carry **independent** randomness vs **shared** randomness — but
> **not** the numeric sample values. A distributed RNG that numerically reproduces an
> all-gathered single-device draw (gather → serial RNG → re-shard, bit-for-bit) **does not
> exist today**; do not assert or rely on shard-wise samples being numerically identical to
> that single-device draw — condition any such goal on that not-yet-available capability.
> What we *can* hold to is the entropy: for each draw, audit **every device-mesh axis (and
> the global rank)** and classify it:
> - **Shared across the axis** — ranks holding replicated / co-replicated data, or a scalar
>   consumed by the whole group → must be **identical**, achieved by generating once and
>   **broadcasting** (`create_and_broadcast_tensor_into_placements`), not by hoping RNG
>   states agree.
> - **Independent across the axis** — ranks owning distinct shards → must draw **different**
>   entropy; the single-device reference does *not* expect these shards to be identical.
>
> **The bug to catch:** sharded ranks drawing **identical** samples where the single-device
> reference would not (e.g. the current coord-noise generation under a shared inference
> seed — every atom shard redraws the same `torch.randn`, a tiled field).
>
> **The fix is per-axis.** Replicate axes stay correct via the broadcast above. For the
> independent axes the practical fix is a **per-rank seed offset** so each rank's RNG stream
> diverges — `seed_everything(seed + dist_manager.group_rank["world"])` (training already
> does this; inference must too). This delivers the *entropy* equivalence we can guarantee;
> it does **not** make the shards numerically match a single-device draw (that needs the
> not-yet-existing consistent distributed RNG), and that is fine — entropy is what the rule
> requires. **Audit per draw and per axis** — different axes in the same draw need different
> treatment (DP axis → independent per input; CP-shard axis → independent; replicate axis →
> broadcast), and a global-RNG seed offset must not silently desync a draw that feeds
> replicated data (verify those still agree).

6. **No implicit DTensor ops on differentiable paths.** Native DTensor dispatch
   (`a + b`, `einsum`, `matmul`, `reshape`/`repeat_interleave`/`squeeze` on a
   sharded axis) can hide all-gathers. Use `torch.autograd.Function` with explicit
   collectives, or the project's composable DTensor utilities. Inside such a
   `Function`: never read `ctx.needs_input_grad` in `forward` (it can hang NCCL — use
   `tensor.requires_grad`); make tensors `.contiguous()` **before every collective**;
   in `backward` reverse the forward send/recv schedule and handle a rank's self-send
   by direct copy, not a P2P op. **Distributed signatures accept only DTensor for
   distributed features/activations** — validate placements *and* shapes at the boundary;
   never pass a "naked" plain tensor on the distributed path (it hides wrong
   placements/strides and silently mis-shards).
7. **All ranks execute collectives in identical order.** Sort dict keys before
   collective loops. In ring/transpose P2P, schedule send/recv by **rank parity**
   (`ternary_parity`) so paired ranks don't both block on send — circular waits
   deadlock. Do **not** build a process group from a *flattened submesh*
   (`submesh._flatten().get_group()`) when that submesh differs per replicate rank —
   the uncoordinated `new_group()` calls deadlock; use the equivalent sequential
   all-reduces instead. Broadcast control-flow scalars (recycling/sampling step
   counts, random architecture choices) across the CP group so every rank takes
   the same branch — divergent branches deadlock NCCL. Per-element randomness
   (dropout, noise, augmentation) must follow **single-device RNG *entropy* equivalence**
   (callout above) — audit each mesh axis for SAME-vs-DIFFERENT rather than assuming a
   blanket "per-rank" or "shared" rule.
8. **No uneven sharding; co-shard coupled axes.** Guard every `Shard` dim with an
   explicit `shape[dim] % mesh_size == 0` check (so per-rank buffers are equal-size).
   Reject unhandled `Partial` placements at `autograd.Function` boundaries. A feature
   with two coupled semantic axes — pair `[N, N, …]` (token×token) or atom→token
   `[N_atoms, N]` (atoms×tokens) — must keep those axes co-sharded (same mesh axis) or
   co-replicated as the consuming algorithm requires (e.g. distributed outer-gather /
   scatter-reduce); sharding them independently misaligns the per-rank tiles. A fused
   kernel may impose its **own** per-shard constraints (a specific dtype, a
   compute-capability floor, or `dim % K == 0` on *each rank* — not merely divisibility
   by the mesh size); gate the kernel behind an explicit capability predicate and pad at
   the data layer to `n_shards × per_shard_multiple`, rather than assuming it always
   applies or forcing a blanket dtype cast to satisfy it.
9. **Always pass explicit `shape`, `stride`, and `placements` to
   `DTensor.from_local()`.** Omitting them triggers a metadata all-gather that can
   deadlock on heterogeneous shards. Update `Shard(dim)` indices whenever a
   reshape/squeeze changes tensor rank. **Never call `full_tensor()` on a forward/
   backward hot path** — it forces an all-gather; it is a test-only reassembly tool
   (use `to_local()` in production).
10. **Reduce gradients in ≥fp32; use `torch.promote_types(dtype, torch.float32)`,
    never hard-coded `.float()`.** Hard-coded `.float()` silently breaks fp64 test
    paths. Cast auxiliary tensors (masks, biases) to the **operand dtype before
    arithmetic** — under autocast an fp32 mask times a bf16 activation silently upcasts
    the result to fp32, poisoning the forward output, the saved tensors, and the
    backward. In `backward`, explicitly restore the compute dtype (via `promote_types`)
    before mixed-precision math: `custom_bwd` does **not** restore autocast on CPU, and
    saved tensors may have been downcast to save memory. An *unintended* conversion is
    also a **memory** hazard, not only a correctness one: an accidental up-cast (e.g.
    bf16→fp32) materializes a **transient** O(activation) tensor that parity tests do not
    catch (the math is right after the down-cast) yet inflates peak memory — keep
    intermediates in the intended dtype and verify peak with `/fold-cp:mem_profile`.
11. **Backward memory must match the forward per-rank budget.** Single/token
    O(N/cp0), pair O(N²/(cp0·cp1)) for 2D. A saved-for-backward
    tensor that scales with full N or S is a critical bug. Activation/gradient
    checkpointing is **not** a free ceiling reduction — it trades stored activations for
    a **transient recompute peak** that becomes the new maximum; budget and profile the
    *recompute* (`/fold-cp:mem_profile`), and apply any sequence-length cap **before** the
    per-rank shard-size rounding, not after.
12. **Fail fast.** Distributed entrypoints must error if `torch.distributed` is not
    initialized — no silent single-GPU fallback.

## Testing (how correctness is proven)

13. **The serial forward+backward is the only golden ground truth.** Every parity
    test compares the CP path against the serial reference on identical inputs and
    weights.
14. **No vacuous tests.** Use explicit random `grad_output` — never
    `.sum().backward()`. Assert sharding is active **when either 2D mesh axis has size > 1**
    (local shape < global on sharded dims; `(cp0, cp1)=(1,1)` is a valid debugging mesh
    where local == global, so skip the check there), replicated values are identical
    across ranks, and gradients are
    non-zero. **Perturb/randomize zero-initialized parameters before testing** — AF/
    residual-style "final" projections are zero-init, so the output and *all* grads
    are vacuously zero and parity passes as `0 == 0`. **Check per-PARAMETER gradient
    parity, not only the input gradient** — a parameter replicated over a sharded mesh
    axis whose grad is left as a per-rank partial (a missing all-reduce) is invisible
    to forward + input-grad parity; only `param.grad.full_tensor()` vs serial catches
    it. Include at least one adversarial/boundary case. **Test features must be non-trivial** —
    realistic value ranges, a valid (not all-masked) mask, and coupled-feature consistency; a
    degenerate / constant / all-masked input passes vacuously.
15. **Derive tolerances from first principles.** Prefer fp64 + `assert_close`
    default tolerances; a tolerance that must be loosened to pass is a bug signal,
    not a knob. A *near*-tolerance fp64 mismatch often traces to an **unmirrored
    incidental dtype promotion in the serial reference** — e.g. an `int64` count
    `+ python-float eps` promotes the sum to fp32 and rounds the eps away; mirror the
    serial promotion bit-for-bit instead of loosening `atol`.
16. **Distributed test commands carry a `timeout` and per-collective timeouts.**
    Prefix runs with `timeout 120 …`; set NCCL/process-group timeouts so deadlocks
    fail fast instead of hanging the session.

## Reduction, lifecycle & dtype-path (correctness)

17. **Reduce on the right mesh axes; pin the replicate axis.** A quantity sharded over
    data/token axes is summed/averaged over the **dp and cp0** axes only — never reduce
    it over the **cp1 replicate axis** (the data is duplicated there; reducing
    double-counts). Conversely, **all-reduce scalar losses over cp1** so every replicate
    rank sees an identical loss, and force determinism on cp1 by computing
    non-associative / iterative ops (SVD/Kabsch alignment, argmin tie-breaks) on **one
    rank in fp32 and broadcasting** — relying on cp1 ranks to compute bit-identically is
    a drift bug.
18. **Distributed lifecycle is part of correctness.** Move modules to device **before**
    DTensor-wrapping (CPU module + CUDA mesh = mismatch). **Every trainable parameter
    must be a DTensor** — a plain-tensor param silently bypasses gradient redistribution
    and diverges across ranks; freeze/placeholder unimplemented modules to keep
    `state_dict` portable. Save checkpoints as **plain tensors** (DTensor → full); on
    load **realign placements via the live model's `state_dict()` template and
    redistribute optimizer state to the parameter placements before `optimizer.step()`**.
    Resume offsets the seed by rank (+ epoch/global_step). EMA must be **Replicate-only**
    (in-place `.data` math bypasses DTensor dispatch). See `/fold-cp:dist_lifecycle`.
19. **Audit the dtype path of every `autograd.Function`.** Decorate it
    `@torch.amp.custom_fwd(device_type="cuda")` / `custom_bwd`, then trace dtypes
    end-to-end against the app's precision semantics: cast operands (and masks/biases) to
    the compute dtype **before** math (Rule 10); **pin every communicated buffer to one
    dtype across ranks before the collective** — a dtype that varies with a per-rank
    branch or autocast state makes `send`/`recv` buffers mismatch and **deadlock**; and
    allow **no unintended up/down-cast** — an accidental cast is correctness-silent
    (parity passes after the round-trip) but a **transient peak-memory** blowup invisible
    to parity tests (catch it with `/fold-cp:mem_profile`). Treat casts as
    **precision-mode-conditional**: gate them on `torch.is_autocast_enabled(...)` — a cast
    that is correct under mixed precision (autocast on, fp32 params) can mis-type or crash
    under true low precision (autocast off, low-precision params), and vice-versa — and
    note that some ops **unconditionally** promote to fp32 regardless of autocast (e.g.
    `logsumexp` in tiled/ring softmax), so `custom_fwd` autocast governs matmuls, not every
    elementwise/reduction dtype. Keep the `Function` **signature** honest: return exactly
    **one gradient per forward input** (a stray/missing `None` silently mis-maps grads) and
    pass flag/boolean arguments **by keyword** so a positional slip cannot swap a dtype flag
    for an unrelated one.

## Liveness & axis bookkeeping (correctness)

20. **Every rank must reach every collective — guard rank-asymmetric work.** A collective
    deadlocks the whole group if any rank fails to arrive: a rank that raises,
    early-returns, or spins in an infinite loop **before** a collective hangs its peers
    (a watchdog cannot fire on a rank that never calls in). So (a) any rank-asymmetric
    pre-collective work — a dataset fetch, a validation step, an item that may be skipped
    — must first **broadcast an ok/error sentinel** (carrying the failing rank's
    traceback) so all ranks abort together; (b) bound every retry/fallback loop to a
    finite depth; (c) for **keyed / dict-valued reductions** (per-chain, per-class
    metrics) union the key set across **every mesh axis that holds independent data — CP
    *and* DP, not just CP** — pad missing keys with sentinels so all ranks issue an
    identical sequence of collectives, reduce `(numerator, denominator)` separately (never
    the pre-divided ratio), and call the collective even with an **empty** local
    accumulator (e.g. `(0, 0)`); (d) prefer an op with a real compute kernel for your
    dtype — an op with **none** (e.g. an int64 `matmul`/one-hot product) silently stalls
    one rank past the collective timeout. Data-pipeline and control-path bugs under CP
    surface as **hangs, not tracebacks** — treat data/featurizer correctness as a
    distributed-liveness concern (`/fold-cp:shard_data_feats`). Use **`try/finally` for cleanup**
    (destroy the process group, free the GPU, save artifacts on exit) — but **never a `try/except`
    that swallows an error and continues**: the throwing rank skips its remaining collectives and
    deadlocks the others; handle a rank-asymmetric error with the ok/error sentinel (a), not a bare
    `except`. **On exit, free assigned GPUs** — destroy process groups and kill orphaned ranks so a
    finished job/agent does not leak GPU memory (the `/fold-cp:dispatch_work` shutdown protocol
    enforces this for teammate agents). **Tear a finished teammate down with an explicit
    `shutdown_request`** (a `SendMessage` it approves): completed background teammates do **not**
    auto-dismiss and `TaskStop` only stops a *running* task, so otherwise they persist as idle
    clutter the lead cannot clear.
21. **Unflatten product axes in the right order before you index or reduce them.** When a
    single tensor axis is a flattened product of two semantic axes — e.g. a leading
    `(batch × multiplicity)` from `repeat_interleave(M, dim=0)`, or `(heads × conformers)`
    — indexing or reducing it as one axis silently crosses sample boundaries (and
    mismatches any un-flattened tensor it combines with). Reshape to the **correct**
    `(batch, M, …)` order (not the transposed `(M, batch, …)`) before the `mean`/reduce,
    and use **ceil-division** for chunk/step counts over it. This is the value-space
    analogue of renumbering `Shard(dim)` after a reshape (Rule 9): merging or splitting an
    axis changes what every later index means — fix the index, not the symptom.

## Orchestration & context economy (working at scale — quality-neutral)

> These are efficiency invariants for the agentic workflow itself: they bound the
> orchestrator's context and the number of round-trips **without changing what code is
> written or how thoroughly it is checked**. A long, many-wave integration re-reads the
> orchestrator's context on every turn, so verbatim agent reports and re-narrated status are
> the dominant *avoidable* cost — distinct from the irreducible cost of the parity tests and
> reviews, which these rules never touch. (They are also model-/hardware-agnostic: nothing
> here assumes control over the runtime's model tiers or its compute environment.)

22. **Verify gates from durable evidence; re-execute only on doubt.** A delegated result
    becomes a passed gate only after you inspect the **evidence on disk**, never a prose
    "PASS": (a) the committed **test source** is non-vacuous (Rule 14); (b) the **diff**
    touches only the distributed tree (Rule 2) and pre-commit is clean; (c) the tee'd **log**
    shows the right mesh/`world_size` and the measured-difference-vs-tolerance numbers for a
    run against the **current** tree. When all three inspect clean, that *is* the verified
    gate — re-running the test then adds GPU cost without adding rigor (a re-run cannot turn a
    vacuous test non-vacuous; only reading its assertions can). **Re-execute only when the
    evidence is missing, stale (the tree changed since the log), ambiguous, or fails
    inspection.** An unverified claim is never a gate. (For this to be checkable, parity tests
    must **log their evidence on success**, not assert silently — see the `test` skill.)
23. **Keep the orchestrator's context small — durable files, not transcript.** Delegated
    agents write their detailed report/log to a **file** and return a **terse, verifiable
    summary** (verdict + artifact paths + the key numbers), never a long narrative dumped into
    the orchestrator's context. Reference stable shared content — playbooks, these rules, the
    recon docs, placement contracts — **by path**, rather than pasting it into spawn prompts
    or status updates; never paste another agent's message history. The **ledger** is the
    durable orchestration state: update it in **batched writes** (once per wave) and keep each
    progress note to one line. None of this changes the work or its checks — only where the
    report lives and how often the big context is re-read.
24. **Batch independent work; minimise round-trips** (generalises Rule 5 to delegated work).
    Launch independent agents/commands **together**, and collect their results together,
    rather than waking the orchestrator one sub-result at a time. When the work-list is known
    up front and the items are independent, prefer a **deterministic pipeline** over many
    separate dispatch round-trips. Same work, same gates — fewer serial wake-ups of the
    (large, re-read-every-turn) orchestrator context.

> Mechanical enforcement of rules 2 and 16 is available via the opt-in PreToolUse
> guard — write `.fold-cp/config.json` (see `/fold-cp:learn_context`) to enable it.

> **Out of scope here:** generic code style (clear naming, no dead code, actionable error
> messages) is deferred to `pre-commit` / the `code_quality` plugin (Rule 5), not these CP
> hard rules. Per-feature hygiene — identify coverage before implementing, test save/load
> back-compat, and sweep `TODO`/`FIXME` after a feature lands — lives in the `test` skill.
