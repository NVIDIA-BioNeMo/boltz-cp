# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Tests for 1D CP DiffusionConditioning1D and the pair-to-atom z-path gather.

Verifies forward and backward parity against serial DiffusionConditioning on
a 2D mesh ``(dp, cp)`` with 2-element placements ``(Shard(0), Shard(1))``.

Also tests the 1D-CP z-path gather directly: a cross-rank query-token row fetch
(:func:`distributed_gather`) followed by a LOCAL key-column gather via the generic
:func:`_dtensor_gather_along_axis` (window-batched key ids broadcast over the query-window
axis by :func:`_broadcast_over_query_window`). The direct test is ADVERSARIAL — query windows
reference token z-rows owned by OTHER ranks — which the prior purely-local clamp-gather
silently corrupted. It reproduces that bug (via :func:`_buggy_pre_mr_local_clamp_gather`) to
confirm the fix removes it, and covers the generic gather's input validation and a
non-window-shape (feature-axis) gather.

Maps to: src/boltz/distributed/model/modules/diffusion_conditioning_1d.py
"""

from functools import partial

import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from boltz.data import const as boltz_const
from boltz.distributed.data.feature.featurizer import pack_atom_features
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.flatten_and_unflatten import shardwise_unflatten_sharded
from boltz.distributed.model.layers.gather import distributed_gather
from boltz.distributed.model.layers.utils import convert_single_repr_to_window_batched_key
from boltz.distributed.model.modules.diffusion_conditioning_1d import (
    DiffusionConditioning1D,
    _broadcast_over_query_window,
    _dtensor_gather_along_axis,
)
from boltz.distributed.testing.utils import create_atom_to_token_dtensor
from boltz.distributed.utils import update_exhaustive_strides
from boltz.model.modules.diffusion_conditioning import DiffusionConditioning as SerialDiffusionConditioning
from boltz.model.modules.encodersv2 import get_indexing_matrix, single_to_keys
from boltz.testing.utils import (
    SetModuleInfValues,
    assert_all_identical,
    assert_tensors_close_with_pad,
    get_param_by_key,
    init_module_params_uniform,
    init_tensors_uniform,
    pad_or_shrink_to_length,
    random_features,
    seed_by_rank,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)

# Symmetric init range keeps outputs O(1), enabling meaningful tolerances.
_INIT_LOW, _INIT_HIGH = -0.08, 0.08

# Subset of atom feature keys needed by DiffusionConditioning1D.
# atom_counts_per_token is passed to random_features but not to the module.
_selected_atom_keys = {
    "atom_pad_mask",
    "ref_pos",
    "ref_space_uid",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "atom_to_token",
    "atom_counts_per_token",
}
_module_atom_keys = _selected_atom_keys - {"atom_counts_per_token"}


# ---------------------------------------------------------------------------
# 1D-CP z-path gather unit test (cross-rank row fetch + local column gather)
# ---------------------------------------------------------------------------


def _buggy_pre_mr_local_clamp_gather(
    z_dt: DTensor,
    q_ids_dt: DTensor,
    k_ids_dt: DTensor,
    device_mesh,
    W: int,
    H: int,
    output_placements: tuple,
    output_global_shape: torch.Size,
) -> DTensor:
    """Reproduce the DELETED pre-MR ``_LocalRowSlabGather`` forward — the bug this MR fixes.

    "Bad-bond drift" root cause: at cp>1 the pre-MR z-path gathered the pair-to-atom z rows
    PURELY LOCALLY. It mapped GLOBAL query token ids to LOCAL row indices via
    ``q_ids - cp_rank * n_tokens_per_shard`` and CLAMPED out of range. But the packed atoms are
    re-sharded by the valid mask (``distributed_pack_and_pad``) INDEPENDENTLY of the z row-slab
    token order, so a boundary window's VALID query atom can reference a token z-row owned by
    ANOTHER cp rank; the shift then lands outside ``[0, n_rows_local)`` and the clamp silently
    returns the WRONG local row. The corrupted ``z_to_p`` pair bias propagates to the
    DiffusionConditioning output and, downstream, to soft-geometry error — bonds exceeding the
    ideal-geometry tolerance (the OST >12σ "bad bond" metric) drift away from the serial/cp=1
    result. Forward-only reconstruction (sufficient to exhibit the wrong-row symptom).
    """
    cp_rank = device_mesh.get_coordinate()[1]  # row-slab owner: rank r owns rows [r*n, (r+1)*n)
    z_local = z_dt.to_local()  # [B_local, n_rows_local, N_full, D]
    b_local, n_rows_local, n_cols, d = z_local.shape
    q_local = q_ids_dt.to_local()  # [B_local, K_local, W] — GLOBAL token ids
    k_local = k_ids_dt.to_local()  # [B_local, K_local, H] — GLOBAL token ids
    k_win = q_local.shape[1]
    # The bug: shift global query ids to local rows + clamp (no cross-rank fetch).
    q_clamped = (q_local - cp_rank * n_rows_local).clamp(0, n_rows_local - 1)
    k_clamped = k_local.clamp(0, n_cols - 1)
    b_idx = torch.arange(b_local, device=z_local.device)[:, None, None]
    q_exp = q_clamped.reshape(b_local, k_win * W, 1).expand(-1, -1, H)
    k_exp = k_clamped.reshape(b_local, k_win, H).unsqueeze(2).expand(-1, -1, W, -1).reshape(b_local, k_win * W, H)
    gathered = z_local[b_idx, q_exp, k_exp, :].reshape(b_local, k_win, W, H, d).contiguous()
    stride = update_exhaustive_strides(gathered.shape, gathered.stride(), output_global_shape)
    return DTensor.from_local(
        gathered, device_mesh=device_mesh, placements=output_placements, shape=output_global_shape, stride=stride
    )


def parallel_assert_zpath_gather(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    B,
    N_tokens,
    D,
    K,
    W,
    H,
    z_global_host,
    q_ids_global_host,
    k_ids_global_host,
    gathered_expected_host,
    d_gathered_global_host,
    d_z_expected_global_host,
):
    """Exercise the 1D-CP z-path gather: cross-rank row fetch (distributed_gather) +
    LOCAL key-column gather (_dtensor_gather_along_axis). The adversarial q_ids reference
    OFF-rank token rows (a query window owned by rank r maps to token rows owned by another
    rank) — the case the old purely-local clamp-gather silently corrupted.

    Three arms (dejunl note 56913168 — what is "bad-bond drift" / what confirms the fix):
      1. FIXED (this MR): cross-rank fetch + local column gather matches serial at fp64 floor.
      2. BUGGY (pre-MR, reconstructed via _buggy_pre_mr_local_clamp_gather): the local
         shift+clamp gather DIVERGES from serial at cp>1 (O(z-scale) wrong-row error) — the
         deterministic root cause of the bad-bond drift; at cp=1 it matches (sharding-induced).
      3. Generic gather input validation + a non-window (feature-axis) gather.

    This deterministic gather-level parity is the confirmation: it pins the exact divergence the
    fix removes. The end-to-end OST bad-bond COUNT is the gold-standard but stochastic/multi-hour
    metric; it was already run and PASSED at the fix (pdna 5->0, 8jfr 30->0; commits 552eecd00 /
    15a723ffe) — cited as confirmation, not re-run here as a gate.
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    seed_by_rank(rank)

    device_mesh = manager.device_mesh
    cp_size = grid_group_sizes["cp"]

    # z: [B, N, N, D] with row-slab placements (Shard(0), Shard(1))
    z_global = z_global_host.to(device=manager.device, dtype=dtype)
    z_dt = distribute_tensor(z_global, device_mesh, (Shard(0), Shard(1))).requires_grad_(True)

    output_placements = (Shard(0), Shard(1))
    output_global_shape = torch.Size([B, K, W, H, D])

    # q_ids/k_ids hold GLOBAL token ids, sharded on the window axis K like the packed atoms.
    q_ids_dt = distribute_tensor(q_ids_global_host.to(device=manager.device), device_mesh, (Shard(0), Shard(1)))
    k_ids_dt = distribute_tensor(k_ids_global_host.to(device=manager.device), device_mesh, (Shard(0), Shard(1)))

    # Sharding is active: local K < global K when cp>1.
    if grid_group_sizes["cp"] > 1:
        assert q_ids_dt.to_local().shape[1] < K, "q_ids not sharded along K"

    # Step 1 (cross-rank): fetch query-token z-rows. Output (B, K, W, N_tokens, D), Shard(1) on K.
    z_rows_dt = distributed_gather(z_dt, q_ids_dt, axis=1, are_ids_contiguous=True)
    # Step 2 (local): window-batching lives in the caller — broadcast the (B, K, H) key ids over
    # the W query-window axis to (B, K, W, H), then gather the key columns (axis 3 = N_tokens)
    # with the generic helper (no window semantics inside the gather).
    k_ids_bkwh_dt = _broadcast_over_query_window(k_ids_dt, W)
    assert k_ids_bkwh_dt.shape == torch.Size([B, K, W, H]), k_ids_bkwh_dt.shape
    gathered_dt = _dtensor_gather_along_axis(z_rows_dt, k_ids_bkwh_dt, dim=3)

    assert isinstance(gathered_dt, DTensor), f"Expected DTensor, got {type(gathered_dt)}"
    assert gathered_dt.placements == output_placements
    assert gathered_dt.shape == output_global_shape, f"{gathered_dt.shape} != {output_global_shape}"

    gathered_full = gathered_dt.full_tensor()
    gathered_expected = gathered_expected_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(gathered_full, gathered_expected, msg="z-path gather forward mismatch")
    assert gathered_full.abs().sum() > 0, "Gathered output is all-zero"

    # Backward with explicit random grad_output (no .sum().backward()).
    d_gathered = distribute_tensor(
        d_gathered_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        output_placements,
    )
    gathered_dt.backward(d_gathered)

    d_z_expected = d_z_expected_global_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(
        z_dt.grad.full_tensor(),
        d_z_expected,
        msg="z-path gather backward mismatch",
    )
    assert z_dt.grad.full_tensor().abs().sum() > 0, "z gradient is all-zero"

    # ------------------------------------------------------------------
    # Symptom reproduction + fix confirmation (dejunl note 56913168): on this ADVERSARIAL
    # cross-rank fixture, the DELETED pre-MR local clamp-gather DIVERGES from serial (it fetches
    # the WRONG z-rows for off-rank query tokens), while the cross-rank fix (above) matches serial
    # at the fp64 floor. This is the deterministic confirmation that the MR removes the
    # bad-bond-drift root cause (corrupted z_to_p pair bias at cp>1).
    # ------------------------------------------------------------------
    buggy_dt = _buggy_pre_mr_local_clamp_gather(
        z_dt.detach(), q_ids_dt, k_ids_dt, device_mesh, W, H, output_placements, output_global_shape
    )
    buggy_full = buggy_dt.full_tensor()
    fixed_err = (gathered_full - gathered_expected).abs().max().item()
    buggy_err = (buggy_full - gathered_expected).abs().max().item()
    if cp_size > 1:
        # Cross-rank windows exist ⇒ the pre-MR clamp fetches WRONG, independent ~U(-0.08, 0.08)
        # z-rows ⇒ O(z-scale ≈ 0.1) divergence, far above the fixed path's fp64 floor.
        assert buggy_err > 1e-2, (
            f"pre-MR local clamp-gather did NOT diverge (buggy_err={buggy_err:.3e}); the adversarial "
            "fixture is not exercising cross-rank z-rows — the symptom test would be vacuous"
        )
        assert fixed_err < 1e-6, f"fixed cross-rank gather not at parity with serial (fixed_err={fixed_err:.3e})"
        assert buggy_err > 1e3 * max(
            fixed_err, 1e-15
        ), f"buggy gather not decisively worse than the fix (buggy={buggy_err:.3e}, fixed={fixed_err:.3e})"
    else:
        # cp=1: a single rank owns all z-rows ⇒ no cross-rank case ⇒ the pre-MR gather is also
        # correct. Confirms the bug is sharding-induced (cp>1-specific), not generic.
        torch.testing.assert_close(buggy_full, gathered_expected, msg="cp=1: pre-MR gather should match serial")

    # ------------------------------------------------------------------
    # Generic gather: input validation (comment #3) + non-window genericity (comment #2)
    # ------------------------------------------------------------------
    # src and idx must both be DTensors.
    with pytest.raises(TypeError):
        _dtensor_gather_along_axis(z_rows_dt.to_local(), k_ids_bkwh_dt, dim=3)
    with pytest.raises(TypeError):
        _dtensor_gather_along_axis(z_rows_dt, k_ids_bkwh_dt.to_local(), dim=3)
    # idx must be an integer tensor.
    with pytest.raises(TypeError):
        _dtensor_gather_along_axis(z_rows_dt, k_ids_bkwh_dt.to(torch.float32), dim=3)
    # idx.ndim must be dim+1 (leading axes + the gather-count axis): k_ids_dt is (B,K,H)=3D
    # but dim=3 requires a 4D idx -> rejected.
    with pytest.raises(ValueError):
        _dtensor_gather_along_axis(z_rows_dt, k_ids_dt, dim=3)
    # The gather axis must be REPLICATED: a (B, cp) idx sharded on axis 1 gathers z_dt along its
    # sharded axis 1 -> rejected (each rank lacks the full extent of the gather axis).
    sharded_axis_idx_global = torch.zeros(B, cp_size, dtype=torch.long)
    sharded_axis_idx_dt = distribute_tensor(
        sharded_axis_idx_global.to(device=manager.device), device_mesh, (Shard(0), Shard(1))
    )
    with pytest.raises(ValueError):
        _dtensor_gather_along_axis(z_dt, sharded_axis_idx_dt, dim=1)

    # Genericity: the helper is NOT window-specific. Gather z (B, N, N, D) along the trailing
    # feature axis (dim=3, no trailing axes) with a hand-built index; compare to torch.gather.
    # The index must be IDENTICAL across ranks (rank-independent generator) so the sharded
    # result reassembles to the same global tensor as the serial reference.
    M_feat = 2
    feat_gen = torch.Generator().manual_seed(20260615)
    feat_idx_global = torch.randint(0, D, (B, N_tokens, N_tokens, M_feat), generator=feat_gen)
    feat_idx_dt = distribute_tensor(feat_idx_global.to(device=manager.device), device_mesh, (Shard(0), Shard(1)))
    z_feat_dt = distribute_tensor(z_global, device_mesh, (Shard(0), Shard(1)))
    feat_gathered = _dtensor_gather_along_axis(z_feat_dt, feat_idx_dt, dim=3)
    feat_expected = torch.gather(z_global, 3, feat_idx_global.to(device=manager.device))
    torch.testing.assert_close(feat_gathered.full_tensor(), feat_expected, msg="generic feature-axis gather mismatch")

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype",
    [
        (((2, 3), True, "cpu", "ENV"), torch.float64),
        (((1, 1), True, "cuda", "ENV"), torch.float64),
        (((1, 2), True, "cuda", "ENV"), torch.float64),
        (((1, 3), True, "cuda", "ENV"), torch.float64),
    ],
    indirect=["setup_env"],
    ids=["dp2_cp3_cpu", "dp1_cp1_cuda", "dp1_cp2_cuda", "dp1_cp3_cuda"],
)
def test_zpath_gather(setup_env, dtype):
    """Test the 1D-CP z-path gather (cross-rank row fetch + local column gather).

    ADVERSARIAL by construction: q_ids reference token rows ACROSS the full N_tokens range
    (not the rank-local rows), so a window owned by rank r maps to token z-rows owned by
    OTHER ranks — the off-rank case that the prior purely-local clamp-gather silently
    corrupted. The cross-rank distributed_gather must fetch the true rows. Serial reference
    is z[b, q_ids, k_ids]; fp64 default tol; explicit random grad_output.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type, world_size)

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        cp_size = grid_group_sizes["cp"]
        B = 2 * grid_group_sizes["dp"]
        N_tokens = 6 * cp_size  # divisible by cp_size
        D = 4
        W = 3
        H = 6  # H <= N_tokens
        N_atoms = N_tokens * W  # one window per token
        K = N_atoms // W  # K = N_tokens

        # z: [B, N_tokens, N_tokens, D]
        z_global = torch.empty(B, N_tokens, N_tokens, D, dtype=dtype)
        init_tensors_uniform([z_global], low=_INIT_LOW, high=_INIT_HIGH)
        z_global.requires_grad_(True)

        # q_ids_global: [B, K, W] — GLOBAL row (token) indices in [0, N_tokens).
        # ADVERSARIAL: window i (owned by rank i//(K/cp)) references row (N_tokens-1 - i),
        # which lands in a DIFFERENT rank's row-slab for boundary windows. This forces the
        # cross-rank fetch; the old local clamp-gather would mis-resolve these.
        q_ids = torch.zeros(B, K, W, dtype=torch.long)
        for i in range(K):
            q_ids[:, i, :] = (N_tokens - 1 - i) % N_tokens
        q_ids = q_ids.clamp(0, N_tokens - 1)

        # k_ids_global: [B, K, H] — column indices in [0, N_tokens)
        k_ids = torch.randint(0, N_tokens, (B, K, H), dtype=torch.long)

        # Serial reference: gather z[b, q, k, :] for each (b, i, w, h)
        gathered_expected = torch.zeros(B, K, W, H, D, dtype=dtype)
        for b in range(B):
            for i in range(K):
                for w in range(W):
                    for h in range(H):
                        gathered_expected[b, i, w, h, :] = z_global[b, q_ids[b, i, w], k_ids[b, i, h], :]

        # Random upstream gradient
        d_gathered = torch.empty_like(gathered_expected)
        init_tensors_uniform([d_gathered], low=_INIT_LOW, high=_INIT_HIGH)

        # Serial backward: scatter-add
        z_serial = z_global.detach().clone().requires_grad_(True)
        gathered_serial = torch.zeros(B, K, W, H, D, dtype=dtype)
        for b in range(B):
            for i in range(K):
                for w in range(W):
                    for h in range(H):
                        gathered_serial[b, i, w, h, :] = z_serial[b, q_ids[b, i, w], k_ids[b, i, h], :]
        gathered_serial.backward(d_gathered)
        d_z_expected = z_serial.grad.detach()

    spawn_multiprocessing(
        parallel_assert_zpath_gather,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        B,
        N_tokens,
        D,
        K,
        W,
        H,
        z_global.detach().cpu(),
        q_ids.cpu(),
        k_ids.cpu(),
        gathered_expected.detach().cpu(),
        d_gathered.detach().cpu(),
        d_z_expected.cpu(),
    )


# ---------------------------------------------------------------------------
# DiffusionConditioning1D full-module parity test
# ---------------------------------------------------------------------------


def _worker_diffusion_conditioning_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    # Module dimensions
    atom_s,
    atom_z,
    token_s,
    token_z,
    atom_feature_dim,
    W,
    H,
    atom_encoder_depth,
    atom_encoder_heads,
    token_transformer_depth,
    token_transformer_heads,
    atom_decoder_depth,
    atom_decoder_heads,
    layer_state_dict,
    # Inputs
    feats_global_host,
    s_trunk_global_host,
    z_trunk_global_host,
    rel_pos_enc_global_host,
    # Expected outputs
    q_expected_global_host,
    c_expected_global_host,
    atom_enc_bias_expected_global_host,
    atom_dec_bias_expected_global_host,
    token_trans_bias_expected_global_host,
    # Upstream grads
    d_q_global_host,
    d_c_global_host,
    d_atom_enc_bias_global_host,
    d_atom_dec_bias_global_host,
    d_token_trans_bias_global_host,
    # Expected input grads
    d_s_trunk_expected_global_host,
    d_z_trunk_expected_global_host,
    d_rel_pos_enc_expected_global_host,
    # Expected param grads
    expected_param_grads_global_host_dict,
):
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    seed_by_rank(rank)

    # 1D CP uses 2D mesh (dp, cp)
    device_mesh = manager.device_mesh

    # Build serial module on device, then wrap into DiffusionConditioning1D
    module_serial = SerialDiffusionConditioning(
        token_s=token_s,
        token_z=token_z,
        atom_s=atom_s,
        atom_z=atom_z,
        atoms_per_window_queries=W,
        atoms_per_window_keys=H,
        atom_encoder_depth=atom_encoder_depth,
        atom_encoder_heads=atom_encoder_heads,
        token_transformer_depth=token_transformer_depth,
        token_transformer_heads=token_transformer_heads,
        atom_decoder_depth=atom_decoder_depth,
        atom_decoder_heads=atom_decoder_heads,
        atom_feature_dim=atom_feature_dim,
    )
    module_serial = module_serial.to(device=manager.device, dtype=dtype)
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.train()
    module_serial.apply(SetModuleInfValues())

    module = DiffusionConditioning1D(
        layer=module_serial,
        device_mesh=device_mesh,
    ).train()

    # ------------------------------------------------------------------
    # Distribute atom features with 1D CP placements
    # ------------------------------------------------------------------
    # Under 1D-CP, atom features (``atom_pad_mask``, ``ref_pos`` etc.) have
    # placements ``(Shard(0), Shard(1))`` — batch on dp, N_atoms on cp. The
    # ``atom_to_token`` mapping uses the same numeric placement, but its token
    # dim (dim 2) stays REPLICATED across the cp axis; we route it through
    # ``create_atom_to_token_dtensor`` so the layout invariant — local one-hot
    # spans the full global token dim, matching the production featurizer
    # (``placements_1d.py`` ``PLACEMENT_1D_ATOM`` + ``featurizer.py`` ``ndim==2``
    # branch + ``reconstruct_atom_to_token_global``'s validator) — is asserted
    # in one place.
    placements_single = (Shard(0), Shard(1))

    feats_dt = {}
    for key, val in feats_global_host.items():
        if key not in _module_atom_keys:
            continue
        val_device = val.to(device=manager.device)
        if val.dtype.is_floating_point:
            val_device = val_device.to(dtype=dtype)
        if key == "atom_to_token":
            feats_dt[key] = create_atom_to_token_dtensor(val_device, device_mesh)
        else:
            feats_dt[key] = distribute_tensor(val_device, device_mesh, placements_single)

    # Compute N_atoms_packed for comparison sizing
    feats_dt_packed_for_sizing = pack_atom_features(feats_dt, _module_atom_keys, W)
    N_atoms_packed = feats_dt_packed_for_sizing["atom_pad_mask"].shape[1]

    # ------------------------------------------------------------------
    # Distribute token-level tensors with 1D CP placements
    # ------------------------------------------------------------------
    s_trunk_dt = distribute_tensor(
        s_trunk_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    z_trunk_dt = distribute_tensor(
        z_trunk_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    rel_pos_enc_dt = distribute_tensor(
        rel_pos_enc_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    q_dt, c_dt, atom_enc_bias_dt, atom_dec_bias_dt, token_trans_bias_dt = module(
        s_trunk=s_trunk_dt,
        z_trunk=z_trunk_dt,
        relative_position_encoding=rel_pos_enc_dt,
        feats=feats_dt,
    )

    # Verify output placements have correct ndim for 2D mesh
    mesh_ndim = device_mesh.ndim
    for name, out_dt in [
        ("q", q_dt),
        ("c", c_dt),
        ("atom_enc_bias", atom_enc_bias_dt),
        ("atom_dec_bias", atom_dec_bias_dt),
        ("token_trans_bias", token_trans_bias_dt),
    ]:
        assert len(out_dt.placements) == mesh_ndim, (
            f"{name} placements should have {mesh_ndim} elements, " f"got {len(out_dt.placements)}: {out_dt.placements}"
        )

    # Non-vacuous: outputs must be non-zero
    for name, out_dt in [("q", q_dt), ("c", c_dt), ("atom_enc_bias", atom_enc_bias_dt)]:
        assert out_dt.full_tensor().abs().sum() > 0, f"{name} output is all-zero"

    # ------------------------------------------------------------------
    # Forward comparison: q and c (atom-level, may be padded)
    # ------------------------------------------------------------------
    mask_dt_full = feats_dt_packed_for_sizing["atom_pad_mask"].full_tensor()
    mask_dt_full_expanded = mask_dt_full.unsqueeze(-1)

    atom_pad_mask_global = feats_global_host["atom_pad_mask"].to(device=manager.device, dtype=dtype)
    atom_pad_mask_expanded_global = atom_pad_mask_global.unsqueeze(-1)

    q_expected_device = q_expected_global_host.to(device=manager.device, dtype=dtype)
    c_expected_device = c_expected_global_host.to(device=manager.device, dtype=dtype)

    assert_tensors_close_with_pad(
        q_dt.full_tensor() * mask_dt_full_expanded,
        q_expected_device * atom_pad_mask_expanded_global,
        axis=1,
        pad_val=0,
    )
    assert_tensors_close_with_pad(
        c_dt.full_tensor() * mask_dt_full_expanded,
        c_expected_device * atom_pad_mask_expanded_global,
        axis=1,
        pad_val=0,
    )

    # ------------------------------------------------------------------
    # Forward comparison: atom_enc_bias, atom_dec_bias (window-batched pair)
    # ------------------------------------------------------------------
    K_packed = N_atoms_packed // W
    N_atoms_serial = feats_global_host["atom_pad_mask"].shape[1]
    K_serial = N_atoms_serial // W

    mask_dt_query = shardwise_unflatten_sharded(
        feats_dt_packed_for_sizing["atom_pad_mask"], axis=1, sizes=(K_packed, W)
    )
    mask_dt_query_full = mask_dt_query.full_tensor()
    mask_dt_query_full_expanded = mask_dt_query_full[:, :, :, None, None]
    mask_dt_key = convert_single_repr_to_window_batched_key(feats_dt_packed_for_sizing["atom_pad_mask"], W, H)
    mask_dt_key_full = mask_dt_key.full_tensor()
    mask_dt_key_full_expanded = mask_dt_key_full[:, :, None, :, None]
    mask_dt_pair_full_expanded = mask_dt_query_full_expanded * mask_dt_key_full_expanded

    compute_dtype = torch.promote_types(dtype, torch.float32)
    index_matrix = get_indexing_matrix(K_serial, W, H, manager.device).to(dtype=compute_dtype)
    to_keys_fn = partial(single_to_keys, indexing_matrix=index_matrix, W=W, H=H)

    mask_key_expected = to_keys_fn(
        feats_global_host["atom_pad_mask"].to(device=manager.device, dtype=compute_dtype).unsqueeze(-1)
    )
    mask_key_expected_expanded = mask_key_expected[:, :, None, :, :]
    mask_query_expected_expanded = atom_pad_mask_expanded_global.unflatten(
        1, (atom_pad_mask_expanded_global.shape[1] // W, W)
    )[:, :, :, None, :]
    mask_pair_expected_expanded = mask_query_expected_expanded * mask_key_expected_expanded

    for name, bias_dt, bias_expected_host in [
        ("atom_enc_bias", atom_enc_bias_dt, atom_enc_bias_expected_global_host),
        ("atom_dec_bias", atom_dec_bias_dt, atom_dec_bias_expected_global_host),
    ]:
        bias_expected_device = bias_expected_host.to(device=manager.device, dtype=dtype)
        assert_tensors_close_with_pad(
            bias_dt.full_tensor() * mask_dt_pair_full_expanded,
            bias_expected_device * mask_pair_expected_expanded,
            axis=1,
            pad_val=0,
        )

    # ------------------------------------------------------------------
    # Forward comparison: token_trans_bias (token pair level)
    # ------------------------------------------------------------------
    token_trans_bias_expected_device = token_trans_bias_expected_global_host.to(device=manager.device, dtype=dtype)
    torch.testing.assert_close(token_trans_bias_dt.full_tensor(), token_trans_bias_expected_device)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    d_q_padded = pad_or_shrink_to_length(
        d_q_global_host.to(device=manager.device, dtype=dtype), axis=1, target_length=N_atoms_packed
    )
    d_c_padded = pad_or_shrink_to_length(
        d_c_global_host.to(device=manager.device, dtype=dtype), axis=1, target_length=N_atoms_packed
    )
    d_atom_enc_bias_padded = pad_or_shrink_to_length(
        d_atom_enc_bias_global_host.to(device=manager.device, dtype=dtype), axis=1, target_length=K_packed
    )
    d_atom_dec_bias_padded = pad_or_shrink_to_length(
        d_atom_dec_bias_global_host.to(device=manager.device, dtype=dtype), axis=1, target_length=K_packed
    )

    d_q_dt = distribute_tensor(d_q_padded, device_mesh, q_dt.placements)
    d_c_dt = distribute_tensor(d_c_padded, device_mesh, c_dt.placements)
    d_atom_enc_bias_dt = distribute_tensor(d_atom_enc_bias_padded, device_mesh, atom_enc_bias_dt.placements)
    d_atom_dec_bias_dt = distribute_tensor(d_atom_dec_bias_padded, device_mesh, atom_dec_bias_dt.placements)
    d_token_trans_bias_dt = distribute_tensor(
        d_token_trans_bias_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        token_trans_bias_dt.placements,
    )

    torch.autograd.backward(
        [q_dt, c_dt, atom_enc_bias_dt, atom_dec_bias_dt, token_trans_bias_dt],
        [d_q_dt, d_c_dt, d_atom_enc_bias_dt, d_atom_dec_bias_dt, d_token_trans_bias_dt],
    )

    # Check token-level input gradients
    torch.testing.assert_close(
        s_trunk_dt.grad.full_tensor(),
        d_s_trunk_expected_global_host.to(device=manager.device, dtype=dtype),
        msg="s_trunk gradient mismatch",
    )
    torch.testing.assert_close(
        z_trunk_dt.grad.full_tensor(),
        d_z_trunk_expected_global_host.to(device=manager.device, dtype=dtype),
        msg="z_trunk gradient mismatch",
    )
    torch.testing.assert_close(
        rel_pos_enc_dt.grad.full_tensor(),
        d_rel_pos_enc_expected_global_host.to(device=manager.device, dtype=dtype),
        msg="rel_pos_enc gradient mismatch",
    )

    # Non-vacuous: input gradients must be non-zero
    assert s_trunk_dt.grad.full_tensor().abs().sum() > 0, "s_trunk gradient is all-zero"
    assert z_trunk_dt.grad.full_tensor().abs().sum() > 0, "z_trunk gradient is all-zero"
    assert rel_pos_enc_dt.grad.full_tensor().abs().sum() > 0, "rel_pos_enc gradient is all-zero"

    # Parameter grad parity + replicated-param identity across CP ranks
    cp_group = manager.group["cp"]
    for name, grad_expected_global in expected_param_grads_global_host_dict.items():
        grad_param = get_param_by_key(module, name).grad
        assert grad_param is not None, f"Missing grad for param {name}"

        if hasattr(grad_param, "full_tensor"):
            grad_global = grad_param.full_tensor().cpu()
            grad_to_check = grad_param.full_tensor()
        else:
            grad_global = grad_param.detach().cpu()
            grad_to_check = grad_param

        torch.testing.assert_close(
            grad_global,
            grad_expected_global.to(dtype=dtype),
            msg=f"Parameter gradient mismatch for '{name}'",
        )
        assert_all_identical(grad_to_check, cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, pad_zpath",
    [
        (((2, 3), True, "cpu", "ENV"), torch.float64, False),
        (((1, 1), True, "cuda", "ENV"), torch.float64, False),
        (((1, 2), True, "cuda", "ENV"), torch.float64, False),
        (((1, 3), True, "cuda", "ENV"), torch.float64, False),
        # pad_zpath: leave the pair-bias upstream grad UNMASKED so gradient reaches z through
        # the window-batching boundary padding-key positions — the non-vacuous probe of the
        # z-path pad mask (serial z_to_p=0 at pad keys; an unmasked gather would leak z[q,0]
        # and z_trunk grad would diverge). Verified to FAIL without the key mask.
        (((2, 3), True, "cpu", "ENV"), torch.float64, True),
        (((1, 2), True, "cuda", "ENV"), torch.float64, True),
    ],
    indirect=["setup_env"],
    ids=[
        "dp2_cp3_cpu",
        "dp1_cp1_cuda",
        "dp1_cp2_cuda",
        "dp1_cp3_cuda",
        "dp2_cp3_cpu_padzpath",
        "dp1_cp2_cuda_padzpath",
    ],
)
def test_diffusion_conditioning_1d(setup_env, dtype, pad_zpath):
    """Test DiffusionConditioning1D forward/backward parity vs serial DiffusionConditioning.

    Covers:
    * Forward parity for q, c (atom-level), atom_enc_bias, atom_dec_bias
      (window-batched pair), and token_trans_bias (token pair).
    * Backward parity for input gradients (s_trunk, z_trunk, relative_position_encoding)
      and parameter gradients.
    * Non-vacuousness: outputs and gradients are non-zero.
    * Replicated parameter gradients are identical across CP ranks.
    * ``pad_zpath``: with the pair-bias upstream gradient left UNMASKED at padding positions,
      gradient flows into z through padding-key positions, making the z_trunk-gradient
      comparison a decisive (non-vacuous) check of the z-path padding mask.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type, world_size)

    seed = 42
    with torch.random.fork_rng(devices=[], enabled=True):
        seed_by_rank(0, seed=seed)

        cp_size = grid_group_sizes["cp"]
        B = 1 * grid_group_sizes["dp"]

        W = 4
        H = 8

        # Uniform atom counts per token so that naive distribute_tensor aligns
        # atom shards with token boundaries. This enables simple testing without
        # the full pad_and_scatter_atom_features_dtensor pipeline.
        n_atoms_per_token = W  # atoms_per_window_queries, keeps windows aligned
        N_tokens = 6 * cp_size  # small but non-trivial
        N_atoms = N_tokens * n_atoms_per_token
        n_atoms_per_token_min = n_atoms_per_token
        n_atoms_per_token_max = n_atoms_per_token
        N_msa = 1

        atom_s = 8
        atom_z = 8
        token_s = 4
        token_z = 4

        atom_encoder_depth = 2
        atom_encoder_heads = 2
        token_transformer_depth = 3
        token_transformer_heads = 2
        atom_decoder_depth = 2
        atom_decoder_heads = 2

        atom_feature_dim = 3 + 1 + boltz_const.num_elements + 4 * 64

        selected_keys = list(_selected_atom_keys)

        feats = random_features(
            size_batch=B,
            n_tokens=N_tokens,
            n_atoms=N_atoms,
            n_msa=N_msa,
            atom_counts_per_token_range=(n_atoms_per_token_min, n_atoms_per_token_max),
            device=torch.device(device_type),
            float_value_range=(_INIT_LOW, _INIT_HIGH),
            selected_keys=selected_keys,
        )
        feats = {k: v.to(dtype=dtype) if v.dtype == torch.float64 else v for k, v in feats.items()}

        N_atoms_actual = feats["atom_pad_mask"].shape[1]
        K = N_atoms_actual // W

        # Token-level inputs
        s_trunk = torch.empty((B, N_tokens, token_s), device=device_type, dtype=dtype, requires_grad=True)
        z_trunk = torch.empty((B, N_tokens, N_tokens, token_z), device=device_type, dtype=dtype, requires_grad=True)
        rel_pos_enc = torch.empty((B, N_tokens, N_tokens, token_z), device=device_type, dtype=dtype, requires_grad=True)
        init_tensors_uniform([s_trunk, z_trunk, rel_pos_enc], low=_INIT_LOW, high=_INIT_HIGH)

        # Build serial reference module
        reference_module = SerialDiffusionConditioning(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            atoms_per_window_queries=W,
            atoms_per_window_keys=H,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            token_transformer_depth=token_transformer_depth,
            token_transformer_heads=token_transformer_heads,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            atom_feature_dim=atom_feature_dim,
        ).to(device=device_type, dtype=dtype)
        reference_module.train()
        init_module_params_uniform(reference_module, low=_INIT_LOW, high=_INIT_HIGH)
        reference_module.apply(SetModuleInfValues())
        layer_state_dict = reference_module.state_dict()

        # Serial forward pass
        feats_serial = {k: v.detach().clone() for k, v in feats.items()}
        s_trunk_serial = s_trunk.detach().clone().requires_grad_(True)
        z_trunk_serial = z_trunk.detach().clone().requires_grad_(True)
        rel_pos_enc_serial = rel_pos_enc.detach().clone().requires_grad_(True)

        # Serial DiffusionConditioning returns (q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias)
        q_expected, c_expected, _to_keys, atom_enc_bias_expected, atom_dec_bias_expected, token_trans_bias_expected = (
            reference_module(
                s_trunk=s_trunk_serial,
                z_trunk=z_trunk_serial,
                relative_position_encoding=rel_pos_enc_serial,
                feats=feats_serial,
            )
        )

        # Upstream gradients (random, not .sum().backward())
        d_q = torch.empty_like(q_expected)
        d_c = torch.empty_like(c_expected)
        d_atom_enc_bias = torch.empty_like(atom_enc_bias_expected)
        d_atom_dec_bias = torch.empty_like(atom_dec_bias_expected)
        d_token_trans_bias = torch.empty_like(token_trans_bias_expected)
        init_tensors_uniform(
            [d_q, d_c, d_atom_enc_bias, d_atom_dec_bias, d_token_trans_bias],
            low=_INIT_LOW,
            high=_INIT_HIGH,
        )

        # Mask upstream gradients at padding positions
        mask_expanded = feats_serial["atom_pad_mask"].unsqueeze(-1)
        d_q = d_q * mask_expanded
        d_c = d_c * mask_expanded

        compute_dtype = torch.promote_types(dtype, torch.float32)
        index_matrix = get_indexing_matrix(K, W, H, device_type).to(dtype=compute_dtype)
        to_keys_fn_serial = partial(single_to_keys, indexing_matrix=index_matrix, W=W, H=H)
        mask_key_serial = to_keys_fn_serial(
            feats_serial["atom_pad_mask"].to(dtype=compute_dtype, device=d_atom_enc_bias.device).unsqueeze(-1)
        )
        pair_mask = mask_key_serial[:, :, None, :, :] * mask_expanded.unflatten(1, (K, W))[:, :, :, None, :]
        if not pad_zpath:
            # Default: mask the upstream pair-bias gradient at padding (query/key) positions, so
            # the comparison focuses on valid positions. When ``pad_zpath`` is set we deliberately
            # leave it UNMASKED so gradient flows into z through the (window-batching boundary)
            # padding-key positions — the decisive, non-vacuous probe of the z-path pad mask:
            # serial z_to_p=0 at pad keys, so z_trunk grad there must be 0; without the z-path
            # key mask the gather would leak z[q, pad_token] and z_trunk grad would diverge.
            d_atom_enc_bias = d_atom_enc_bias * pair_mask
            d_atom_dec_bias = d_atom_dec_bias * pair_mask
        else:
            # Non-vacuity guard: pad_zpath only probes the z-path key mask if the fixture
            # actually has window-boundary padding-keys (window-batched key mask has False
            # entries). Fail LOUDLY if a future W/H/N choice removes them — otherwise the
            # unmasked-bias-grad probe would silently regress to vacuous.
            assert (mask_key_serial == 0).any(), (
                "pad_zpath probe is vacuous: no boundary padding-keys in this fixture "
                f"(W={W}, H={H}, K={K}); choose W/H/N so to_keys yields zero rows."
            )

        # Serial backward
        torch.autograd.backward(
            [q_expected, c_expected, atom_enc_bias_expected, atom_dec_bias_expected, token_trans_bias_expected],
            [d_q, d_c, d_atom_enc_bias, d_atom_dec_bias, d_token_trans_bias],
        )

        expected_param_grads = {
            name: param.grad.detach().cpu()
            for name, param in reference_module.named_parameters()
            if param.grad is not None
        }

    spawn_multiprocessing(
        _worker_diffusion_conditioning_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atom_feature_dim,
        W,
        H,
        atom_encoder_depth,
        atom_encoder_heads,
        token_transformer_depth,
        token_transformer_heads,
        atom_decoder_depth,
        atom_decoder_heads,
        {k: v.detach().cpu() for k, v in layer_state_dict.items()},
        {k: v.detach().cpu() for k, v in feats.items()},
        s_trunk.detach().cpu(),
        z_trunk.detach().cpu(),
        rel_pos_enc.detach().cpu(),
        q_expected.detach().cpu(),
        c_expected.detach().cpu(),
        atom_enc_bias_expected.detach().cpu(),
        atom_dec_bias_expected.detach().cpu(),
        token_trans_bias_expected.detach().cpu(),
        d_q.detach().cpu(),
        d_c.detach().cpu(),
        d_atom_enc_bias.detach().cpu(),
        d_atom_dec_bias.detach().cpu(),
        d_token_trans_bias.detach().cpu(),
        s_trunk_serial.grad.detach().cpu(),
        z_trunk_serial.grad.detach().cpu(),
        rel_pos_enc_serial.grad.detach().cpu(),
        expected_param_grads,
    )
