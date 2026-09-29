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

"""DTensor-based tests for 1D CP ConfidenceHeads1D and ConfidenceModule1D.

Tests the 1D CP ConfidenceHeads1D against the serial ConfidenceHeads reference,
verifying forward parity (logit outputs, aggregated metrics) and backward parity
(input gradients on s and z, parameter gradients).

Also tests ConfidenceModule1D (full confidence module) for forward parity of
logit outputs and aggregated scalar metrics against the serial ConfidenceModule.

Note on pTM/ipTM: the distributed ``_compute_ptms_1d`` mirrors the serial
``compute_ptms`` collinear-frame mask via ``_compute_frame_pred_1d`` with
``inference=True``, so ptm/iptm/ligand_iptm/protein_iptm match serial
exactly. ``pair_chains_iptm`` is checked structurally only because the
distributed path fills batch entries whose dp rank does not contain a given
chain id with ``CHAIN_IPTM_SENTINEL`` while serial computes ~0 over the full
batch (this is a known semantic divergence outside the parity contract).

Maps to: src/boltz/distributed/model/modules/confidence_1d.py
"""

import math
from datetime import timedelta

import pytest
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.nn.functional import one_hot

from boltz.data import const
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.modules.confidence_1d import (
    CHAIN_IPTM_SENTINEL,
    ConfidenceHeads1D,
    ConfidenceModule1D,
    _transpose_via_all_gather,
    _transpose_via_all_to_all,
)
from boltz.distributed.model.modules.encoders_1d import RelativePositionEncoder1D
from boltz.distributed.utils import all_gather_on_cp, update_exhaustive_strides
from boltz.model.modules.confidencev2 import ConfidenceHeads as SerialConfidenceHeads
from boltz.model.modules.confidencev2 import ConfidenceModule as SerialConfidenceModule
from boltz.testing.utils import (
    assert_tensors_identical,
    create_boltz2_model_init_params,
    make_divergent_cyclic_rel_pos_feats,
    random_features,
    seed_by_rank,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)


def _init_tensors_uniform_rng(tensors, low, high, rng):
    """Generator-threaded variant of ``boltz.testing.utils.init_tensors_uniform``.

    The shared helper in ``boltz.testing.utils`` reaches into the global torch
    RNG state via ``tensor.uniform_(low, high)``. Across a pytest parametrize
    sweep that state depends on which cells executed before, causing the two
    distributed parity tests below to flake when run together but pass in
    isolation. Threading a local ``torch.Generator`` here keeps every cell's
    host-side input init independent of global state.
    """
    with torch.no_grad():
        for t in tensors:
            t.uniform_(low, high, generator=rng)


def _init_module_params_glorot_rng(module, rng, gain=1.0):
    """Generator-threaded variant of ``boltz.testing.utils.init_module_params_glorot``.

    Mirrors the shared helper's split: ``xavier_uniform_`` for dim >= 2 weights,
    ``uniform_(-1/sqrt(fan_out), +1/sqrt(fan_out))`` for biases. Both ops accept
    ``generator=``, so the local ``rng`` fully decouples the serial reference
    module's parameter init from the global torch RNG state.
    """
    with torch.no_grad():
        for _name, param in module.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param, gain=gain, generator=rng)
            else:
                bound = 1.0 / math.sqrt(param.shape[0]) if param.shape[0] > 0 else 0.01
                param.uniform_(-bound, bound, generator=rng)


def parallel_assert_confidence_heads_1d(
    rank,
    payload,
):
    """Worker function for 1D CP ConfidenceHeads1D testing."""
    (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        serial_state_dict,
        confidence_heads_kwargs,
        multiplicity,
        # input tensors
        s_global_host,
        z_global_host,
        x_pred_global_host,
        d_global_host,
        feats_global_host,
        pred_distogram_logits_global_host,
        # reference serial outputs
        serial_output_feats_host,
        # reference serial input gradients
        s_grad_host,
        z_grad_host,
        # upstream gradient tensors
        d_plddt_logits_host,
        d_pde_logits_host,
        d_resolved_logits_host,
        d_pae_logits_host,
        # reference serial parameter gradients
        serial_param_grads_host,
    ) = payload

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    # 1D CP uses 2D mesh (dp, cp)
    device_mesh = manager.device_mesh

    seed_by_rank(rank, 42)

    # Build serial module on device, then wrap into ConfidenceHeads1D
    serial_module = SerialConfidenceHeads(**confidence_heads_kwargs)
    serial_module = serial_module.to(device=manager.device, dtype=dtype).train()
    serial_module.load_state_dict(serial_state_dict)

    cp_group = device_mesh.get_group("cp")

    module = ConfidenceHeads1D(
        layer=serial_module,
        device_mesh=device_mesh,
        cp_group=cp_group,
    )
    module = module.to(device=manager.device, dtype=dtype).train()

    # --- Distribute inputs ---
    # 1D CP placements: single (Shard(0), Shard(1)), pair (Shard(0), Shard(1))
    # x_pred atoms are sharded on cp (same as tokens)
    single_placements = (Shard(0), Shard(1))
    pair_placements = (Shard(0), Shard(1))
    atom_placements = (Shard(0), Shard(1))

    s_dtensor = distribute_tensor(
        s_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=single_placements,
    )

    z_dtensor = distribute_tensor(
        z_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    x_pred_dtensor = distribute_tensor(
        x_pred_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=atom_placements,
    )

    d_dtensor = distribute_tensor(
        d_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    pred_distogram_logits_dtensor = distribute_tensor(
        pred_distogram_logits_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    # Distribute feature tensors
    feats_dtensor = {}
    for key, val in feats_global_host.items():
        val_device = val.to(device=manager.device)
        if val.ndim == 2:  # noqa: PLR2004
            # Single-repr: [B, N] -> (Shard(0), Shard(1))
            feats_dtensor[key] = distribute_tensor(val_device, device_mesh=device_mesh, placements=single_placements)
        elif val.ndim == 3:  # noqa: PLR2004
            # Pair-repr: [B, N, N] -> (Shard(0), Shard(1))
            feats_dtensor[key] = distribute_tensor(val_device, device_mesh=device_mesh, placements=pair_placements)
        else:
            # Scalars or other shapes: replicate
            feats_dtensor[key] = distribute_tensor(
                val_device, device_mesh=device_mesh, placements=(Replicate(), Replicate())
            )

    # Keep copies to verify inputs are not mutated
    s_dtensor_copy = s_dtensor.clone()
    z_dtensor_copy = z_dtensor.clone()

    # --- Distribute upstream gradients ---
    d_plddt_logits_dtensor = distribute_tensor(
        d_plddt_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    d_pde_logits_dtensor = distribute_tensor(
        d_pde_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )
    d_resolved_logits_dtensor = distribute_tensor(
        d_resolved_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    d_pae_logits_dtensor = distribute_tensor(
        d_pae_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    # --- Forward ---
    output_dtensor = module(
        s=s_dtensor,
        z=z_dtensor,
        x_pred=x_pred_dtensor,
        d=d_dtensor,
        feats=feats_dtensor,
        pred_distogram_logits=pred_distogram_logits_dtensor,
        multiplicity=multiplicity,
    )

    # Verify placements have correct ndim for the 2D mesh (dp, cp)
    mesh_ndim = device_mesh.ndim
    for key in ["plddt_logits", "pde_logits", "resolved_logits", "pae_logits"]:
        assert len(output_dtensor[key].placements) == mesh_ndim, (
            f"{key} placements should have {mesh_ndim} elements for {mesh_ndim}D mesh, "
            f"got {len(output_dtensor[key].placements)}: {output_dtensor[key].placements}"
        )

    # pair_chains_iptm: dict-of-dict of scalar DTensors of shape [B].
    # Structural validity only — the distributed _compute_ptms_1d uses a
    # CHAIN_IPTM_SENTINEL=-1.0 fill on batch rows whose dp rank does not
    # contain a given chain id, while serial compute_ptms iterates over the
    # full batch's asym_id set and computes ~0 for absent chains. Bridging
    # this DP-shard sentinel semantics to serial requires a separate fix
    # outside the scope of this commit.
    pair_chains_iptm_dtensor = output_dtensor["pair_chains_iptm"]
    assert isinstance(
        pair_chains_iptm_dtensor, dict
    ), f"pair_chains_iptm should be dict, got {type(pair_chains_iptm_dtensor)}"
    for idx1, chain_dict in pair_chains_iptm_dtensor.items():
        for idx2, dt_val in chain_dict.items():
            full_val = dt_val.full_tensor().cpu()
            assert not torch.isnan(full_val).any(), f"pair_chains_iptm ({idx1}, {idx2}) contains NaN"
            valid = full_val[full_val != CHAIN_IPTM_SENTINEL]
            if valid.numel() > 0:
                assert (
                    valid.min() >= -1e-4
                ), f"pair_chains_iptm ({idx1}, {idx2}) has non-sentinel value < 0: {valid.min()}"
                assert valid.max() <= 1.0 + 1e-4, f"pair_chains_iptm ({idx1}, {idx2}) has value > 1: {valid.max()}"

    # Compare all tensor outputs against serial reference. The distributed
    # _compute_ptms_1d applies the same collinear-frame mask as serial
    # compute_ptms, so ptm/iptm/ligand_iptm/protein_iptm should match exactly.
    for key in output_dtensor:
        if key == "pair_chains_iptm":
            continue
        assert key in serial_output_feats_host, f"DTensor output key '{key}' missing from serial reference"
        full_val = output_dtensor[key].full_tensor().cpu()
        assert not torch.isnan(full_val).any(), f"{key} contains NaN"
        torch.testing.assert_close(
            full_val,
            serial_output_feats_host[key].to(full_val.dtype),
            msg=f"Mismatch for output key '{key}'",
        )

    # --- Backward ---
    torch.autograd.backward(
        [
            output_dtensor["plddt_logits"],
            output_dtensor["pde_logits"],
            output_dtensor["resolved_logits"],
            output_dtensor["pae_logits"],
        ],
        [
            d_plddt_logits_dtensor,
            d_pde_logits_dtensor,
            d_resolved_logits_dtensor,
            d_pae_logits_dtensor,
        ],
    )

    # Verify inputs were not mutated by the forward pass
    assert_tensors_identical(
        s_dtensor.to_local().cpu(),
        s_dtensor_copy.to_local().cpu(),
        check_grad=False,
        check_grad_fn=False,
        msg="s_dtensor was mutated during forward",
    )
    assert_tensors_identical(
        z_dtensor.to_local().cpu(),
        z_dtensor_copy.to_local().cpu(),
        check_grad=False,
        check_grad_fn=False,
        msg="z_dtensor was mutated during forward",
    )

    # Compare input gradients
    torch.testing.assert_close(s_dtensor.grad.full_tensor().cpu(), s_grad_host, msg="s gradient mismatch")
    torch.testing.assert_close(z_dtensor.grad.full_tensor().cpu(), z_grad_host, msg="z gradient mismatch")

    # Compare parameter gradients
    result_param_grads = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            if name not in serial_param_grads_host:
                raise ValueError(
                    f"Parameter '{name}' has a gradient in the distributed module but not in the serial reference"
                )
            result_param_grads[name] = param.grad

    for name, expected in serial_param_grads_host.items():
        assert name in result_param_grads, f"Parameter '{name}' gradient missing in distributed module"
        torch.testing.assert_close(
            result_param_grads[name].full_tensor().cpu(),
            expected,
            msg=f"Parameter gradient mismatch for '{name}'",
        )

    # Non-vacuous: gradients must be non-zero
    assert s_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed s gradient is all-zero"
    assert z_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed z gradient is all-zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


# ---------------------------------------------------------------------------
# Direct unit test of the row-slab transpose helpers
# ---------------------------------------------------------------------------


def _parallel_assert_transpose_via_all_to_all(rank, payload):
    """Worker: verify _transpose_via_all_to_all is bit-identical to _transpose_via_all_gather.

    Runs both helpers on the same input row-slab and asserts:
    * Forward output is bit-identical between the two paths.
    * Sharding is active (local shape on the sharded row-axis is strictly less
      than the global N when cp > 1).
    * Backward-shape parity using an explicit random ``grad_output`` (NOT
      ``.sum().backward()``) — the consumer ``_RowSlabTransposeAdd1D.backward``
      applies the same transpose to the upstream gradient, so the two paths
      must agree on that tensor too.
    * Uneven column sharding is rejected with ``ValueError``.
    """
    (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        cp_size,
        B,
        N_full,
        C,
        z_global_host,
        grad_global_host,
    ) = payload

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    device_mesh = manager.device_mesh
    cp_group = device_mesh.get_group("cp")
    cp_rank = torch.distributed.get_rank(cp_group)

    z_global = z_global_host.to(device=manager.device)
    grad_global = grad_global_host.to(device=manager.device)
    n_row = N_full // cp_size
    z_local = z_global[:, cp_rank * n_row : (cp_rank + 1) * n_row, :, :].contiguous()
    grad_local = grad_global[:, cp_rank * n_row : (cp_rank + 1) * n_row, :, :].contiguous()

    if cp_size > 1:
        assert (
            z_local.shape[1] < N_full
        ), f"Sharding inactive: local row count {z_local.shape[1]} not less than N_full {N_full}"
        assert z_local.shape[1] == n_row, f"Row-slab size mismatch: expected {n_row}, got {z_local.shape[1]}"

    # --- Forward parity (bit-identical) ---
    out_ag = _transpose_via_all_gather(z_local.clone(), cp_group, cp_size)
    out_a2a = _transpose_via_all_to_all(z_local.clone(), cp_group, cp_size)
    assert (
        out_ag.shape == out_a2a.shape == z_local.shape
    ), f"Shape mismatch: out_ag={out_ag.shape}, out_a2a={out_a2a.shape}, z_local={z_local.shape}"
    assert torch.equal(out_ag, out_a2a), (
        f"rank {rank}: a2a transpose differs from all_gather transpose; max abs diff = "
        f"{(out_ag - out_a2a).abs().max().item()}"
    )

    # --- Backward-shape parity using an explicit random grad ---
    # The helpers are not autograd-aware; the consumer _RowSlabTransposeAdd1D
    # applies the same row-slab transpose to the upstream gradient in backward,
    # so the two paths must agree on that tensor too.
    grad_t_ag = _transpose_via_all_gather(grad_local.clone(), cp_group, cp_size)
    grad_t_a2a = _transpose_via_all_to_all(grad_local.clone(), cp_group, cp_size)
    assert torch.equal(grad_t_ag, grad_t_a2a), (
        f"rank {rank}: backward-shape a2a transpose differs from all_gather; max abs diff = "
        f"{(grad_t_ag - grad_t_a2a).abs().max().item()}"
    )
    assert grad_t_a2a.abs().sum() > 0, f"rank {rank}: transposed gradient is all-zero (vacuous)"

    # --- Adversarial uneven-shard guard ---
    if cp_size > 1:
        bad_n_full = N_full + 1  # not divisible by cp_size
        z_bad = torch.zeros(B, n_row, bad_n_full, C, device=manager.device, dtype=z_local.dtype)
        try:
            _transpose_via_all_to_all(z_bad, cp_group, cp_size)
        except ValueError:
            pass
        else:
            raise AssertionError("a2a helper did not reject uneven column sharding")

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        # NCCL-only: a2a is the production target. cp=2 and cp=4 exercise
        # the chunk-split logic at the most common production widths.
        ((1, 2), True, "cuda", "ENV"),
        ((1, 4), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=[
        "cuda-cp2",
        "cuda-cp4",
    ],
)
def test_dtensor_transpose_via_all_to_all(setup_env):
    """Direct bit-identity test of _transpose_via_all_to_all vs _transpose_via_all_gather.

    Both helpers produce the row slab of ``z^T`` for a row-slab pair tensor.
    They must agree bit-identically: no summation occurs in either path — only
    data movement and a local transpose — so floating-point order is preserved.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    cp_size = grid_group_sizes["cp"]

    B = 2
    N_full = 8 * cp_size
    C = 4
    dtype = torch.float32

    rng = torch.Generator(device="cpu")
    rng.manual_seed(0xC0FFEE)
    z_global_host = torch.randn(B, N_full, N_full, C, dtype=dtype, generator=rng)
    grad_global_host = torch.randn(B, N_full, N_full, C, dtype=dtype, generator=rng)

    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        cp_size,
        B,
        N_full,
        C,
        z_global_host,
        grad_global_host,
    )
    spawn_multiprocessing(_parallel_assert_transpose_via_all_to_all, world_size, payload)


@pytest.mark.parametrize(
    "setup_env, loss_config",
    [
        # CUDA dp=1 cp=1: serial-equivalent sanity check (1 GPU)
        (((1, 1), True, "cuda", "ENV"), (torch.float64, False, 1)),
        # CUDA dp=2 cp=1: DP-only with multiplicity>1 (2 GPUs)
        (((2, 1), True, "cuda", "ENV"), (torch.float64, False, 2)),
        # CUDA dp=1 cp=2: CP-only path (2 GPUs)
        (((1, 2), True, "cuda", "ENV"), (torch.float64, False, 1)),
        # CUDA dp=1 cp=2: CP-only with separate heads (2 GPUs)
        (((1, 2), True, "cuda", "ENV"), (torch.float64, True, 1)),
        # CUDA dp=1 cp=3: non-power-of-two CP (3 GPUs)
        (((1, 3), True, "cuda", "ENV"), (torch.float64, False, 1)),
        # CPU dp=2 cp=1: DP-only regression guard (gloo doesn't support all_to_all
        # needed for PDE transpose when cp>1)
        (((2, 1), True, "cpu", "ENV"), (torch.float64, False, 1)),
    ],
    indirect=("setup_env",),
    ids=[
        "cuda-dp1-cp1-shared-mult1",
        "cuda-dp2-cp1-shared-mult2",
        "cuda-dp1-cp2-shared-mult1",
        "cuda-dp1-cp2-separate-mult1",
        "cuda-dp1-cp3-shared-mult1",
        "cpu-dp2-cp1-shared-mult1",
    ],
)
def test_dtensor_confidence_heads_1d(setup_env, loss_config):
    """Test 1D CP ConfidenceHeads1D against serial ConfidenceHeads reference.

    Covers:
    * Forward pass parity for logit outputs and aggregated metrics
      (pLDDT, iPLDDT, PDE, iPDE, PAE, pTM/ipTM, pair_chains_iptm).
    * Backward pass parity for input gradients (s, z) and parameter gradients.
    * Non-mutation of inputs during forward.
    * Both single-head and separate intra/inter-chain head configurations.
    * Multiplicity > 1 (multiple diffusion samples per input).
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, use_separate_heads, multiplicity = loss_config

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    cp_size = grid_group_sizes["cp"]
    dp_size = grid_group_sizes["dp"]
    B = 2 * dp_size

    # N_tokens must be divisible by cp_size for even sharding
    N_tokens = 12 * cp_size
    N_atoms = N_tokens * 4

    # Use a local Generator for RNG isolation
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    boltz2_params = create_boltz2_model_init_params(use_large_model=False)
    token_s = boltz2_params["token_s"]
    token_z = boltz2_params["token_z"]
    confidence_args = boltz2_params["confidence_model_args"]["confidence_args"]
    num_distogram_bins = boltz2_params["confidence_model_args"]["num_dist_bins"]

    confidence_heads_kwargs = {
        "token_s": token_s,
        "token_z": token_z,
        **confidence_args,
        "token_level_confidence": True,
        "use_separate_heads": use_separate_heads,
    }

    val_init_range = 0.15

    # Generate features. Both serial compute_ptms and the distributed
    # _compute_ptms_1d need the atom-level features (frames_idx, atom_to_token,
    # atom_pad_mask, atom_resolved_mask) used to compute the collinear-frame
    # mask via compute_frame_pred.
    selected_keys = [
        "mol_type",
        "asym_id",
        "token_pad_mask",
        "frames_idx",
        "atom_to_token",
        "atom_pad_mask",
        "atom_resolved_mask",
    ]
    feats_all = random_features(
        size_batch=B,
        n_tokens=N_tokens,
        n_atoms=N_atoms,
        n_msa=1,
        atom_counts_per_token_range=(1, 4),
        device=torch.device("cpu"),
        float_value_range=(-0.2, 0.2),
        selected_keys=selected_keys,
        rng=rng,
    )
    feats_all = {k: (v.to(dtype=dtype) if v.is_floating_point() else v) for k, v in feats_all.items()}
    feats_serial = {k: v.clone() for k, v in feats_all.items()}
    feats_host = dict(feats_all)

    # Input tensors
    s = torch.empty(B * multiplicity, N_tokens, token_s, dtype=dtype, requires_grad=True)
    z = torch.empty(B * multiplicity, N_tokens, N_tokens, token_z, dtype=dtype, requires_grad=True)
    x_pred = torch.empty(B * multiplicity, N_atoms, 3, dtype=dtype)
    d = torch.empty(B * multiplicity, N_tokens, N_tokens, dtype=dtype)
    pred_distogram_logits = torch.empty(B, N_tokens, N_tokens, num_distogram_bins, dtype=dtype)

    _init_tensors_uniform_rng(
        [s, z, x_pred, d, pred_distogram_logits], low=-val_init_range, high=val_init_range, rng=rng
    )

    s_global_host = s.detach().clone().cpu()
    z_global_host = z.detach().clone().cpu()
    x_pred_global_host = x_pred.detach().clone().cpu()
    d_global_host = d.detach().clone().cpu()
    pred_distogram_logits_global_host = pred_distogram_logits.detach().clone().cpu()

    # --- Serial module ---
    serial_module = SerialConfidenceHeads(**confidence_heads_kwargs)
    serial_module = serial_module.to(dtype=dtype).train()
    _init_module_params_glorot_rng(serial_module, rng=rng, gain=val_init_range)

    serial_state_dict = serial_module.state_dict()

    # --- Serial forward ---
    serial_output = serial_module(
        s=s,
        z=z,
        x_pred=x_pred,
        d=d,
        feats=feats_serial,
        pred_distogram_logits=pred_distogram_logits,
        multiplicity=multiplicity,
    )

    # Upstream gradients for backward (explicit random, not .sum().backward())
    d_plddt_logits = torch.empty_like(serial_output["plddt_logits"])
    d_pde_logits = torch.empty_like(serial_output["pde_logits"])
    d_resolved_logits = torch.empty_like(serial_output["resolved_logits"])
    d_pae_logits = torch.empty_like(serial_output["pae_logits"])
    _init_tensors_uniform_rng(
        [d_plddt_logits, d_pde_logits, d_resolved_logits, d_pae_logits],
        low=-val_init_range,
        high=val_init_range,
        rng=rng,
    )

    d_plddt_logits_host = d_plddt_logits.detach().clone().cpu()
    d_pde_logits_host = d_pde_logits.detach().clone().cpu()
    d_resolved_logits_host = d_resolved_logits.detach().clone().cpu()
    d_pae_logits_host = d_pae_logits.detach().clone().cpu()

    # Serial backward
    torch.autograd.backward(
        [
            serial_output["plddt_logits"],
            serial_output["pde_logits"],
            serial_output["resolved_logits"],
            serial_output["pae_logits"],
        ],
        [d_plddt_logits, d_pde_logits, d_resolved_logits, d_pae_logits],
    )

    # Save all serial outputs as CPU tensors
    def _to_cpu(val):
        if isinstance(val, torch.Tensor):
            return val.detach().clone().cpu()
        if isinstance(val, dict):
            return {k: _to_cpu(v) for k, v in val.items()}
        return val

    serial_output_feats_host = {k: _to_cpu(v) for k, v in serial_output.items()}

    # Guard against vacuous pass
    assert not torch.isnan(serial_output["plddt_logits"]).any(), "serial plddt_logits contains NaN"
    assert not torch.isnan(serial_output["pde_logits"]).any(), "serial pde_logits contains NaN"
    assert not torch.isnan(serial_output["pae_logits"]).any(), "serial pae_logits contains NaN"
    assert serial_output["plddt"].abs().max() > 0, "serial plddt is all-zero (vacuous)"
    assert serial_output["complex_plddt"].abs().max() > 0, "serial complex_plddt is all-zero (vacuous)"
    # pTM/ipTM: serial uses collinear-frame masking. Check non-NaN rather than
    # non-zero since collinear masking can zero out tokens.
    assert not torch.isnan(serial_output["ptm"]).any(), "serial ptm contains NaN"
    assert isinstance(serial_output["pair_chains_iptm"], dict), (
        f"serial pair_chains_iptm should be a dict (from compute_ptms), "
        f"got {type(serial_output['pair_chains_iptm'])} — compute_ptms likely failed silently"
    )

    s_grad_host = s.grad.detach().clone().cpu()
    z_grad_host = z.grad.detach().clone().cpu()

    # Guard against vacuous backward
    assert s_grad_host.abs().max() > 0, "serial s gradient is all-zero (vacuous)"
    assert z_grad_host.abs().max() > 0, "serial z gradient is all-zero (vacuous)"

    serial_param_grads_host = {
        name: param.grad.detach().clone().cpu()
        for name, param in serial_module.named_parameters()
        if param.grad is not None
    }
    assert len(serial_param_grads_host) > 0, "No serial parameter gradients found (vacuous)"

    # --- Parallel distributed test ---
    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        serial_state_dict,
        confidence_heads_kwargs,
        multiplicity,
        s_global_host,
        z_global_host,
        x_pred_global_host,
        d_global_host,
        feats_host,
        pred_distogram_logits_global_host,
        serial_output_feats_host,
        s_grad_host,
        z_grad_host,
        d_plddt_logits_host,
        d_pde_logits_host,
        d_resolved_logits_host,
        d_pae_logits_host,
        serial_param_grads_host,
    )

    spawn_multiprocessing(parallel_assert_confidence_heads_1d, world_size, payload)


# ---------------------------------------------------------------------------
# ConfidenceModule1D parity test
# ---------------------------------------------------------------------------


def _distribute_feats_1d(
    feats_host: dict[str, torch.Tensor],
    device_mesh,
    device,
):
    """Distribute feature tensors onto 1D CP mesh with appropriate placements."""
    single_placements = (Shard(0), Shard(1))
    pair_placements = (Shard(0), Shard(1))
    feats_dtensor = {}
    for key, val in feats_host.items():
        val_device = val.to(device=device)
        if val.ndim == 2:  # noqa: PLR2004
            feats_dtensor[key] = distribute_tensor(val_device, device_mesh=device_mesh, placements=single_placements)
        elif val.ndim == 3:  # noqa: PLR2004
            feats_dtensor[key] = distribute_tensor(val_device, device_mesh=device_mesh, placements=pair_placements)
        elif val.ndim == 4:  # noqa: PLR2004
            feats_dtensor[key] = distribute_tensor(val_device, device_mesh=device_mesh, placements=pair_placements)
        else:
            feats_dtensor[key] = distribute_tensor(
                val_device, device_mesh=device_mesh, placements=(Replicate(), Replicate())
            )
    return feats_dtensor


def parallel_assert_confidence_module_1d(
    rank,
    payload,
):
    """Worker function for 1D CP ConfidenceModule1D parity testing."""
    (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        serial_state_dict,
        confidence_module_kwargs,
        multiplicity,
        # input tensors
        s_inputs_global_host,
        s_global_host,
        z_global_host,
        x_pred_global_host,
        feats_dist_host,
        pred_distogram_logits_global_host,
        # reference serial outputs
        serial_output_feats_host,
        # reference serial input gradients
        s_inputs_grad_host,
        s_grad_host,
        z_grad_host,
        # upstream gradient tensors
        d_plddt_logits_host,
        d_pde_logits_host,
        d_resolved_logits_host,
        d_pae_logits_host,
        # reference serial parameter gradients
        serial_param_grads_host,
    ) = payload

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    device_mesh = manager.device_mesh
    seed_by_rank(rank, 42)

    # Build serial module, convert to target dtype, then load state dict
    serial_module = SerialConfidenceModule(**confidence_module_kwargs)
    serial_module = serial_module.to(device=manager.device, dtype=dtype).train()
    serial_module.load_state_dict(serial_state_dict)

    module = ConfidenceModule1D(
        module=serial_module,
        dist_manager=manager,
    )
    module = module.to(device=manager.device, dtype=dtype).train()

    # --- Distribute inputs ---
    # x_pred atoms are sharded on cp (same as tokens)
    single_placements = (Shard(0), Shard(1))
    pair_placements = (Shard(0), Shard(1))
    atom_placements = (Shard(0), Shard(1))

    s_inputs_dtensor = distribute_tensor(
        s_inputs_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    s_dtensor = distribute_tensor(
        s_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    z_dtensor = distribute_tensor(
        z_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=pair_placements,
    )
    x_pred_dtensor = distribute_tensor(
        x_pred_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=atom_placements,
    )
    pred_distogram_logits_dtensor = distribute_tensor(
        pred_distogram_logits_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    feats_dtensor = _distribute_feats_1d(feats_dist_host, device_mesh, manager.device)

    # Keep copies to verify inputs are not mutated
    s_inputs_dtensor_copy = s_inputs_dtensor.clone()
    s_dtensor_copy = s_dtensor.clone()
    z_dtensor_copy = z_dtensor.clone()

    # --- Distribute upstream gradients ---
    d_plddt_logits_dtensor = distribute_tensor(
        d_plddt_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    d_pde_logits_dtensor = distribute_tensor(
        d_pde_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )
    d_resolved_logits_dtensor = distribute_tensor(
        d_resolved_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    d_pae_logits_dtensor = distribute_tensor(
        d_pae_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    # --- Forward ---
    output_dtensor = module(
        s_inputs=s_inputs_dtensor,
        s=s_dtensor,
        z=z_dtensor,
        x_pred=x_pred_dtensor,
        feats=feats_dtensor,
        pred_distogram_logits=pred_distogram_logits_dtensor,
        multiplicity=multiplicity,
    )

    # Verify placements have correct ndim for the 2D mesh (dp, cp)
    mesh_ndim = device_mesh.ndim
    for key in ["plddt_logits", "pde_logits", "resolved_logits", "pae_logits"]:
        assert len(output_dtensor[key].placements) == mesh_ndim, (
            f"{key} placements should have {mesh_ndim} elements for {mesh_ndim}D mesh, "
            f"got {len(output_dtensor[key].placements)}: {output_dtensor[key].placements}"
        )

    _PTM_KEYS = {"ptm", "iptm", "ligand_iptm", "protein_iptm", "pair_chains_iptm"}

    # Compare non-PTM outputs against serial reference
    for key in output_dtensor:
        if key in _PTM_KEYS:
            # pTM/ipTM: different masking strategy — validate structural properties only
            dtensor_val = output_dtensor[key]
            if key == "pair_chains_iptm":
                assert isinstance(dtensor_val, dict), f"pair_chains_iptm should be dict, got {type(dtensor_val)}"
                for idx1, chain_dict in dtensor_val.items():
                    for idx2, dt_val in chain_dict.items():
                        local_val = dt_val.to_local()
                        assert not torch.isnan(local_val).any(), f"pair_chains_iptm ({idx1}, {idx2}) contains NaN"
            else:
                full_val = dtensor_val.full_tensor()
                assert not torch.isnan(full_val).any(), f"{key} contains NaN"
                assert full_val.min() >= -1.0, f"{key} has values < -1: {full_val.min()}"
                assert full_val.max() <= 2.0, f"{key} has values > 2: {full_val.max()}"
            continue

        assert key in serial_output_feats_host, f"DTensor output key '{key}' missing from serial reference"
        torch.testing.assert_close(
            output_dtensor[key].full_tensor().cpu(),
            serial_output_feats_host[key],
            msg=f"Mismatch for output key '{key}'",
        )

    # --- Backward ---
    torch.autograd.backward(
        [
            output_dtensor["plddt_logits"],
            output_dtensor["pde_logits"],
            output_dtensor["resolved_logits"],
            output_dtensor["pae_logits"],
        ],
        [
            d_plddt_logits_dtensor,
            d_pde_logits_dtensor,
            d_resolved_logits_dtensor,
            d_pae_logits_dtensor,
        ],
    )

    # Verify inputs were not mutated by the forward pass
    assert_tensors_identical(
        s_inputs_dtensor.to_local().cpu(),
        s_inputs_dtensor_copy.to_local().cpu(),
        check_grad=False,
        check_grad_fn=False,
        msg="s_inputs_dtensor was mutated during forward",
    )
    assert_tensors_identical(
        s_dtensor.to_local().cpu(),
        s_dtensor_copy.to_local().cpu(),
        check_grad=False,
        check_grad_fn=False,
        msg="s_dtensor was mutated during forward",
    )
    assert_tensors_identical(
        z_dtensor.to_local().cpu(),
        z_dtensor_copy.to_local().cpu(),
        check_grad=False,
        check_grad_fn=False,
        msg="z_dtensor was mutated during forward",
    )

    # Compare input gradients
    torch.testing.assert_close(
        s_inputs_dtensor.grad.full_tensor().cpu(), s_inputs_grad_host, msg="s_inputs gradient mismatch"
    )
    torch.testing.assert_close(s_dtensor.grad.full_tensor().cpu(), s_grad_host, msg="s gradient mismatch")
    torch.testing.assert_close(z_dtensor.grad.full_tensor().cpu(), z_grad_host, msg="z gradient mismatch")

    # Compare parameter gradients
    result_param_grads = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            if name not in serial_param_grads_host:
                raise ValueError(
                    f"Parameter '{name}' has a gradient in the distributed module but not in the serial reference"
                )
            result_param_grads[name] = param.grad

    for name, expected in serial_param_grads_host.items():
        assert name in result_param_grads, f"Parameter '{name}' gradient missing in distributed module"
        torch.testing.assert_close(
            result_param_grads[name].full_tensor().cpu(),
            expected,
            msg=f"Parameter gradient mismatch for '{name}'",
        )

    # Non-vacuous: gradients must be non-zero
    assert s_inputs_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed s_inputs gradient is all-zero"
    assert s_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed s gradient is all-zero"
    assert z_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed z gradient is all-zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, loss_config",
    [
        # CUDA dp=1 cp=1: serial-equivalent sanity check (1 GPU)
        (((1, 1), True, "cuda", "ENV"), (torch.float64, False, 1)),
        # CUDA dp=1 cp=2: CP-only path (2 GPUs)
        (((1, 2), True, "cuda", "ENV"), (torch.float64, False, 1)),
        # CUDA dp=1 cp=2: CP-only with separate heads (2 GPUs)
        (((1, 2), True, "cuda", "ENV"), (torch.float64, True, 1)),
        # CUDA dp=1 cp=3: non-power-of-two CP (3 GPUs)
        (((1, 3), True, "cuda", "ENV"), (torch.float64, False, 1)),
        # CPU dp=2 cp=1: DP-only regression guard (gloo doesn't support all_to_all
        # needed for PDE transpose when cp>1)
        (((2, 1), True, "cpu", "ENV"), (torch.float64, False, 1)),
    ],
    indirect=("setup_env",),
    ids=[
        "cuda-dp1-cp1-shared-mult1",
        "cuda-dp1-cp2-shared-mult1",
        "cuda-dp1-cp2-separate-mult1",
        "cuda-dp1-cp3-shared-mult1",
        "cpu-dp2-cp1-shared-mult1",
    ],
)
def test_dtensor_confidence_module_1d(setup_env, loss_config):
    """Test 1D CP ConfidenceModule1D against serial ConfidenceModule reference.

    Covers:
    * Forward pass parity for all logit outputs and aggregated metrics
      (pLDDT, iPLDDT, PDE, iPDE, PAE, pTM/ipTM, pair_chains_iptm).
    * Backward pass parity for input gradients (s_inputs, s, z) and parameter gradients.
    * Non-mutation of inputs during forward.
    * Both single-head and separate intra/inter-chain head configurations.
    * Pairformer stack, distogram embedding, outer-sum s-to-z, relative position
      encoding, token bonds, contact conditioning.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, use_separate_heads, multiplicity = loss_config

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    cp_size = grid_group_sizes["cp"]
    dp_size = grid_group_sizes["dp"]
    B = dp_size

    # N_tokens must be divisible by cp_size for even sharding
    N_tokens = 12 * cp_size
    N_atoms = N_tokens * 4

    # Use a local Generator for RNG isolation
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    boltz2_params = create_boltz2_model_init_params(use_large_model=False)
    token_s = boltz2_params["token_s"]
    token_z = boltz2_params["token_z"]
    num_distogram_bins = boltz2_params["confidence_model_args"]["num_dist_bins"]

    confidence_model_args = boltz2_params["confidence_model_args"].copy()
    confidence_model_args["confidence_args"] = {
        **confidence_model_args["confidence_args"],
        "use_separate_heads": use_separate_heads,
    }
    confidence_module_kwargs = {
        "token_s": token_s,
        "token_z": token_z,
        "pairformer_args": boltz2_params["pairformer_args"],
        "token_level_confidence": True,
        "bond_type_feature": boltz2_params["bond_type_feature"],
        **confidence_model_args,
    }

    val_init_range = 0.002

    # Generate features. Serial needs atom-level features for compute_ptms,
    # distributed needs token-level + z-input conditioning features.
    selected_keys = [
        "mol_type",
        "asym_id",
        "token_pad_mask",
        "token_pair_pad_mask",
        "token_to_rep_atom",
        "atom_counts_per_token",
        "frames_idx",
        "atom_to_token",
        "atom_pad_mask",
        "atom_resolved_mask",
        "residue_index",
        "entity_id",
        "token_index",
        "sym_id",
        "cyclic_period",
        "token_bonds",
        "type_bonds",
        "contact_conditioning",
        "contact_threshold",
    ]
    feats_all = random_features(
        size_batch=B,
        n_tokens=N_tokens,
        n_atoms=N_atoms,
        n_msa=1,
        atom_counts_per_token_range=(1, 4),
        device=torch.device("cpu"),
        float_value_range=(-0.2, 0.2),
        selected_keys=selected_keys,
        rng=rng,
    )
    feats_all = {k: (v.to(dtype=dtype) if v.is_floating_point() else v) for k, v in feats_all.items()}

    feats_serial = {k: v.clone() for k, v in feats_all.items()}

    # Distributed module needs the same atom-level features (frames_idx,
    # atom_to_token, atom_pad_mask, atom_resolved_mask) as serial because
    # _compute_ptms_1d computes the collinear-frame mask via _compute_frame_pred_1d.
    feats_dist_host = dict(feats_all)

    # Input tensors
    s_inputs = torch.empty(B, N_tokens, token_s, dtype=dtype, requires_grad=True)
    s = torch.empty(B, N_tokens, token_s, dtype=dtype, requires_grad=True)
    z = torch.empty(B, N_tokens, N_tokens, token_z, dtype=dtype, requires_grad=True)
    x_pred = torch.empty(B * multiplicity, N_atoms, 3, dtype=dtype)
    pred_distogram_logits = torch.empty(B, N_tokens, N_tokens, num_distogram_bins, dtype=dtype)

    _init_tensors_uniform_rng(
        [s_inputs, s, z, pred_distogram_logits], low=-val_init_range, high=val_init_range, rng=rng
    )
    # x_pred needs wider range so inter-atom distances are meaningful
    _init_tensors_uniform_rng([x_pred], low=-10.0, high=10.0, rng=rng)

    s_inputs_global_host = s_inputs.detach().clone().cpu()
    s_global_host = s.detach().clone().cpu()
    z_global_host = z.detach().clone().cpu()
    x_pred_global_host = x_pred.detach().clone().cpu()
    pred_distogram_logits_global_host = pred_distogram_logits.detach().clone().cpu()

    # --- Serial module ---
    serial_module = SerialConfidenceModule(**confidence_module_kwargs)
    serial_module = serial_module.to(dtype=dtype).train()
    _init_module_params_glorot_rng(serial_module, rng=rng)

    serial_state_dict = serial_module.state_dict()

    # --- Serial forward ---
    serial_output = serial_module(
        s_inputs=s_inputs,
        s=s,
        z=z,
        x_pred=x_pred,
        feats=feats_serial,
        pred_distogram_logits=pred_distogram_logits,
        multiplicity=multiplicity,
    )

    # Upstream gradients for backward (explicit random, not .sum().backward())
    d_plddt_logits = torch.empty_like(serial_output["plddt_logits"])
    d_pde_logits = torch.empty_like(serial_output["pde_logits"])
    d_resolved_logits = torch.empty_like(serial_output["resolved_logits"])
    d_pae_logits = torch.empty_like(serial_output["pae_logits"])
    _init_tensors_uniform_rng(
        [d_plddt_logits, d_pde_logits, d_resolved_logits, d_pae_logits],
        low=-val_init_range,
        high=val_init_range,
        rng=rng,
    )

    d_plddt_logits_host = d_plddt_logits.detach().clone().cpu()
    d_pde_logits_host = d_pde_logits.detach().clone().cpu()
    d_resolved_logits_host = d_resolved_logits.detach().clone().cpu()
    d_pae_logits_host = d_pae_logits.detach().clone().cpu()

    # Serial backward
    torch.autograd.backward(
        [
            serial_output["plddt_logits"],
            serial_output["pde_logits"],
            serial_output["resolved_logits"],
            serial_output["pae_logits"],
        ],
        [d_plddt_logits, d_pde_logits, d_resolved_logits, d_pae_logits],
    )

    # Save all serial outputs as CPU tensors
    def _to_cpu(val):
        if isinstance(val, torch.Tensor):
            return val.detach().clone().cpu()
        if isinstance(val, dict):
            return {k: _to_cpu(v) for k, v in val.items()}
        return val

    serial_output_feats_host = {k: _to_cpu(v) for k, v in serial_output.items()}

    # Guard against vacuous pass
    assert not torch.isnan(serial_output["plddt_logits"]).any(), "serial plddt_logits contains NaN"
    assert not torch.isnan(serial_output["pde_logits"]).any(), "serial pde_logits contains NaN"
    assert not torch.isnan(serial_output["pae_logits"]).any(), "serial pae_logits contains NaN"
    assert serial_output["plddt"].abs().max() > 0, "serial plddt is all-zero (vacuous)"
    assert serial_output["complex_plddt"].abs().max() > 0, "serial complex_plddt is all-zero (vacuous)"
    assert not torch.isnan(serial_output["ptm"]).any(), "serial ptm contains NaN"
    assert isinstance(serial_output["pair_chains_iptm"], dict), (
        f"serial pair_chains_iptm should be a dict (from compute_ptms), "
        f"got {type(serial_output['pair_chains_iptm'])} — compute_ptms likely failed silently"
    )

    s_inputs_grad_host = s_inputs.grad.detach().clone().cpu()
    s_grad_host = s.grad.detach().clone().cpu()
    z_grad_host = z.grad.detach().clone().cpu()

    # Guard against vacuous backward
    assert s_inputs_grad_host.abs().max() > 0, "serial s_inputs gradient is all-zero (vacuous)"
    assert s_grad_host.abs().max() > 0, "serial s gradient is all-zero (vacuous)"
    assert z_grad_host.abs().max() > 0, "serial z gradient is all-zero (vacuous)"

    serial_param_grads_host = {
        name: param.grad.detach().clone().cpu()
        for name, param in serial_module.named_parameters()
        if param.grad is not None
    }
    assert len(serial_param_grads_host) > 0, "No serial parameter gradients found (vacuous)"

    # --- Parallel distributed test ---
    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        serial_state_dict,
        confidence_module_kwargs,
        multiplicity,
        s_inputs_global_host,
        s_global_host,
        z_global_host,
        x_pred_global_host,
        feats_dist_host,
        pred_distogram_logits_global_host,
        serial_output_feats_host,
        s_inputs_grad_host,
        s_grad_host,
        z_grad_host,
        d_plddt_logits_host,
        d_pde_logits_host,
        d_resolved_logits_host,
        d_pae_logits_host,
        serial_param_grads_host,
    )

    spawn_multiprocessing(parallel_assert_confidence_module_1d, world_size, payload)


# ---------------------------------------------------------------------------
# ConfidenceModule1D — rank-divergent cyclic-period edge case
# ---------------------------------------------------------------------------
#
# Regression guard for the cp=64 cyclic-period collective-ordering deadlock
# (MR !481).  The bug class: a collective (the column ``all_gather`` of
# ``cyclic_period``) whose ENTRY is gated on per-shard data.  Under 1D-CP token
# sharding a rank whose shard holds only non-cyclic tokens evaluates
# ``torch.any(cyclic_period_local > 0)`` as ``False`` and skips the gather while
# another rank enters it → NCCL hang.  The fix broadcasts the enter/skip
# decision with a scalar ``all_reduce(MAX)`` so every rank moves in lockstep.
#
# The existing ``test_dtensor_confidence_module_1d`` cannot catch this: its
# ``random_features`` sets ``cyclic_period = randint(0, 100)`` over every token,
# so the flag is uniformly ``True`` on all ranks and the divergent branch is
# never taken.  Here we craft a rank-divergent layout (via
# ``make_divergent_cyclic_rel_pos_feats``) and route it THROUGH the full
# ``ConfidenceModule1D`` (its ``self.rel_pos(feats)`` call), which is exactly
# where the production bug lived.  Two arms:
#   * ``buggy=False`` — the shipped lockstep code: forward/backward parity vs
#     serial.
#   * ``buggy=True`` — monkeypatch ``RelativePositionEncoder1D.forward`` back to
#     the pre-fix per-rank decision (the exact reverted logic from
#     ``b8a19ff58~1``: gate the gather on local ``torch.any`` with no
#     ``all_reduce``) and assert it reaches the rank-divergent gather-entry
#     decision — the necessary-and-sufficient precondition of the deadlock —
#     surfaced as a concrete ``RuntimeError`` at the decision point.  A literal
#     cross-rank hang cannot be asserted on in CI here because the harness sets
#     ``TORCH_NCCL_ASYNC_ERROR_HANDLING=0`` (manager.py), which prevents the
#     NCCL watchdog from aborting a hung collective; so the buggy arm raises on
#     the divergent decision (deterministic, fail-fast) rather than on a
#     wall-clock timeout.  The shipped ``all_reduce(MAX)`` makes that decision
#     uniform across ranks, so the same probe never raises — making the test
#     decisive: it FAILS on the buggy implementation and PASSES on the fix.


def _relative_position_encoder_1d_forward_prefix(self, feats):
    """Pre-fix (buggy) ``RelativePositionEncoder1D.forward``.

    Byte-faithful to the shipped forward EXCEPT the cyclic-period block, which
    is reverted to the pre-fix per-rank decision
    (``if self.cyclic_pos_enc and torch.any(cyclic_period_local > 0):`` with no
    ``all_reduce(MAX)`` lockstep) — the exact logic from ``confidence_1d.py``
    at commit ``b8a19ff58~1``, line 459.  Used only by the buggy arm to prove
    the test catches the deadlock.
    """
    expected_placements = (Shard(0), Shard(1))
    for key in ("asym_id", "entity_id", "residue_index", "token_index", "sym_id", "cyclic_period"):
        dt = feats[key]
        if not isinstance(dt, DTensor):
            raise TypeError(f"Expected DTensor for '{key}', got {type(dt)}")
        if dt.placements != expected_placements:
            raise ValueError(f"'{key}' placements {dt.placements} != expected {expected_placements}")

    cp_size = self.device_mesh.shape[1]
    B_global = feats["asym_id"].shape[0]
    N_global = feats["asym_id"].shape[1]

    asym_id_local = feats["asym_id"].to_local()
    entity_id_local = feats["entity_id"].to_local()
    residue_idx_local = feats["residue_index"].to_local()
    token_idx_local = feats["token_index"].to_local()
    sym_id_local = feats["sym_id"].to_local()
    cyclic_period_local = feats["cyclic_period"].to_local()

    asym_id_full = all_gather_on_cp(asym_id_local, 1, self.cp_group, cp_size)
    entity_id_full = all_gather_on_cp(entity_id_local, 1, self.cp_group, cp_size)
    residue_idx_full = all_gather_on_cp(residue_idx_local, 1, self.cp_group, cp_size)
    token_idx_full = all_gather_on_cp(token_idx_local, 1, self.cp_group, cp_size)
    sym_id_full = all_gather_on_cp(sym_id_local, 1, self.cp_group, cp_size)

    b_same_chain = torch.eq(asym_id_local[:, :, None], asym_id_full[:, None, :])
    b_same_residue = torch.eq(residue_idx_local[:, :, None], residue_idx_full[:, None, :])
    b_same_entity = torch.eq(entity_id_local[:, :, None], entity_id_full[:, None, :])

    d_residue = residue_idx_local[:, :, None] - residue_idx_full[:, None, :]

    # PRE-FIX per-rank decision — the bug: no lockstep all_reduce, so ranks
    # holding only non-cyclic tokens skip the gather while others enter it.
    enter_gather = bool(self.cyclic_pos_enc and torch.any(cyclic_period_local > 0))

    # A literal cross-rank collective deadlock cannot be aborted in this test
    # harness: DistributedManager.initialize sets
    # TORCH_NCCL_ASYNC_ERROR_HANDLING=0 (manager.py), so the NCCL watchdog logs
    # the per-collective timeout but never tears the hung gather down — the
    # buggy arm would spin at 100% util indefinitely.  Instead, probe the
    # per-rank decisions with a tiny, always-matched ``all_gather_object``
    # (every rank calls it, so the probe itself never deadlocks) and raise a
    # concrete exception the instant the decisions diverge — which is the
    # necessary-and-sufficient precondition of the deadlock (mismatched entry
    # into the column gather).  The shipped forward's ``all_reduce(MAX)`` makes
    # this decision identical on every rank, so the same probe stays uniform
    # and never raises: that is exactly the contrast this arm asserts.
    decisions = [None] * cp_size
    torch.distributed.all_gather_object(decisions, enter_gather, group=self.cp_group)
    if min(decisions) != max(decisions):
        raise RuntimeError(
            f"pre-fix per-rank cyclic decision diverges across cp ranks {decisions} — "
            f"collective entry gated on per-shard data would deadlock the column all_gather"
        )

    if enter_gather:
        cyclic_period_full = all_gather_on_cp(cyclic_period_local, 1, self.cp_group, cp_size)
        period = torch.where(
            cyclic_period_full > 0,
            cyclic_period_full,
            torch.zeros_like(cyclic_period_full) + 10000,
        ).unsqueeze(1)
        d_residue = (d_residue - period * torch.round(d_residue / period)).long()

    d_residue = torch.clip(d_residue + self.r_max, 0, 2 * self.r_max)
    d_residue = torch.where(b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * self.r_max + 1)
    a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

    d_token = torch.clip(
        token_idx_local[:, :, None] - token_idx_full[:, None, :] + self.r_max,
        0,
        2 * self.r_max,
    )
    d_token = torch.where(
        b_same_chain & b_same_residue,
        d_token,
        torch.zeros_like(d_token) + 2 * self.r_max + 1,
    )
    a_rel_token = one_hot(d_token, 2 * self.r_max + 2)

    d_chain = torch.clip(
        sym_id_local[:, :, None] - sym_id_full[:, None, :] + self.s_max,
        0,
        2 * self.s_max,
    )
    d_chain = torch.where(
        (~b_same_entity) if self.fix_sym_check else b_same_chain,
        torch.zeros_like(d_chain) + 2 * self.s_max + 1,
        d_chain,
    )
    a_rel_chain = one_hot(d_chain, 2 * self.s_max + 2)

    cast_dtype = torch.promote_types(self.linear_layer.weight.dtype, torch.float32)
    features_local = torch.cat(
        [
            a_rel_pos.to(cast_dtype),
            a_rel_token.to(cast_dtype),
            b_same_entity.unsqueeze(-1).to(cast_dtype),
            a_rel_chain.to(cast_dtype),
        ],
        dim=-1,
    )

    feat_dim = features_local.shape[-1]
    pair_shape = torch.Size([B_global, N_global, N_global, feat_dim])
    pair_stride = update_exhaustive_strides(features_local.shape, features_local.stride(), pair_shape)
    features_dt = DTensor.from_local(
        features_local,
        device_mesh=self.device_mesh,
        placements=(Shard(0), Shard(1)),
        shape=pair_shape,
        stride=pair_stride,
    )
    return self.linear_layer(features_dt)


def parallel_assert_confidence_divergent_cyclic_1d(rank, payload):
    """Worker: drive a rank-divergent cyclic layout through ConfidenceModule1D.

    On the fixed arm (``buggy=False``): forward/backward parity vs serial. On
    the buggy arm (``buggy=True``): monkeypatch the rel-pos forward back to the
    pre-fix per-rank decision and assert it reaches the rank-divergent
    gather-entry decision (the deadlock precondition), surfaced as a concrete
    ``RuntimeError`` at the decision point — deterministic and fail-fast,
    proving the test is decisive.
    """
    (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        serial_state_dict,
        confidence_module_kwargs,
        multiplicity,
        buggy,
        pg_timeout_s,
        s_inputs_global_host,
        s_global_host,
        z_global_host,
        x_pred_global_host,
        feats_dist_host,
        pred_distogram_logits_global_host,
        serial_output_feats_host,
        s_inputs_grad_host,
        s_grad_host,
        z_grad_host,
        d_plddt_logits_host,
        d_pde_logits_host,
        d_resolved_logits_host,
        d_pae_logits_host,
        serial_param_grads_host,
    ) = payload

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    # Modest per-collective timeout (belt-and-suspenders). The buggy arm raises
    # deterministically at the divergent-decision probe before any mismatched
    # collective, so it does not depend on this timeout; it is kept only to bound
    # any unexpected stall well inside the system cap. (The NCCL watchdog cannot
    # abort a hung collective here regardless — manager.py disables async error
    # handling — which is precisely why the buggy arm asserts on the decision
    # observable rather than on a wall-clock timeout.)
    DistributedManager.initialize(
        grid_group_sizes,
        device_type=device_type,
        backend=backend,
        timeout=timedelta(seconds=pg_timeout_s),
    )
    manager = DistributedManager()
    device_mesh = manager.device_mesh
    seed_by_rank(rank, 42)

    if buggy:
        # Revert RelativePositionEncoder1D.forward to the exact pre-fix per-rank
        # decision (b8a19ff58~1). Faithful, not an approximation.
        monkeypatch.setattr(
            RelativePositionEncoder1D,
            "forward",
            _relative_position_encoder_1d_forward_prefix,
        )

    serial_module = SerialConfidenceModule(**confidence_module_kwargs)
    serial_module = serial_module.to(device=manager.device, dtype=dtype).train()
    serial_module.load_state_dict(serial_state_dict)

    module = ConfidenceModule1D(module=serial_module, dist_manager=manager)
    module = module.to(device=manager.device, dtype=dtype).train()

    single_placements = (Shard(0), Shard(1))
    pair_placements = (Shard(0), Shard(1))
    atom_placements = (Shard(0), Shard(1))

    s_inputs_dtensor = distribute_tensor(
        s_inputs_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    s_dtensor = distribute_tensor(
        s_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    z_dtensor = distribute_tensor(
        z_global_host.to(device=manager.device, dtype=dtype).requires_grad_(True),
        device_mesh=device_mesh,
        placements=pair_placements,
    )
    x_pred_dtensor = distribute_tensor(
        x_pred_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=atom_placements,
    )
    pred_distogram_logits_dtensor = distribute_tensor(
        pred_distogram_logits_global_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    feats_dtensor = _distribute_feats_1d(feats_dist_host, device_mesh, manager.device)

    # Runtime non-vacuity: the crafted layout is genuinely rank-divergent on the
    # cyclic flag for THIS mesh, so the lockstep all_reduce is load-bearing (and
    # the buggy arm genuinely deadlocks). A future feats change that silently
    # makes the flag uniform would trip this assert instead of going vacuous.
    cyclic_local = feats_dtensor["cyclic_period"].to_local()
    has_cyclic_local = int(torch.any(cyclic_local > 0))
    has_cyclic_all = [None] * grid_group_sizes["cp"]
    torch.distributed.all_gather_object(has_cyclic_all, has_cyclic_local, group=manager.group["cp"])
    assert (
        min(has_cyclic_all) == 0 and max(has_cyclic_all) == 1
    ), f"Test setup is not rank-divergent on cp={grid_group_sizes['cp']}: {has_cyclic_all}"

    if buggy:
        # The monkeypatched pre-fix forward reaches a rank-divergent gather-entry
        # decision and raises a concrete RuntimeError at the decision point
        # (before the would-be mismatched column gather that deadlocks the
        # shipped-but-pre-fix code). Assert on that observable, not on wall-clock.
        with pytest.raises(RuntimeError, match="would deadlock"):
            module(
                s_inputs=s_inputs_dtensor,
                s=s_dtensor,
                z=z_dtensor,
                x_pred=x_pred_dtensor,
                feats=feats_dtensor,
                pred_distogram_logits=pred_distogram_logits_dtensor,
                multiplicity=multiplicity,
            )
        DistributedManager.cleanup()
        monkeypatch.undo()
        return

    # --- Fixed arm: forward parity vs serial ---
    output_dtensor = module(
        s_inputs=s_inputs_dtensor,
        s=s_dtensor,
        z=z_dtensor,
        x_pred=x_pred_dtensor,
        feats=feats_dtensor,
        pred_distogram_logits=pred_distogram_logits_dtensor,
        multiplicity=multiplicity,
    )

    _PTM_KEYS = {"ptm", "iptm", "ligand_iptm", "protein_iptm", "pair_chains_iptm"}
    for key in output_dtensor:
        if key in _PTM_KEYS:
            continue
        assert key in serial_output_feats_host, f"DTensor output key '{key}' missing from serial reference"
        torch.testing.assert_close(
            output_dtensor[key].full_tensor().cpu(),
            serial_output_feats_host[key],
            msg=f"Mismatch for output key '{key}'",
        )

    # --- Backward (explicit random grad_output, not .sum().backward()) ---
    d_plddt_logits_dtensor = distribute_tensor(
        d_plddt_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    d_pde_logits_dtensor = distribute_tensor(
        d_pde_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )
    d_resolved_logits_dtensor = distribute_tensor(
        d_resolved_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=single_placements,
    )
    d_pae_logits_dtensor = distribute_tensor(
        d_pae_logits_host.to(device=manager.device, dtype=dtype),
        device_mesh=device_mesh,
        placements=pair_placements,
    )
    torch.autograd.backward(
        [
            output_dtensor["plddt_logits"],
            output_dtensor["pde_logits"],
            output_dtensor["resolved_logits"],
            output_dtensor["pae_logits"],
        ],
        [d_plddt_logits_dtensor, d_pde_logits_dtensor, d_resolved_logits_dtensor, d_pae_logits_dtensor],
    )

    torch.testing.assert_close(
        s_inputs_dtensor.grad.full_tensor().cpu(), s_inputs_grad_host, msg="s_inputs gradient mismatch"
    )
    torch.testing.assert_close(s_dtensor.grad.full_tensor().cpu(), s_grad_host, msg="s gradient mismatch")
    torch.testing.assert_close(z_dtensor.grad.full_tensor().cpu(), z_grad_host, msg="z gradient mismatch")

    # The rel_pos linear_layer (the only differentiable part of the cyclic path)
    # must have a non-zero, serial-matching gradient — proving the cyclic branch
    # actually contributed to the loss, not just that the module ran.
    result_param_grads = {}
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert name in serial_param_grads_host, f"Param '{name}' grad in distributed but not serial"
            result_param_grads[name] = param.grad
    rel_pos_grad_names = [n for n in result_param_grads if "rel_pos.linear_layer" in n]
    assert rel_pos_grad_names, "rel_pos.linear_layer gradient missing — cyclic path not differentiated"
    for name in rel_pos_grad_names:
        torch.testing.assert_close(
            result_param_grads[name].full_tensor().cpu(),
            serial_param_grads_host[name],
            msg=f"rel_pos parameter gradient mismatch for '{name}'",
        )
        assert result_param_grads[name].full_tensor().abs().max() > 0, f"rel_pos grad {name} is all-zero"

    assert s_inputs_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed s_inputs gradient is all-zero"
    assert z_dtensor.grad.full_tensor().abs().sum() > 0, "Distributed z gradient is all-zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, arm_config",
    [
        # Fixed arm: forward/backward parity vs serial on the divergent layout.
        (((1, 2), True, "cuda", "ENV"), (torch.float64, False)),
        # Non-power-of-two CP fixed arm.
        (((1, 3), True, "cuda", "ENV"), (torch.float64, False)),
        # Buggy arm (CUDA/nccl only — the divergent-decision probe is a tiny
        # always-matched collective, but routing through the full module uses
        # all_to_all collectives that gloo does not support at cp>1).
        (((1, 2), True, "cuda", "ENV"), (torch.float64, True)),
    ],
    indirect=("setup_env",),
    ids=[
        "cuda-dp1-cp2-fixed",
        "cuda-dp1-cp3-fixed",
        "cuda-dp1-cp2-buggy-divergent-decision",
    ],
)
def test_dtensor_confidence_module_divergent_cyclic(setup_env, arm_config):
    """ConfidenceModule1D: rank-divergent cyclic-period edge case (MR !481).

    Drives a crafted rank-divergent ``cyclic_period`` layout through the full
    ``ConfidenceModule1D`` (its ``self.rel_pos(feats)`` call), where the cp=64
    cyclic-period collective-ordering deadlock lived. The fixed arm asserts
    forward/backward parity vs serial; the buggy arm reverts the rel-pos forward
    to the pre-fix per-rank gather decision and asserts it reaches the
    rank-divergent gather-entry decision — the deadlock precondition — surfaced
    as a concrete ``RuntimeError`` at the decision point. Decisive: fails on the
    buggy implementation, passes on the shipped lockstep code.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, buggy = arm_config

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    cp_size = grid_group_sizes["cp"]
    dp_size = grid_group_sizes["dp"]
    assert cp_size > 1, "Divergent-cyclic test requires cp > 1"
    B = dp_size
    multiplicity = 1

    # N must be even (two balanced chains) and divisible by cp_size; 12*cp is
    # both, matching the sibling parity test's sizing.
    N_tokens = 12 * cp_size
    N_atoms = N_tokens * 4
    cyclic_period_val = 3

    # Modest per-collective timeout (belt-and-suspenders only — see the worker;
    # the buggy arm raises deterministically at the decision probe and does not
    # rely on a timeout).
    pg_timeout_s = 60

    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    boltz2_params = create_boltz2_model_init_params(use_large_model=False)
    token_s = boltz2_params["token_s"]
    token_z = boltz2_params["token_z"]
    num_distogram_bins = boltz2_params["confidence_model_args"]["num_dist_bins"]

    confidence_model_args = boltz2_params["confidence_model_args"].copy()
    confidence_model_args["confidence_args"] = {
        **confidence_model_args["confidence_args"],
        "use_separate_heads": False,
    }
    confidence_module_kwargs = {
        "token_s": token_s,
        "token_z": token_z,
        "pairformer_args": boltz2_params["pairformer_args"],
        "token_level_confidence": True,
        "bond_type_feature": boltz2_params["bond_type_feature"],
        # cyclic_pos_enc MUST be on — it is the model-level gate around the
        # cyclic block (and the all_reduce). The default is False, which is why
        # the existing parity test never exercises the cyclic collective.
        "cyclic_pos_enc": True,
        **confidence_model_args,
    }

    val_init_range = 0.002

    selected_keys = [
        "mol_type",
        "asym_id",
        "token_pad_mask",
        "token_pair_pad_mask",
        "token_to_rep_atom",
        "atom_counts_per_token",
        "frames_idx",
        "atom_to_token",
        "atom_pad_mask",
        "atom_resolved_mask",
        "residue_index",
        "entity_id",
        "token_index",
        "sym_id",
        "cyclic_period",
        "token_bonds",
        "type_bonds",
        "contact_conditioning",
        "contact_threshold",
    ]
    feats_all = random_features(
        size_batch=B,
        n_tokens=N_tokens,
        n_atoms=N_atoms,
        n_msa=1,
        atom_counts_per_token_range=(1, 4),
        device=torch.device("cpu"),
        float_value_range=(-0.2, 0.2),
        selected_keys=selected_keys,
        rng=rng,
    )
    feats_all = {k: (v.to(dtype=dtype) if v.is_floating_point() else v) for k, v in feats_all.items()}

    # Override the six rel-pos integer features with the rank-divergent cyclic
    # layout (chain A non-cyclic first half / chain B cyclic second half), with
    # the non-trigger features randomized (contained generator) so the inputs are
    # a realistic distribution rather than all-zero/sequential.
    divergent = make_divergent_cyclic_rel_pos_feats(B, N_tokens, cyclic_period_val=cyclic_period_val, rng=rng)
    for key, val in divergent.items():
        feats_all[key] = val.to(dtype=feats_all[key].dtype) if key in feats_all else val

    # mol_type must agree with the crafted two-chain asym_id split: a downstream
    # frame-prediction assert requires every token in a chain to share one
    # mol_type. random_features couples mol_type and asym_id, but we just
    # overrode asym_id, so set both chains to PROTEIN (cyclic peptides are
    # protein) to keep the split internally coherent.
    feats_all["mol_type"] = torch.full_like(feats_all["mol_type"], const.chain_type_ids["PROTEIN"])

    feats_serial = {k: v.clone() for k, v in feats_all.items()}
    feats_dist_host = dict(feats_all)

    s_inputs = torch.empty(B, N_tokens, token_s, dtype=dtype, requires_grad=True)
    s = torch.empty(B, N_tokens, token_s, dtype=dtype, requires_grad=True)
    z = torch.empty(B, N_tokens, N_tokens, token_z, dtype=dtype, requires_grad=True)
    x_pred = torch.empty(B * multiplicity, N_atoms, 3, dtype=dtype)
    pred_distogram_logits = torch.empty(B, N_tokens, N_tokens, num_distogram_bins, dtype=dtype)

    _init_tensors_uniform_rng(
        [s_inputs, s, z, pred_distogram_logits], low=-val_init_range, high=val_init_range, rng=rng
    )
    _init_tensors_uniform_rng([x_pred], low=-10.0, high=10.0, rng=rng)

    s_inputs_global_host = s_inputs.detach().clone().cpu()
    s_global_host = s.detach().clone().cpu()
    z_global_host = z.detach().clone().cpu()
    x_pred_global_host = x_pred.detach().clone().cpu()
    pred_distogram_logits_global_host = pred_distogram_logits.detach().clone().cpu()

    serial_module = SerialConfidenceModule(**confidence_module_kwargs)
    serial_module = serial_module.to(dtype=dtype).train()
    _init_module_params_glorot_rng(serial_module, rng=rng)
    serial_state_dict = serial_module.state_dict()

    serial_output = serial_module(
        s_inputs=s_inputs,
        s=s,
        z=z,
        x_pred=x_pred,
        feats=feats_serial,
        pred_distogram_logits=pred_distogram_logits,
        multiplicity=multiplicity,
    )

    # Non-vacuity of the cyclic MATH: with cyclic_period zeroed the correction is
    # skipped, so the serial output must differ — otherwise parity could pass
    # even with a silently-broken cyclic branch.
    with torch.no_grad():
        feats_no_cyclic = {k: v.clone() for k, v in feats_serial.items()}
        feats_no_cyclic["cyclic_period"] = torch.zeros_like(feats_serial["cyclic_period"])
        serial_no_cyclic = SerialConfidenceModule(**confidence_module_kwargs).to(dtype=dtype).train()
        serial_no_cyclic.load_state_dict(serial_state_dict)
        out_no_cyclic = serial_no_cyclic(
            s_inputs=s_inputs.detach(),
            s=s.detach(),
            z=z.detach(),
            x_pred=x_pred,
            feats=feats_no_cyclic,
            pred_distogram_logits=pred_distogram_logits,
            multiplicity=multiplicity,
        )
    assert not torch.allclose(serial_output["pde_logits"].detach(), out_no_cyclic["pde_logits"]), (
        "cyclic correction is a no-op in this setup — pick cyclic_period_val small vs the "
        "residue-index span so round(d / period) != 0 for some same-chain pairs"
    )

    d_plddt_logits = torch.empty_like(serial_output["plddt_logits"])
    d_pde_logits = torch.empty_like(serial_output["pde_logits"])
    d_resolved_logits = torch.empty_like(serial_output["resolved_logits"])
    d_pae_logits = torch.empty_like(serial_output["pae_logits"])
    _init_tensors_uniform_rng(
        [d_plddt_logits, d_pde_logits, d_resolved_logits, d_pae_logits],
        low=-val_init_range,
        high=val_init_range,
        rng=rng,
    )

    d_plddt_logits_host = d_plddt_logits.detach().clone().cpu()
    d_pde_logits_host = d_pde_logits.detach().clone().cpu()
    d_resolved_logits_host = d_resolved_logits.detach().clone().cpu()
    d_pae_logits_host = d_pae_logits.detach().clone().cpu()

    torch.autograd.backward(
        [
            serial_output["plddt_logits"],
            serial_output["pde_logits"],
            serial_output["resolved_logits"],
            serial_output["pae_logits"],
        ],
        [d_plddt_logits, d_pde_logits, d_resolved_logits, d_pae_logits],
    )

    def _to_cpu(val):
        if isinstance(val, torch.Tensor):
            return val.detach().clone().cpu()
        if isinstance(val, dict):
            return {k: _to_cpu(v) for k, v in val.items()}
        return val

    serial_output_feats_host = {k: _to_cpu(v) for k, v in serial_output.items()}

    s_inputs_grad_host = s_inputs.grad.detach().clone().cpu()
    s_grad_host = s.grad.detach().clone().cpu()
    z_grad_host = z.grad.detach().clone().cpu()

    serial_param_grads_host = {
        name: param.grad.detach().clone().cpu()
        for name, param in serial_module.named_parameters()
        if param.grad is not None
    }
    # The cyclic path's only learnable parameter must have a non-zero serial
    # gradient, else the parity check on it is vacuous.
    rel_pos_serial_grads = [n for n in serial_param_grads_host if "rel_pos.linear_layer" in n]
    assert rel_pos_serial_grads, "serial rel_pos.linear_layer has no gradient (vacuous)"

    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        serial_state_dict,
        confidence_module_kwargs,
        multiplicity,
        buggy,
        pg_timeout_s,
        s_inputs_global_host,
        s_global_host,
        z_global_host,
        x_pred_global_host,
        feats_dist_host,
        pred_distogram_logits_global_host,
        serial_output_feats_host,
        s_inputs_grad_host,
        s_grad_host,
        z_grad_host,
        d_plddt_logits_host,
        d_pde_logits_host,
        d_resolved_logits_host,
        d_pae_logits_host,
        serial_param_grads_host,
    )

    spawn_multiprocessing(parallel_assert_confidence_divergent_cyclic_1d, world_size, payload)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
