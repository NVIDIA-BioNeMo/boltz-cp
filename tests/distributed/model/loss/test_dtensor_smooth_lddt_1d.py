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

"""DTensor-based tests for 1D CP smooth LDDT loss.

Tests the 1D CP smooth LDDT loss against the serial diffusionv2
smooth_lddt_loss reference.
Maps to: src/boltz/distributed/model/loss/smooth_lddt_1d.py
"""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.testing import assert_close

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.loss.smooth_lddt_1d import smooth_lddt_loss_1d as smooth_lddt_loss_dtensor
from boltz.model.loss.diffusion import smooth_lddt_loss as smooth_lddt_loss_serial_v1
from boltz.model.loss.diffusionv2 import smooth_lddt_loss as smooth_lddt_loss_serial_v2
from boltz.testing.utils import (
    assert_tensors_identical,
    init_tensors_uniform,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)


def _worker_smooth_lddt_loss_1d_parity(
    rank: int,
    pred_coords_host: torch.Tensor,
    true_coords_host: torch.Tensor,
    is_nucleotide_host: torch.Tensor,
    coords_mask_host: torch.Tensor,
    loss_ref: float,
    pred_grad_ref_host: torch.Tensor,
    d_global_loss_host: torch.Tensor,
    grid_group_sizes: dict,
    device_type: str,
    backend: str,
    multiplicity: int,
    nucleic_acid_cutoff: float,
    other_cutoff: float,
    v2: bool,
    env_per_rank: dict[str, str] | None = None,
):
    """Worker: compare 1D CP smooth LDDT loss against serial reference."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    dm = DistributedManager()

    device_mesh = dm.device_mesh

    # pred_coords and true_coords: [B*M, N, 3] with (Shard(0), Shard(1))
    coords_placements = (Shard(0), Shard(1))
    # is_nucleotide and coords_mask: [B, N] with (Shard(0), Shard(1))
    mask_placements = (Shard(0), Shard(1))

    pred_dt = distribute_tensor(pred_coords_host.to(dm.device), device_mesh, coords_placements).requires_grad_(True)
    true_dt = distribute_tensor(true_coords_host.to(dm.device), device_mesh, coords_placements)
    is_nuc_dt = distribute_tensor(is_nucleotide_host.to(dm.device), device_mesh, mask_placements)
    mask_dt = distribute_tensor(coords_mask_host.to(dm.device), device_mesh, mask_placements)

    # Clone inputs for immutability check
    pred_dt_clone = pred_dt.detach().clone().requires_grad_(pred_dt.requires_grad)
    true_dt_clone = true_dt.detach().clone()

    dp_group = device_mesh.get_group("dp")
    cp_group = device_mesh.get_group("cp")

    # Forward pass — force PyTorch backend (no Triton on CPU)
    loss_dt = smooth_lddt_loss_dtensor(
        pred_dt,
        true_dt,
        is_nuc_dt,
        mask_dt,
        device_mesh=device_mesh,
        dp_group=dp_group,
        cp_group=cp_group,
        nucleic_acid_cutoff=nucleic_acid_cutoff,
        other_cutoff=other_cutoff,
        multiplicity=multiplicity,
        use_triton=False,
    )

    # Immutability check
    assert_tensors_identical(
        pred_dt.to_local(),
        pred_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )
    assert_tensors_identical(
        true_dt.to_local(),
        true_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )

    # Forward loss parity.
    # The forward reduces (num, den) via a single packed cp-wide all_reduce in
    # fp64, then divides.  Measured on cpu-dp2-cp3 (M1 and M2): bit-identical
    # to the serial reference (max abs diff 0).  We set assert_close's fp64
    # defaults (atol=rtol=1e-7) as a small headroom in case CUDA reductions
    # differ from CPU at higher cp_size — still ~7 orders of magnitude tighter
    # than the prior 1e-5 bound.
    loss_val = loss_dt.full_tensor().item()
    assert loss_val > 0, f"Distributed smooth LDDT loss is {loss_val} — non-vacuous guard"
    assert_close(
        torch.tensor(loss_val),
        torch.tensor(loss_ref),
        msg=lambda m: f"Rank {rank} loss mismatch\n{m}",
    )

    # Backward with explicit random grad_output (not .sum().backward())
    d_global_loss_dtensor = distribute_tensor(
        d_global_loss_host.to(dm.device),
        device_mesh=device_mesh,
        placements=(Replicate(), Replicate()),
    )
    loss_dt.backward(d_global_loss_dtensor)

    # Immutability after backward
    assert_tensors_identical(
        pred_dt.to_local(),
        pred_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )
    assert_tensors_identical(
        true_dt.to_local(),
        true_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )

    # Pred gradient parity via full_tensor (canonical DTensor vs serial reference pattern)
    assert pred_dt.grad is not None, "pred gradient is None"
    pred_grad_full = pred_dt.grad.full_tensor()

    # Pred-gradient parity.  Measured on cpu-dp2-cp3 with the test seed:
    #   M=1: max abs diff = 6.7e-12, max rel diff = 3.5e-5  (rel-blowup is at
    #        a near-zero element; abs diff bounds the meaningful error)
    #   M=2: max abs diff = 4.2e-11, max rel diff = 1.4e-5
    # Backward reorders pairwise distance accumulations across cp chunks; the
    # cp-wide all_reduce of (num, den) feeds back through the chain rule.  At
    # fp64 (eps_mach=2.2e-16), N=O(36) accumulations, the per-element error
    # budget is ~N*eps*|grad_output| ≈ 1e-14 worst-case; the observed 4e-11
    # also reflects multi-stage accumulation through scatter and reduce.  We
    # set atol=5e-10 (~12x over the worst measured) and rtol=1e-4 (~7x over
    # the worst measured), tight enough to fail on a real regression while
    # leaving headroom for CUDA-vs-CPU reduction-order differences.
    assert_close(
        pred_grad_full.cpu(),
        pred_grad_ref_host,
        atol=5e-10,
        rtol=1e-4,
        msg=lambda m: f"Rank {rank} pred grad mismatch\n{m}",
    )

    # Non-vacuous: gradient must be non-zero
    assert pred_grad_full.abs().sum() > 0, "Distributed pred gradient is all-zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize("v2", [True, False], ids=["v2", "v1"])
@pytest.mark.parametrize(
    "setup_env, multiplicity",
    [
        # CPU dp=2 cp=3: DP + non-power-of-two CP for CPU-only CI
        (((2, 3), True, "cpu", "ENV"), 1),
        # CPU dp=2 cp=3: with multiplicity > 1
        (((2, 3), True, "cpu", "ENV"), 2),
        # CUDA dp=1 cp=1: serial-equivalent sanity check (1 GPU)
        (((1, 1), True, "cuda", "ENV"), 1),
        # CUDA dp=1 cp=2: CP-only under CUDA (2 GPUs)
        (((1, 2), True, "cuda", "ENV"), 1),
        # CUDA dp=1 cp=3: non-power-of-two CP on GPU (3 GPUs)
        (((1, 3), True, "cuda", "ENV"), 1),
    ],
    indirect=("setup_env",),
    ids=[
        "cpu-dp2-cp3-M1",
        "cpu-dp2-cp3-M2",
        "cuda-dp1-cp1-M1",
        "cuda-dp1-cp2-M1",
        "cuda-dp1-cp3-M1",
    ],
)
def test_dtensor_smooth_lddt_loss_1d_forward_backward(setup_env, multiplicity, v2):
    """1D CP smooth LDDT loss: distributed loss and pred gradient match serial reference."""
    if not v2:
        pytest.skip("1D smooth_lddt_loss_1d only supports v2 — v1 has no 1D CP wrapper")

    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type=device_type, world_size=world_size)

    cp_size = grid_group_sizes["cp"]
    dp_size = grid_group_sizes["dp"]
    B = 2 * dp_size  # batch size (pre-multiplicity)
    N_per_shard = 12
    N = N_per_shard * cp_size  # tokens, divisible by cp
    nucleic_acid_cutoff = 30.0
    other_cutoff = 15.0
    dtype = torch.float64

    smooth_lddt_loss_serial = smooth_lddt_loss_serial_v2 if v2 else smooth_lddt_loss_serial_v1

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        # Scale coordinates by cutoff so some distances exceed it (tests cutoff masking)
        pred_coords = torch.randn(B * multiplicity, N, 3, dtype=dtype) * nucleic_acid_cutoff
        pred_coords.requires_grad_(True)

        true_coords = torch.randn(B * multiplicity, N, 3, dtype=dtype) * nucleic_acid_cutoff

        # is_nucleotide: [B, N] — pre-multiplicity, binary float64
        is_nucleotide = torch.randint(0, 2, (B, N)).to(dtype=dtype)

        # coords_mask: [B, N] — mask last 2 atoms per shard to test diagonal-zeroing
        # at cp-rank boundaries (emulates virtual atoms in the middle of the sequence)
        coords_mask = torch.ones(B, N, dtype=dtype)
        for r in range(cp_size):
            end_idx = (r + 1) * N_per_shard
            start_idx = end_idx - 2
            coords_mask[:, start_idx:end_idx] = 0.0

        # Serial forward
        loss_ref = smooth_lddt_loss_serial(
            pred_coords,
            true_coords,
            is_nucleotide,
            coords_mask,
            nucleic_acid_cutoff=nucleic_acid_cutoff,
            other_cutoff=other_cutoff,
            multiplicity=multiplicity,
        )

        loss_ref_val = loss_ref.item()
        assert loss_ref_val > 0, (
            f"Serial smooth LDDT loss is {loss_ref_val} — test data produced zero "
            f"loss, making parity checks vacuous."
        )

        # Explicit random grad_output (not implicit 1.0) per CLAUDE.md convention
        d_global_loss = torch.empty(loss_ref.shape, dtype=loss_ref.dtype)
        init_tensors_uniform([d_global_loss], low=-0.5, high=0.5)

        loss_ref.backward(d_global_loss)
        pred_grad_ref = pred_coords.grad.detach().cpu().clone()

    spawn_multiprocessing(
        _worker_smooth_lddt_loss_1d_parity,
        world_size,
        pred_coords.detach().cpu(),
        true_coords.detach().cpu(),
        is_nucleotide.detach().cpu(),
        coords_mask.detach().cpu(),
        loss_ref_val,
        pred_grad_ref,
        d_global_loss.detach().cpu(),
        grid_group_sizes,
        device_type,
        backend,
        multiplicity,
        nucleic_acid_cutoff,
        other_cutoff,
        v2,
        env_per_rank,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
