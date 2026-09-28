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

"""DTensor-based tests for 1D CP B-factor loss.

Tests the 1D CP bfactor loss against the serial bfactor_loss_fn.
Maps to: src/boltz/distributed/model/loss/bfactor_1d.py
"""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.testing import assert_close

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.loss.bfactor_1d import bfactor_loss_1d as bfactor_loss_dtensor
from boltz.model.loss.bfactor import bfactor_loss_fn as bfactor_loss_serial
from boltz.testing.utils import (
    assert_tensors_identical,
    init_tensors_uniform,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)

SEED = 42


def _worker_bfactor_loss_1d_parity(
    rank: int,
    pred_on_host: torch.Tensor,
    t2ra_on_host: torch.Tensor,
    bf_on_host: torch.Tensor,
    loss_ref: float,
    pred_grad_ref_on_host: torch.Tensor,
    d_global_loss_host: torch.Tensor,
    bf_placement_kind: str,
    grid_group_sizes: dict,
    device_type: str,
    backend: str,
    env_map: dict[str, str] | None = None,
):
    """Worker: compare 1D CP bfactor loss against serial reference.

    ``bf_placement_kind`` selects the bfactor input placement:
    - "replicated": (Shard(0), Replicate()) — already aligned with t2ra's atom axis.
    - "sharded":    (Shard(0), Shard(1))    — production atom placement; exercises
                    the wrapper-level redistribute path in ``bfactor_loss_1d``.
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_map is not None:
        for var_name, value in env_map.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    dm = DistributedManager()

    # 1D CP uses 2D mesh (dp, cp)
    device_mesh = dm.device_mesh

    single_placements = (Shard(0), Shard(1))
    if bf_placement_kind == "replicated":
        # bfactor is (B, A) replicated on the atom axis — pre-aligned input path.
        atom_placements = (Shard(0), Replicate())
    elif bf_placement_kind == "sharded":
        # Production atom placement — atoms sharded on CP. The wrapper must
        # redistribute to (Shard(0), Replicate()) before the bmm against t2ra.
        atom_placements = (Shard(0), Shard(1))
    else:
        raise ValueError(f"Unknown bf_placement_kind: {bf_placement_kind!r}")

    pred_dt = distribute_tensor(pred_on_host.to(dm.device), device_mesh, single_placements).requires_grad_(True)
    t2ra_dt = distribute_tensor(t2ra_on_host.to(dm.device), device_mesh, single_placements)
    bf_dt = distribute_tensor(bf_on_host.to(dm.device), device_mesh, atom_placements)

    # Non-vacuous sharding guard: when atoms are sharded, the local atom axis
    # must be strictly smaller than the global atom axis so the test actually
    # exercises the wrapper redistribute.
    if bf_placement_kind == "sharded" and grid_group_sizes["cp"] > 1:
        assert bf_dt.to_local().shape[1] < bf_on_host.shape[1], (
            f"Rank {rank}: bfactor not sharded — local atoms={bf_dt.to_local().shape[1]} "
            f"== global atoms={bf_on_host.shape[1]} (cp={grid_group_sizes['cp']})"
        )

    # Clone inputs for immutability check
    pred_dt_clone = pred_dt.detach().clone().requires_grad_(pred_dt.requires_grad)
    t2ra_dt_clone = t2ra_dt.detach().clone().requires_grad_(t2ra_dt.requires_grad)
    bf_dt_clone = bf_dt.detach().clone().requires_grad_(bf_dt.requires_grad)

    output = {"pbfactor": pred_dt}
    feats = {"token_to_rep_atom": t2ra_dt, "bfactor": bf_dt}

    dp_group = device_mesh.get_group("dp")
    cp_group = device_mesh.get_group("cp")

    # Forward
    loss_dt = bfactor_loss_dtensor(
        output,
        feats,
        device_mesh=device_mesh,
        dp_group=dp_group,
        cp_group=cp_group,
    )

    # Immutability check
    assert_tensors_identical(
        pred_dt.to_local(),
        pred_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )
    assert_tensors_identical(
        t2ra_dt.to_local(),
        t2ra_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )
    assert_tensors_identical(
        bf_dt.to_local(),
        bf_dt_clone.to_local(),
        check_grad=False,
        check_grad_fn=False,
    )

    # Forward loss parity
    loss_val = loss_dt.full_tensor().item()
    # fp32 defaults bound the mean-reduced BCE at N=B*N_tokens=2*8*cp_size <= 48
    assert_close(
        torch.tensor(loss_val),
        torch.tensor(loss_ref),
        msg=lambda m: f"Rank {rank} loss mismatch\n{m}",
    )

    # Backward with explicit random grad_output (not implicit 1.0)
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

    # Pred gradient parity
    assert pred_dt.grad is not None, "pred gradient is None"
    pred_grad_full = pred_dt.grad.full_tensor()
    # fp32 defaults bound the mean-reduced BCE at N=B*N_tokens=2*8*cp_size <= 48
    assert_close(
        pred_grad_full,
        pred_grad_ref_on_host.to(dm.device),
        msg=lambda m: f"Rank {rank} pred grad mismatch\n{m}",
    )

    # Dtype checks
    assert loss_dt.dtype == torch.float32, f"Loss dtype should be fp32, got {loss_dt.dtype}"
    assert pred_dt.grad.dtype == torch.float32, f"Grad dtype should be fp32, got {pred_dt.grad.dtype}"

    # Non-vacuous: gradient must be non-zero
    assert pred_grad_full.abs().sum() > 0, "Distributed pred gradient is all-zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "bf_placement_kind",
    ["replicated", "sharded"],
    ids=["bf-replicated", "bf-sharded"],
)
@pytest.mark.parametrize(
    "setup_env",
    [
        # CPU dp=2 cp=3: DP + non-power-of-two CP for CPU-only CI
        ((2, 3), True, "cpu", "ENV"),
        # CUDA dp=1 cp=1: serial-equivalent sanity check (1 GPU)
        ((1, 1), True, "cuda", "ENV"),
        # CUDA dp=2 cp=1: DP-only path (2 GPUs)
        ((2, 1), True, "cuda", "ENV"),
        # CUDA dp=1 cp=2: CP-only under CUDA (2 GPUs)
        ((1, 2), True, "cuda", "ENV"),
        # CUDA dp=1 cp=3: non-power-of-two CP on GPU (3 GPUs)
        ((1, 3), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp2-cp3", "cuda-dp1-cp1", "cuda-dp2-cp1", "cuda-dp1-cp2", "cuda-dp1-cp3"],
)
def test_dtensor_bfactor_loss_1d_forward_backward(setup_env, bf_placement_kind):
    """1D CP bfactor loss: distributed loss and pred gradient match serial reference.

    The ``bf_placement_kind`` axis exercises both bfactor input placements:
    ``bf-replicated`` covers the pre-aligned ``(Shard(0), Replicate())`` path
    used by existing call sites, and ``bf-sharded`` covers the production
    ``(Shard(0), Shard(1))`` atom placement, which goes through the wrapper-level
    redistribute in ``bfactor_loss_1d``. Without ``bf-sharded`` coverage the
    structural shape bug fixed by reusing the redistribute pattern from
    ``confidence_1d.resolved_loss_1d`` would have remained invisible to CI.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type=device_type, world_size=world_size)

    B = 2
    bins = 8
    cp_size = grid_group_sizes["cp"]
    N = 8 * cp_size
    A = N  # one atom per token for simplicity

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(SEED)

        pred = torch.randn(B, N, bins, requires_grad=True)

        # token_to_rep_atom: identity-like (each token maps to one atom)
        t2ra = torch.zeros(B, N, A, dtype=torch.float32)
        for b in range(B):
            for i in range(N):
                t2ra[b, i, i] = 1.0

        # bfactor: realistic values (0-100 range), some zeros
        bf = torch.rand(B, A) * 80.0
        bf[:, : A // 4] = 0.0

        # Serial loss
        output_serial = {"pbfactor": pred}
        feats_serial = {"token_to_rep_atom": t2ra, "bfactor": bf}
        loss_ref = bfactor_loss_serial(output_serial, feats_serial)

        loss_ref_val = loss_ref.item()
        assert loss_ref_val > 0, (
            f"Serial bfactor loss is {loss_ref_val} — test data produced zero " f"loss, making parity checks vacuous."
        )

        # Explicit random grad_output (not implicit 1.0) per CLAUDE.md convention
        d_global_loss = torch.empty(loss_ref.shape, dtype=loss_ref.dtype)
        init_tensors_uniform([d_global_loss], low=-0.5, high=0.5)

        loss_ref.backward(d_global_loss)
        pred_grad_ref = pred.grad.detach().cpu().clone()

    spawn_multiprocessing(
        _worker_bfactor_loss_1d_parity,
        world_size,
        pred.detach().cpu(),
        t2ra.detach().cpu(),
        bf.detach().cpu(),
        loss_ref_val,
        pred_grad_ref,
        d_global_loss.detach().cpu(),
        bf_placement_kind,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
