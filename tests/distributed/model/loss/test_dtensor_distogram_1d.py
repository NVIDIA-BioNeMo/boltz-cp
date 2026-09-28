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

"""DTensor-based tests for 1D CP distogram loss.

Tests the 1D CP distogram loss against serial distogramv2 references.
Maps to: src/boltz/distributed/model/loss/distogram_1d.py
"""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.loss.distogram_1d import (
    distogram_loss_1d as distogram_loss_dtensor,
)
from boltz.model.loss.distogram import distogram_loss as distogram_loss_v1
from boltz.model.loss.distogramv2 import distogram_loss as distogram_loss_v2
from boltz.testing.utils import (
    assert_tensors_identical,
    chunk_along_dim,
    init_tensors_uniform,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)


def parallel_assert_distogram_loss_1d(
    rank,
    payload,
):
    """Worker function for 1D CP distogram loss testing."""
    (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        pred_global_host,
        target_global_host,
        mask_global_host,
        global_loss_expected_host,
        batch_loss_expected_host,
        d_global_loss_host,
        d_pred_expected_host,
        aggregate_distogram,
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

    # 1D CP uses 2D mesh (dp, cp) — no subgroups
    device_mesh = manager.device_mesh

    # Distribute tensors — pair placements for 1D CP: (Shard(0), Shard(1))
    pair_placements = (Shard(0), Shard(1))
    single_placements = (Shard(0), Shard(1))

    pred_dtensor = distribute_tensor(
        pred_global_host.to(manager.device),
        device_mesh=device_mesh,
        placements=pair_placements,
    ).requires_grad_(True)

    target_dtensor = distribute_tensor(
        target_global_host.to(manager.device),
        device_mesh=device_mesh,
        placements=pair_placements,
    )

    mask_dtensor = distribute_tensor(
        mask_global_host.to(manager.device),
        device_mesh=device_mesh,
        placements=single_placements,
    )

    output_dtensor = {"pdistogram": pred_dtensor}
    feats_dtensor = {
        "disto_target": target_dtensor,
        "token_disto_mask": mask_dtensor,
    }

    # Create copies for immutability check
    output_dtensor_copy = {
        key: tensor.detach().clone().requires_grad_(tensor.requires_grad) for key, tensor in output_dtensor.items()
    }
    feats_dtensor_copy = {
        key: tensor.detach().clone().requires_grad_(tensor.requires_grad) for key, tensor in feats_dtensor.items()
    }

    dp_group = device_mesh.get_group("dp")
    cp_group = device_mesh.get_group("cp")

    # Forward pass
    global_loss_result, batch_loss_result = distogram_loss_dtensor(
        output_dtensor,
        feats_dtensor,
        device_mesh=device_mesh,
        dp_group=dp_group,
        cp_group=cp_group,
        aggregate_distogram=aggregate_distogram,
    )

    # Verify placements have correct ndim for the 2D mesh (dp, cp)
    mesh_ndim = device_mesh.ndim
    assert len(global_loss_result.placements) == mesh_ndim, (
        f"global_loss placements should have {mesh_ndim} elements for {mesh_ndim}D mesh, "
        f"got {len(global_loss_result.placements)}: {global_loss_result.placements}"
    )
    assert len(batch_loss_result.placements) == mesh_ndim, (
        f"batch_loss placements should have {mesh_ndim} elements for {mesh_ndim}D mesh, "
        f"got {len(batch_loss_result.placements)}: {batch_loss_result.placements}"
    )

    # Verify inputs weren't modified
    assert_tensors_identical(
        output_dtensor_copy["pdistogram"].to_local(),
        output_dtensor["pdistogram"].to_local(),
        check_grad=False,
        check_grad_fn=False,
    )
    for key in feats_dtensor:
        assert_tensors_identical(
            feats_dtensor_copy[key].to_local(),
            feats_dtensor[key].to_local(),
            check_grad=False,
            check_grad_fn=False,
        )

    # Forward loss parity
    torch.testing.assert_close(
        global_loss_result.full_tensor(),
        global_loss_expected_host.to(manager.device),
    )

    # batch_loss is [B] sharded on DP dim
    dp_rank = manager.group_rank["dp"]
    dp_size = grid_group_sizes["dp"]
    batch_loss_expected_local = chunk_along_dim(batch_loss_expected_host, dim=0, chunk_i=dp_rank, chunks=dp_size)
    torch.testing.assert_close(
        batch_loss_result.to_local(),
        batch_loss_expected_local.to(manager.device),
    )

    # Backward pass
    d_global_loss_dtensor = distribute_tensor(
        d_global_loss_host.to(manager.device),
        device_mesh=device_mesh,
        placements=(Replicate(), Replicate()),
    )

    global_loss_result.backward(d_global_loss_dtensor)

    # Verify gradient on pred
    assert output_dtensor["pdistogram"].grad is not None, "pred gradient is None"
    assert d_pred_expected_host is not None, "Reference pred gradient is None"

    # Shard the reference gradient to match this rank's local portion
    dp_rank = manager.group_rank["dp"]
    cp_rank = manager.group_rank["cp"]
    cp_size = grid_group_sizes["cp"]

    d_pred_expected_local = chunk_along_dim(d_pred_expected_host, dim=0, chunk_i=dp_rank, chunks=dp_size)
    d_pred_expected_local = chunk_along_dim(d_pred_expected_local, dim=1, chunk_i=cp_rank, chunks=cp_size)

    torch.testing.assert_close(
        output_dtensor["pdistogram"].grad.to_local().cpu(),
        d_pred_expected_local,
        msg=lambda m: f"Pred gradient mismatch on rank {rank}\n{m}",
    )

    # Non-vacuous: gradient must be non-zero
    assert output_dtensor["pdistogram"].grad.full_tensor().abs().sum() > 0, "Distributed pred gradient is all-zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, loss_config",
    [
        # CUDA dp=1 cp=1: serial-equivalent sanity check (1 GPU)
        (((1, 1), True, "cuda", "ENV"), (1, 1, True)),
        # CUDA dp=2 cp=1: DP-only path (2 GPUs)
        (((2, 1), True, "cuda", "ENV"), (1, 1, True)),
        # CUDA dp=1 cp=2: CP-only with aggregate (2 GPUs)
        (((1, 2), True, "cuda", "ENV"), (1, 1, True)),
        # CPU dp=2 cp=1: dp>1 regression guard
        (((2, 1), True, "cpu", "ENV"), (1, 1, True)),
        # CPU dp=2 cp=3: DP + non-power-of-two CP with non-aggregate
        (((2, 3), True, "cpu", "ENV"), (3, 2, False)),
        # CPU dp=1 cp=2: K>1 conformers with aggregate
        (((1, 2), True, "cpu", "ENV"), (3, 1, True)),
        # CUDA dp=1 cp=3: non-power-of-two CP on GPU (3 GPUs)
        (((1, 3), True, "cuda", "ENV"), (1, 1, True)),
    ],
    indirect=("setup_env",),
    ids=[
        "cuda-dp1-cp1-K1D1-agg",
        "cuda-dp2-cp1-K1D1-agg",
        "cuda-dp1-cp2-K1D1-agg",
        "cpu-dp2-cp1-K1D1-agg",
        "cpu-dp2-cp3-K3D2-noagg",
        "cpu-dp1-cp2-K3D1-agg",
        "cuda-dp1-cp3-K1D1-agg",
    ],
)
def test_dtensor_distogram_loss_1d(setup_env, loss_config):
    """Test 1D CP distogram loss against serial reference."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    K, D, aggregate_distogram = loss_config

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 10
    num_bins = 16

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        min_init_val = -0.5
        max_init_val = 0.5

        pred_global = torch.empty((B, N, N, D, num_bins), requires_grad=True)
        init_tensors_uniform([pred_global], low=min_init_val, high=max_init_val)

        if aggregate_distogram and K == 1:
            target_idx = torch.randint(0, num_bins, (B, N, N))
            target_global = torch.nn.functional.one_hot(target_idx, num_classes=num_bins).float()
            target_global = target_global.unsqueeze(3)
        else:
            target_global = torch.empty((B, N, N, K, num_bins))
            init_tensors_uniform([target_global], low=0.01, high=1.0)
            target_global = target_global / target_global.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_global.requires_grad_(False)

        mask_global = torch.randint(0, 2, (B, N), dtype=torch.bool)

        # Run serial forward pass
        output_global = {"pdistogram": pred_global}
        feats_global = {"disto_target": target_global, "token_disto_mask": mask_global}

        global_loss_expected, batch_loss_expected = distogram_loss_v2(
            output_global, feats_global, aggregate_distogram=aggregate_distogram
        )

        d_global_loss = torch.empty(global_loss_expected.shape, dtype=global_loss_expected.dtype)
        init_tensors_uniform([d_global_loss], low=min_init_val, high=max_init_val)

        global_loss_expected.backward(d_global_loss)

    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        pred_global.detach().clone().cpu(),
        target_global.detach().clone().cpu(),
        mask_global.detach().clone().cpu(),
        global_loss_expected.detach().clone().cpu(),
        batch_loss_expected.detach().clone().cpu(),
        d_global_loss.detach().clone().cpu(),
        pred_global.grad.detach().clone().cpu(),
        aggregate_distogram,
    )

    spawn_multiprocessing(parallel_assert_distogram_loss_1d, world_size, payload)


@pytest.mark.parametrize(
    "setup_env",
    [
        # CPU dp=2 cp=1: minimal distributed setup to verify v1 equivalence
        ((2, 1), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=["cpu-dp2-cp1"],
)
def test_dtensor_distogram_loss_1d_v1_compat(setup_env):
    """Test that the 1D CP distogram loss with D=1, K=1 matches the v1 serial loss.

    The v1 serial loss uses 4D tensors [B, N, N, bins]. To use the v2/CP
    implementation as a v1 loss, unsqueeze dim 3 of pred and target to get
    [B, N, N, 1, bins] with D=1 and K=1. With aggregate_distogram=True,
    min-over-D and mean-over-K are identity ops, so the result must match v1.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    skip_if_cuda_not_avail_or_device_count_less_than_word_size(
        device_type=device_type,
        world_size=world_size,
    )

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 10
    num_bins = 16

    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(42)

        min_init_val = -0.5
        max_init_val = 0.5

        # v1 tensors are 4D: [B, N, N, bins]
        pred_4d = torch.empty((B, N, N, num_bins), requires_grad=True)
        init_tensors_uniform([pred_4d], low=min_init_val, high=max_init_val)

        target_idx = torch.randint(0, num_bins, (B, N, N))
        target_4d = torch.nn.functional.one_hot(target_idx, num_classes=num_bins).float()

        mask_global = torch.randint(0, 2, (B, N), dtype=torch.bool)

        # Run v1 serial loss as reference
        output_v1 = {"pdistogram": pred_4d}
        feats_v1 = {"disto_target": target_4d, "token_disto_mask": mask_global}
        global_loss_expected, batch_loss_expected = distogram_loss_v1(output_v1, feats_v1)

        d_global_loss = torch.empty(global_loss_expected.shape, dtype=global_loss_expected.dtype)
        init_tensors_uniform([d_global_loss], low=min_init_val, high=max_init_val)
        global_loss_expected.backward(d_global_loss)

        # Unsqueeze to 5D for the CP implementation: [B,N,N,bins] -> [B,N,N,1,bins]
        pred_5d = pred_4d.detach().clone().unsqueeze(3)
        target_5d = target_4d.detach().clone().unsqueeze(3)
        # Gradient also gains the unsqueeze dim
        d_pred_5d = pred_4d.grad.detach().clone().unsqueeze(3)

    payload = (
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        pred_5d.cpu(),
        target_5d.cpu(),
        mask_global.detach().clone().cpu(),
        global_loss_expected.detach().clone().cpu(),
        batch_loss_expected.detach().clone().cpu(),
        d_global_loss.detach().clone().cpu(),
        d_pred_5d.cpu(),
        True,  # aggregate_distogram
    )

    spawn_multiprocessing(parallel_assert_distogram_loss_1d, world_size, payload)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
