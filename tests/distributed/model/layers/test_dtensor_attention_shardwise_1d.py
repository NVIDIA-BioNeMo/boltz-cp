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

"""Parity tests for AttentionPairBiasShardwise on a 2D mesh ``(dp, cp)``.

The existing ``test_dtensor_attention.py`` tests ``AttentionPairBiasShardwise``
on a 3D mesh ``(dp, cp0, cp1)`` with placements ``(Shard(0), Shard(1), Replicate())``.
This test verifies the same module works correctly on a 2D mesh ``(dp, cp)`` with
placements ``(Shard(0), Shard(1))`` — the layout used by 1D CP atom-level attention
(AtomTransformer inside DiffusionModule1D).

The ``_AttentionPairBiasShardwiseImpl`` autograd function is mesh-dimensionality-agnostic:
it only checks that all inputs share the same placements and device mesh, then operates
on local shards. No cross-rank communication is needed because window-batched attention
is shard-local by construction.
"""

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.attention import AttentionPairBiasShardwise
from boltz.distributed.model.modules.utils import SDPAWithBiasBackend
from boltz.model.layers.attentionv2 import AttentionPairBias as SerialAttentionPairBiasV2
from boltz.testing.utils import (
    assert_all_identical,
    seed_by_rank,
    spawn_multiprocessing,
)


def _worker_shardwise_attention_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    c_s,
    num_heads,
    inf,
    compute_pair_bias,
    layer_state_dict,
    s_global_host,
    z_global_host,
    mask_global_host,
    k_in_global_host,
    output_expected_global_host,
    d_output_global_host,
    d_s_expected_global_host,
    d_k_in_expected_global_host,
    d_z_expected_global_host,
    grad_params_expected_global_host,
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

    device = manager.device
    device_mesh = manager.device_mesh

    # Create serial module and load state dict
    serial_module = SerialAttentionPairBiasV2(
        c_s=c_s,
        c_z=c_s if compute_pair_bias else None,
        num_heads=num_heads,
        inf=inf,
        compute_pair_bias=compute_pair_bias,
    )
    serial_module.to(dtype=dtype, device=device)
    serial_module.load_state_dict(layer_state_dict)
    serial_module.train()

    # Create distributed module on 2D mesh (dp, cp)
    module = AttentionPairBiasShardwise(
        attn_pair_bias=serial_module,
        device_mesh=device_mesh,
        sdpa_with_bias_backend=SDPAWithBiasBackend.REFERENCE,
        apply_initial_norm=False,
        compute_pair_bias=compute_pair_bias,
        use_model_cache=False,
    )
    module.train()

    # 2D mesh placements: (Shard(0), Shard(1))
    # dp shards B, cp shards K (number of windows)
    placements = (Shard(0), Shard(1))

    # Distribute input tensors
    s_dtensor = distribute_tensor(
        s_global_host.to(dtype=dtype, device=device),
        device_mesh,
        placements,
    ).requires_grad_(True)

    z_dtensor = distribute_tensor(
        z_global_host.to(dtype=dtype, device=device),
        device_mesh,
        placements,
    ).requires_grad_(True)

    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=device),
        device_mesh,
        placements,
    )

    d_output_dtensor = distribute_tensor(
        d_output_global_host.to(dtype=dtype, device=device),
        device_mesh,
        placements,
    )

    # V2 API: pass pre-computed k_in
    k_in_dtensor = distribute_tensor(
        k_in_global_host.to(dtype=dtype, device=device),
        device_mesh,
        placements,
    ).requires_grad_(True)
    output_dtensor = module(s_dtensor, z_dtensor, mask_dtensor, k_in=k_in_dtensor)

    # Forward parity: local shard comparison (no communication)
    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=device),
        device_mesh,
        placements,
        src_data_rank=None,
    )
    # fp64 defaults: shardwise attn is shard-local (no cp-reorder); SDPA reduction over key-window N=128
    torch.testing.assert_close(
        output_dtensor.to_local(),
        output_expected_dtensor.to_local(),
    )

    # Forward parity: full tensor comparison (requires all-gather)
    output_global_result = output_dtensor.full_tensor().cpu()
    # fp64 defaults: shardwise attn is shard-local (no cp-reorder); SDPA reduction over key-window N=128
    torch.testing.assert_close(
        output_global_result,
        output_expected_global_host.to(dtype=dtype),
    )

    # Backward pass
    output_dtensor.backward(d_output_dtensor)

    # Input gradient parity (full tensor)
    d_s_global_result = s_dtensor.grad.full_tensor().cpu()
    # fp64 defaults: shardwise attn is shard-local (no cp-reorder); SDPA reduction over key-window N=128
    torch.testing.assert_close(
        d_s_global_result,
        d_s_expected_global_host.to(dtype=dtype),
    )

    # z gradient parity (full tensor)
    d_z_global_result = z_dtensor.grad.full_tensor().cpu()
    # fp64 defaults: shardwise attn is shard-local (no cp-reorder); SDPA reduction over key-window N=128
    torch.testing.assert_close(
        d_z_global_result,
        d_z_expected_global_host.to(dtype=dtype),
    )

    # k_in gradient parity
    d_k_in_global_result = k_in_dtensor.grad.full_tensor().cpu()
    # fp64 defaults: shardwise attn is shard-local (no cp-reorder); SDPA reduction over key-window N=128
    torch.testing.assert_close(
        d_k_in_global_result,
        d_k_in_expected_global_host.to(dtype=dtype),
    )

    # Parameter gradient parity
    for name, grad_param_expected in grad_params_expected_global_host.items():
        param = dict(module.named_parameters())[name]
        assert param.grad is not None, f"Parameter {name} has no gradient"
        param_grad_result = param.grad.full_tensor().cpu()
        # fp64 defaults: shardwise attn is shard-local (no cp-reorder); SDPA reduction over key-window N=128
        torch.testing.assert_close(
            param_grad_result,
            grad_param_expected.to(dtype=dtype),
        )
        # Replicated params should have identical gradients across cp ranks
        assert_all_identical(param.grad.full_tensor(), manager.group["cp"])

    # Guard against vacuous pass: verify cp sharding is active
    cp_placement = s_dtensor.placements[1]
    if isinstance(cp_placement, Shard):
        local_size = s_dtensor.to_local().shape[cp_placement.dim]
        global_size = s_dtensor.shape[cp_placement.dim]
        assert (
            local_size < global_size
        ), f"CP sharding not active on dim {cp_placement.dim}: local={local_size}, global={global_size}"

    # Guard: gradients are non-zero
    assert s_dtensor.grad.to_local().abs().sum() > 0, "s gradients are all zero"
    assert z_dtensor.grad.to_local().abs().sum() > 0, "z gradients are all zero"
    assert k_in_dtensor.grad.to_local().abs().sum() > 0, "k_in gradients are all zero"
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert param.grad.to_local().abs().sum() > 0, f"Gradient for {name} is all zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
        ((2, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cuda", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
@pytest.mark.parametrize(
    "compute_pair_bias",
    [True, False],
    ids=lambda x: f"pair_bias={'yes' if x else 'no'}",
)
def test_shardwise_attention_pair_bias_1d(setup_env, compute_pair_bias):
    """Test AttentionPairBiasShardwise on 2D mesh (dp, cp) with (Shard(0), Shard(1)) placements.

    Verifies forward/backward parity against the serial V2 module using
    pre-computed k_in (V2 API). The window-batched attention is shard-local
    so no cross-rank communication is needed — this test confirms that
    the implementation is mesh-dimensionality-agnostic.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    dtype = torch.float64

    c_s = 32  # c_s // num_heads must be >= 16 and multiple of 4
    c_z = c_s
    num_heads = 2
    inf = 1e6
    z_last_dim = c_z if compute_pair_bias else num_heads

    cp_size = grid_group_sizes["cp"]
    dp_size = grid_group_sizes["dp"]

    # Dimensions: B must be divisible by dp, K must be divisible by cp
    B = 2 * dp_size
    W = 32  # query window size
    H = 128  # key window size
    K = 4 * cp_size  # number of windows, must be divisible by cp
    D = c_s

    seed_by_rank(0, seed=42)

    # Create serial module
    serial_module = SerialAttentionPairBiasV2(
        c_s=c_s,
        c_z=c_z if compute_pair_bias else None,
        num_heads=num_heads,
        inf=inf,
        compute_pair_bias=compute_pair_bias,
    )
    with torch.no_grad():
        for param in serial_module.parameters():
            param.uniform_(-5e-2, 5e-2)
    layer_state_dict = serial_module.state_dict()
    serial_module = serial_module.to(dtype=dtype, device=device_type).train()

    # Create inputs: window-batched shapes
    # s: (B, K, W, D) — single repr in window-batched form
    # z: (B, K, W, H, z_last_dim) — pair repr / pre-computed bias
    # mask: (B, K, H) — key-aligned mask for V2 API
    # k_in: (B, K, H, D) — pre-computed key input
    s = torch.empty(B, K, W, D, dtype=dtype, device=device_type)
    z = torch.empty(B, K, W, H, z_last_dim, dtype=dtype, device=device_type)
    k_in = torch.empty(B, K, H, D, dtype=dtype, device=device_type)
    mask_key = torch.randint(0, 2, (B, K, H), dtype=torch.float64, device=device_type)

    with torch.no_grad():
        s.uniform_(-5e-2, 5e-2)
        z.uniform_(-5e-2, 5e-2)
        k_in.uniform_(-5e-2, 5e-2)

    s = s.requires_grad_(True)
    z = z.requires_grad_(True)
    k_in = k_in.requires_grad_(True)

    # Reshape for serial module: (B, K, ...) -> (B*K, ...)
    s_reshaped = s.view(B * K, W, -1)
    z_reshaped = z.view(B * K, W, H, -1)
    k_in_reshaped = k_in.view(B * K, H, -1)
    mask_reshaped = mask_key.view(B * K, H)

    # Run serial forward (V2 API: k_in is mandatory)
    o_serial = serial_module(
        s=s_reshaped,
        z=z_reshaped,
        mask=mask_reshaped,
        k_in=k_in_reshaped,
    )

    # Clone forward output
    o_global_host = o_serial.detach().clone().cpu().view(B, K, W, D)

    # Explicit random grad_output (.sum().backward() is forbidden)
    d_o = torch.empty(B * K, W, D, dtype=dtype, device=device_type)
    with torch.no_grad():
        d_o.uniform_(-5e-2, 5e-2)
    # Mask out gradients for invalid query positions
    mask_query = torch.ones(B, K, W, dtype=dtype, device=device_type)
    d_o = d_o * mask_query.view(B * K, W).unsqueeze(-1)

    o_serial.backward(d_o)

    d_s_expected = s.grad.detach().clone().cpu()
    d_z_expected = z.grad.detach().clone().cpu()
    d_k_in_expected = k_in.grad.detach().clone().cpu()
    grad_params_expected = {name: p.grad.detach().clone().cpu() for name, p in serial_module.named_parameters()}

    s_global_host = s.detach().clone().cpu()
    z_global_host = z.detach().clone().cpu()
    k_in_global_host = k_in.detach().clone().cpu()
    mask_global_host = mask_key.detach().clone().cpu()
    d_o_global_host = d_o.detach().clone().cpu().view(B, K, W, D)

    spawn_multiprocessing(
        _worker_shardwise_attention_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        c_s,
        num_heads,
        inf,
        compute_pair_bias,
        layer_state_dict,
        s_global_host,
        z_global_host,
        mask_global_host,
        k_in_global_host,
        o_global_host,
        d_o_global_host,
        d_s_expected,
        d_k_in_expected,
        d_z_expected,
        grad_params_expected,
    )
