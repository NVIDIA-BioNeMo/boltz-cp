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

"""Parity tests for AttentionPairBias1D under 1D CP (2D mesh ``(dp, cp)``)."""

import functools
from unittest.mock import MagicMock

import pytest
import torch
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.attention_1d import AttentionPairBias1D
from boltz.distributed.model.modules.utils import SDPAWithBiasBackend
from boltz.model.layers.attentionv2 import AttentionPairBias as SerialAttentionPairBias
from boltz.testing.utils import (
    assert_all_identical,
    assert_no_percentile_upshift,
    get_param_by_key,
    seed_by_rank,
    spawn_multiprocessing,
)


@functools.lru_cache(maxsize=1)
def _flex_attn_compile_works() -> bool:
    """Probe whether torch.compile(flex_attention) succeeds on this PyTorch build.

    PyTorch 2.9.0a0 nightly has an Inductor lowering bug in flex_attention's
    BlockMask._transpose_ordered that causes a NotImplementedError during
    codegen. This probe catches that at test time so we can skip gracefully.
    """
    try:
        from boltz.distributed.model.layers.attention_1d import flex_attention_compiled

        if flex_attention_compiled is None:
            return False
        q = torch.randn(1, 1, 128, 16, device="cuda")
        k = torch.randn(1, 1, 128, 16, device="cuda")
        v = torch.randn(1, 1, 128, 16, device="cuda")
        flex_attention_compiled(q, k, v)
        return True
    except Exception:
        return False


def _worker_attention_pair_bias_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    c_s,
    c_z,
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
    sdpa_with_bias_backend,
    output_global_fp32_host=None,
    d_s_global_fp32_host=None,
    d_z_global_fp32_host=None,
    d_k_in_global_fp32_host=None,
    grad_params_fp32_global_host=None,
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

    check_error_hist = output_global_fp32_host is not None

    seed_by_rank(rank)

    # Create serial module and load state dict
    serial_module = SerialAttentionPairBias(
        c_s=c_s,
        c_z=c_z if compute_pair_bias else None,
        num_heads=num_heads,
        inf=inf,
        compute_pair_bias=compute_pair_bias,
    )
    serial_module.to(dtype=dtype, device=manager.device)
    serial_module.load_state_dict(layer_state_dict)
    serial_module.train()

    # Create distributed module -- uses 2D device_mesh (dp, cp)
    module = AttentionPairBias1D(
        serial_module,
        manager.device_mesh,
        manager.group["cp"],
        sdpa_with_bias_backend=sdpa_with_bias_backend,
    )
    module.train()

    # Guard: verify flex_attention will actually activate when requested,
    # preventing silent fallback to REFERENCE that would make the test vacuous.
    if sdpa_with_bias_backend == SDPAWithBiasBackend.TORCH_FLEX_ATTN:
        from boltz.distributed.model.layers.attention_1d import HAS_FLEX_ATTN, is_power_of_2

        head_dim = c_s // num_heads
        assert HAS_FLEX_ATTN, "flex_attention not available — test cannot exercise TORCH_FLEX_ATTN path"
        assert is_power_of_2(head_dim) and head_dim >= 16, (
            f"head_dim={head_dim} does not satisfy flex_attention requirements "
            "(power of 2, >= 16); adjust c_s or num_heads so the TORCH_FLEX_ATTN "
            "path actually activates"
        )

    # 1D CP placements
    placements_single = [Shard(0), Shard(1)]
    placements_pair = [Shard(0), Shard(1)]
    placements_mask = [Shard(0), Shard(1)]

    # Distribute input tensors
    s_dtensor = distribute_tensor(
        s_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_single,
    ).requires_grad_(True)

    z_dtensor = distribute_tensor(
        z_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_pair,
    ).requires_grad_(True)

    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_mask,
    )

    has_k_in = k_in_global_host is not None
    if has_k_in:
        k_in_dtensor = distribute_tensor(
            k_in_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
        ).requires_grad_(True)
    else:
        k_in_dtensor = None

    # Distribute expected references for comparison
    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_single,
        src_data_rank=None,
    )
    d_output_dtensor = distribute_tensor(
        d_output_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh,
        placements=placements_single,
    )

    if check_error_hist:
        # Forward + backward
        output_dtensor = module(s_dtensor, z_dtensor, mask_dtensor, k_in=k_in_dtensor)
        output_dtensor.backward(d_output_dtensor)

        # Distribute serial fp32 references for histogram comparison
        output_fp32_dtensor = distribute_tensor(
            output_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        d_s_fp32_dtensor = distribute_tensor(
            d_s_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        d_z_fp32_dtensor = distribute_tensor(
            d_z_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )
        d_s_expected_dtensor = distribute_tensor(
            d_s_expected_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_single,
            src_data_rank=None,
        )
        d_z_expected_dtensor = distribute_tensor(
            d_z_expected_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=manager.device_mesh,
            placements=placements_pair,
            src_data_rank=None,
        )

        # Error histogram: output
        assert_no_percentile_upshift(
            output_dtensor.to_local(),
            output_expected_dtensor.to_local(),
            output_fp32_dtensor.to_local(),
            names_input=("output_cp_fp32", "output_serial_fp64", "output_serial_fp32"),
        )
        # Error histogram: s gradient
        assert_no_percentile_upshift(
            s_dtensor.grad.to_local(),
            d_s_expected_dtensor.to_local(),
            d_s_fp32_dtensor.to_local(),
            names_input=("d_s_cp_fp32", "d_s_serial_fp64", "d_s_serial_fp32"),
        )
        # Error histogram: z gradient
        assert_no_percentile_upshift(
            z_dtensor.grad.to_local(),
            d_z_expected_dtensor.to_local(),
            d_z_fp32_dtensor.to_local(),
            names_input=("d_z_cp_fp32", "d_z_serial_fp64", "d_z_serial_fp32"),
        )
        # Error histogram: k_in gradient
        if has_k_in:
            d_k_in_fp32_dtensor = distribute_tensor(
                d_k_in_global_fp32_host.to(device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_single,
                src_data_rank=None,
            )
            d_k_in_expected_dtensor = distribute_tensor(
                d_k_in_expected_global_host.to(dtype=dtype, device=manager.device),
                device_mesh=manager.device_mesh,
                placements=placements_single,
                src_data_rank=None,
            )
            assert_no_percentile_upshift(
                k_in_dtensor.grad.to_local(),
                d_k_in_expected_dtensor.to_local(),
                d_k_in_fp32_dtensor.to_local(),
                names_input=("d_k_in_cp_fp32", "d_k_in_serial_fp64", "d_k_in_serial_fp32"),
            )

        # Error histogram: parameter gradients
        for name, grad_param_expected_global in grad_params_expected_global_host.items():
            grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
            assert_no_percentile_upshift(
                grad_param_result_global,
                grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                grad_params_fp32_global_host[name],
                names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
            )
    else:
        # Forward pass
        output_dtensor = module(s_dtensor, z_dtensor, mask_dtensor, k_in=k_in_dtensor)

        # Forward parity: local shard comparison (no communication).
        # Bound: torch.testing.assert_close dtype defaults. Ring-attention
        # online-softmax error is O(cp_size * eps * N_per_chunk) per element
        # (FlashAttention §3.1; Ring Attention §4). For this geometry
        # (N_per_chunk=8 fp32 / 64 fp64, cp_size <= 4), the algorithmic bound
        # is below the defaults (atol_fp32=1e-5, atol_fp64=1e-7, rtol=1.3e-6).
        torch.testing.assert_close(
            output_dtensor.to_local(),
            output_expected_dtensor.to_local(),
        )

        # Forward parity: full tensor comparison (requires all-gather)
        output_global_result = output_dtensor.full_tensor().cpu()
        torch.testing.assert_close(
            output_global_result,
            output_expected_global_host.to(dtype=dtype),
        )

        # Backward pass
        output_dtensor.backward(d_output_dtensor)

        # Input gradient parity (full tensor). Bound: assert_close defaults.
        # Backward chain-rule depth J<=2 (softmax-Jacobian + LSE-backprop);
        # cp_size * eps * N_per_chunk * 2 stays below defaults in all cells.
        d_s_global_result = s_dtensor.grad.full_tensor().cpu()
        torch.testing.assert_close(
            d_s_global_result,
            d_s_expected_global_host.to(dtype=dtype),
        )

        # z gradient parity (full tensor)
        d_z_global_result = z_dtensor.grad.full_tensor().cpu()
        torch.testing.assert_close(
            d_z_global_result,
            d_z_expected_global_host.to(dtype=dtype),
        )

        # k_in gradient parity
        if has_k_in:
            d_k_in_global_result = k_in_dtensor.grad.full_tensor().cpu()
            torch.testing.assert_close(
                d_k_in_global_result,
                d_k_in_expected_global_host.to(dtype=dtype),
            )

        # Parameter gradient parity
        for name, grad_param_expected in grad_params_expected_global_host.items():
            param = dict(module.named_parameters())[name]
            assert param.grad is not None, f"Parameter {name} has no gradient"
            param_grad_result = param.grad.full_tensor().cpu()
            torch.testing.assert_close(
                param_grad_result,
                grad_param_expected.to(dtype=dtype),
            )
            # Replicated params should have identical gradients across cp ranks
            assert_all_identical(param.grad.full_tensor(), manager.group["cp"])

    # Guard against vacuous pass: verify cp sharding is active.
    # Only check the cp axis (mesh dim 1); dp=1 is valid and won't shard dim 0.
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
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert param.grad.to_local().abs().sum() > 0, f"Gradient for {name} is all zero"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cuda", "ENV"),
        ((1, 4), True, "cuda", "ENV"),
        ((2, 2), True, "cuda", "ENV"),
        ((1, 3), True, "cuda", "ENV"),
        ((1, 3), True, "cpu", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
@pytest.mark.parametrize(
    "sdpa_with_bias_backend",
    [SDPAWithBiasBackend.REFERENCE, SDPAWithBiasBackend.TORCH_FLEX_ATTN],
    ids=lambda x: x.value,
)
@pytest.mark.parametrize(
    "dtype_and_check_error_hist",
    [
        (torch.float32, False),
        (torch.float32, True),
        (torch.float64, False),
    ],
    ids=lambda x: f"dtype={x[0]}, check_error_hist={x[1]}",
)
@pytest.mark.parametrize(
    "use_separate_k_in",
    [False, True],
    ids=lambda x: f"k_in={'separate' if x else 'self'}",
)
@pytest.mark.parametrize(
    "compute_pair_bias",
    [True, False],
    ids=lambda x: f"pair_bias={'yes' if x else 'no'}",
)
def test_attention_pair_bias_1d(
    setup_env, sdpa_with_bias_backend, dtype_and_check_error_hist, use_separate_k_in, compute_pair_bias
):
    """Test AttentionPairBias1D parity against serial AttentionPairBias."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype, check_error_hist = dtype_and_check_error_hist

    if sdpa_with_bias_backend == SDPAWithBiasBackend.TORCH_FLEX_ATTN:
        if device_type != "cuda":
            pytest.skip("flex-attention requires CUDA")
        if dtype == torch.float64:
            pytest.skip("flex-attention does not support fp64")
        if not _flex_attn_compile_works():
            pytest.skip("torch.compile(flex_attention) has Inductor lowering bug on this PyTorch build")

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    if check_error_hist and grid_group_sizes["dp"] > 1:
        pytest.skip("skip error histogram check for dp > 1 to save test time")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    # Use large dims for error histogram and fp64 to test numerical stability;
    # smaller dims for fp32 exact-parity to detect logical bugs inexpensively.
    test_large_model = check_error_hist or dtype == torch.float64
    if test_large_model:
        N = cp_size * 64
        c_s = 64
        c_z = 32
        num_heads = 4  # head_dim = 64 // 4 = 16, required >= 16 for flex_attention
    else:
        N = cp_size * 8
        c_s = 32
        c_z = 16
        num_heads = 2  # head_dim = 32 // 2 = 16
    inf = 1e6

    seed_by_rank(0, seed=42)

    # Always compute fp64 reference (ground truth)
    serial_module = SerialAttentionPairBias(
        c_s=c_s,
        c_z=c_z if compute_pair_bias else None,
        num_heads=num_heads,
        inf=inf,
        compute_pair_bias=compute_pair_bias,
    )
    with torch.no_grad():
        for param in serial_module.parameters():
            param.uniform_(-5e-2, 5e-2)
    layer_state_dict_fp64 = serial_module.state_dict()

    serial_module = serial_module.to(dtype=torch.float64, device=device_type).train()

    # Create inputs in fp64
    z_dim = c_z if compute_pair_bias else num_heads
    s_global_fp64 = torch.empty(B, N, c_s, dtype=torch.float64, device=device_type)
    z_global_fp64 = torch.empty(B, N, N, z_dim, dtype=torch.float64, device=device_type)
    mask_global = torch.ones(B, N, dtype=torch.float64, device=device_type)
    mask_global[:, -2:] = 0.0

    with torch.no_grad():
        s_global_fp64.uniform_(-5e-2, 5e-2)
        z_global_fp64.uniform_(-5e-2, 5e-2)

    s_global_fp64 = s_global_fp64.requires_grad_(True)
    z_global_fp64 = z_global_fp64.requires_grad_(True)

    if use_separate_k_in:
        k_in_global_fp64 = torch.empty(B, N, c_s, dtype=torch.float64, device=device_type)
        with torch.no_grad():
            k_in_global_fp64.uniform_(-5e-2, 5e-2)
        k_in_global_fp64 = k_in_global_fp64.requires_grad_(True)
    else:
        k_in_global_fp64 = None

    # Serial fp64 forward + backward
    output_serial_fp64 = serial_module(
        s_global_fp64, z_global_fp64, mask_global, k_in=k_in_global_fp64 if use_separate_k_in else s_global_fp64
    )
    d_output_fp64 = torch.rand_like(output_serial_fp64)
    output_serial_fp64.backward(d_output_fp64)

    d_s_expected_fp64 = s_global_fp64.grad.detach().clone().cpu()
    d_z_expected_fp64 = z_global_fp64.grad.detach().clone().cpu()
    d_k_in_expected_fp64 = k_in_global_fp64.grad.detach().clone().cpu() if use_separate_k_in else None

    grad_params_expected_fp64 = {
        name: param.grad.detach().clone().cpu() for name, param in serial_module.named_parameters()
    }

    # Compute serial fp32 reference when check_error_hist is requested
    if check_error_hist:
        s_global_fp32 = s_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        z_global_fp32 = z_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        mask_global_fp32 = mask_global.detach().clone().to(dtype=torch.float32)

        if use_separate_k_in:
            k_in_global_fp32 = k_in_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        else:
            k_in_global_fp32 = None

        serial_module_fp32 = SerialAttentionPairBias(
            c_s=c_s,
            c_z=c_z if compute_pair_bias else None,
            num_heads=num_heads,
            inf=inf,
            compute_pair_bias=compute_pair_bias,
        )
        serial_module_fp32.load_state_dict(layer_state_dict_fp64)
        serial_module_fp32 = serial_module_fp32.to(dtype=torch.float32, device=device_type).train()

        output_serial_fp32 = serial_module_fp32(
            s_global_fp32,
            z_global_fp32,
            mask_global_fp32,
            k_in=k_in_global_fp32 if use_separate_k_in else s_global_fp32,
        )
        d_output_fp32 = d_output_fp64.to(dtype=torch.float32)
        output_serial_fp32.backward(d_output_fp32)

        output_global_fp32_host = output_serial_fp32.detach().clone().cpu()
        d_s_global_fp32_host = s_global_fp32.grad.detach().clone().cpu()
        d_z_global_fp32_host = z_global_fp32.grad.detach().clone().cpu()
        d_k_in_global_fp32_host = k_in_global_fp32.grad.detach().clone().cpu() if use_separate_k_in else None
        grad_params_fp32_global_host = {
            name: param.grad.detach().clone().cpu() for name, param in serial_module_fp32.named_parameters()
        }
    else:
        output_global_fp32_host = None
        d_s_global_fp32_host = None
        d_z_global_fp32_host = None
        d_k_in_global_fp32_host = None
        grad_params_fp32_global_host = None

    spawn_multiprocessing(
        _worker_attention_pair_bias_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        c_s,
        c_z,
        num_heads,
        inf,
        compute_pair_bias,
        layer_state_dict_fp64,
        s_global_fp64.detach().clone().cpu(),
        z_global_fp64.detach().clone().cpu(),
        mask_global.detach().clone().cpu(),
        k_in_global_fp64.detach().clone().cpu() if use_separate_k_in else None,
        output_serial_fp64.detach().clone().cpu(),
        d_output_fp64.detach().clone().cpu(),
        d_s_expected_fp64,
        d_k_in_expected_fp64,
        d_z_expected_fp64,
        grad_params_expected_fp64,
        sdpa_with_bias_backend,
        output_global_fp32_host,
        d_s_global_fp32_host,
        d_z_global_fp32_host,
        d_k_in_global_fp32_host,
        grad_params_fp32_global_host,
    )


def test_attention_pair_bias_1d_rejects_unsupported_backend():
    """AttentionPairBias1D must reject unsupported backends at init time."""
    serial_module = SerialAttentionPairBias(
        c_s=32,
        c_z=16,
        num_heads=2,
        inf=1e6,
        compute_pair_bias=True,
    )
    mock_mesh = MagicMock(spec=DeviceMesh)
    mock_cp_group = MagicMock()

    with pytest.raises(ValueError, match="Unsupported sdpa_with_bias_backend for 1D CP"):
        AttentionPairBias1D(
            serial_module,
            mock_mesh,
            mock_cp_group,
            sdpa_with_bias_backend=SDPAWithBiasBackend.TORCH_SDPA_EFFICIENT_ATTENTION,
        )
