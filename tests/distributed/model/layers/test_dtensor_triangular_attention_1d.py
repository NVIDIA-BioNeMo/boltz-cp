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

"""Parity tests for 1D CP triangular attention against serial reference.

Tests both starting and ending node variants, verifying:
- Forward pass matches serial fp64 reference
- Backward pass (input gradients + parameter gradients) matches serial
- DTensor shape/stride metadata is correct
- Sharding is active (local shape < global shape on sharded dim)
- Replicated parameter gradients are identical across CP ranks
- Gradients are non-zero (guards against vacuous pass)
"""

import warnings
from collections import OrderedDict

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.triangular_attention_1d import (
    TriangleAttentionEndingNode1D,
    TriangleAttentionStartingNode1D,
    _AllGatherTriangleAttentionStartingNode1DImpl,
    _DAPTriangleAttentionEndingNode1DImpl,
    cueq_is_installed,
)
from boltz.distributed.model.modules.utils import Precision, TriAttnBackend, setup_tf32_env
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.testing.utils import (
    assert_all_identical,
    assert_no_percentile_upshift,
    assert_tensors_identical,
    get_param_by_key,
    init_module_params_uniform,
    init_tensors_uniform,
    seed_by_rank,
    spawn_multiprocessing,
)


def parallel_assert_triangle_attention_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    c_in,
    c_hidden,
    no_heads,
    starting,
    layer_state_dict,
    input_x_global_host,
    mask_global_host,
    output_expected_global_host,
    d_output_expected_global_host,
    d_input_x_expected_global_host,
    grad_params_expected_global_host,
    triattn_backend,
    output_global_fp32_host=None,
    d_input_x_global_fp32_host=None,
    grad_params_fp32_global_host=None,
):
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    # CuEQ backward requires TF32 matmul (full FP32 backward is not implemented).
    # Use setup_tf32_env context manager to save/restore NVIDIA_TF32_OVERRIDE,
    # matmul.allow_tf32, and cudnn.allow_tf32 on exit (even if an exception occurs).
    precision = Precision.TF32 if triattn_backend == TriAttnBackend.CUEQ else Precision.FP32
    with setup_tf32_env(precision):
        DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
        manager = DistributedManager()

        if torch.finfo(dtype).resolution < torch.finfo(output_expected_global_host.dtype).resolution:
            raise ValueError(
                f"Target dtype {dtype} has higher precision than reference output's dtype "
                f"{output_expected_global_host.dtype}"
            )

        cp_group = manager.group["cp"]
        device_mesh = manager.device_mesh

        # Create serial module and load state dict
        if starting:
            module_serial = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
        else:
            module_serial = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)
        module_serial.load_state_dict(layer_state_dict)
        module_serial = module_serial.to(dtype=dtype, device=manager.device)

        # Create distributed module
        if starting:
            module = TriangleAttentionStartingNode1D(module_serial, device_mesh, cp_group)
        else:
            module = TriangleAttentionEndingNode1D(module_serial, device_mesh, cp_group)
        module = module.train()

        # 1D CP placements: (Shard(0), Shard(1)) on 2D mesh (dp, cp)
        placements_z = (Shard(0), Shard(1))

        input_x_dtensor = distribute_tensor(
            input_x_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
        ).requires_grad_(True)

        # mask_global_host is None for the mask_mode == "none" rows: pass mask=None
        # straight through to the module (exercises the wrapper's else-arm).
        if mask_global_host is not None:
            mask_dtensor = distribute_tensor(
                mask_global_host.to(dtype=dtype, device=manager.device),
                device_mesh=device_mesh,
                placements=placements_z,
            )
        else:
            mask_dtensor = None

        d_output_expected_dtensor = distribute_tensor(
            d_output_expected_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
        )
        output_expected_dtensor = distribute_tensor(
            output_expected_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
            src_data_rank=None,
        )
        d_input_x_expected_dtensor = distribute_tensor(
            d_input_x_expected_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
            src_data_rank=None,
        )

        # Verify sharding is active: local shape should be smaller than global.
        # Guarded by cp_size > 1 -- at cp_size==1 the mesh is single-rank so
        # local == global on the sharded dim (nothing to assert).  The cp=1 row
        # is the serial-math anchor; its non-vacuity is held by the gradient-/
        # output-non-zero checks below, not by sharding being active.
        cp_size = grid_group_sizes["cp"]
        if cp_size > 1:
            assert input_x_dtensor.to_local().shape[1] < input_x_dtensor.shape[1], (
                f"Sharding not active: local dim1 {input_x_dtensor.to_local().shape[1]} "
                f"should be < global dim1 {input_x_dtensor.shape[1]}"
            )

        check_error_hist = output_global_fp32_host is not None

        # Create copies to verify inputs aren't modified
        input_x_dtensor_copy = input_x_dtensor.detach().clone().requires_grad_(True)
        mask_dtensor_copy = mask_dtensor.detach().clone() if mask_dtensor is not None else None

        if check_error_hist:
            # Triangular attention backward has softmax + two matmul reductions
            # per head.  The CP all-reduce changes accumulation order, adding
            # O(cp_size * eps_fp32) rounding error.  At the 95th percentile the
            # error can exceed the default assert_close atol of 1e-05 by ~50%.
            # softmax+matmul backward accumulation reorder produces ~3x eps_fp32
            # error at 50th percentile for layernorm bias gradient. Observed
            # 3.05e-5 for cp=2.  Base 4e-05 covers this with margin.
            #
            # Starting-node (both REFERENCE and CUEQ) uses the DAP all-gather
            # impl, which assembles the full bias before a single softmax pass.
            # The accumulation order matches the serial reference (no cp-step
            # chunking), so no cp_size/2 scaling is needed.  Backward reduce-
            # scatter contributes ~cp_size * eps_fp32 (~1e-6 at cp=8), negligible
            # next to the 3e-5 forward error budget.  See item-2/item-3 specs
            # for the full derivation; tightened bound is 4e-05.
            #
            # Ending node (both REFERENCE and CUEQ post-Item-#5) uses the DAP
            # all-to-all path: a2a / all-gather / reduce-scatter are pure
            # data-movement, and the local full N x N softmax-attention (or
            # CUEQ kernel) matches the serial REFERENCE accumulation order.
            # Bound tightens to 4e-05 (constant, no cp_size scaling).  See
            # src/boltz/distributed/docs/dap-migration/item-4-triattn-end-qkv-a2a.md
            # and item-5-triattn-end-cueq-adapter.md for the full derivation.
            hist_atol = 4e-05
            perc = OrderedDict(
                {0.25: (hist_atol, None), 0.5: (hist_atol, None), 0.75: (hist_atol, None), 0.95: (hist_atol, None)}
            )

            # Forward and backward pass for error histogram checking
            output_dtensor_result = module(input_x_dtensor, mask_dtensor, triattn_backend=triattn_backend)
            output_dtensor_result.backward(d_output_expected_dtensor)

            output_fp32_dtensor = distribute_tensor(
                output_global_fp32_host.to(device=manager.device),
                device_mesh=device_mesh,
                placements=placements_z,
                src_data_rank=None,
            )
            d_input_x_fp32_dtensor = distribute_tensor(
                d_input_x_global_fp32_host.to(device=manager.device),
                device_mesh=device_mesh,
                placements=placements_z,
                src_data_rank=None,
            )

            # Check output DTensor metadata
            assert output_dtensor_result.shape == output_expected_dtensor.shape, (
                f"Output DTensor has shape {output_dtensor_result.shape} "
                f"but expected shape {output_expected_dtensor.shape}"
            )
            assert output_dtensor_result.stride() == output_expected_dtensor.stride(), (
                f"Output DTensor has stride {output_dtensor_result.stride()} "
                f"but expected stride {output_expected_dtensor.stride()}"
            )
            assert input_x_dtensor.grad.shape == d_input_x_expected_dtensor.shape, (
                f"Input DTensor grad has shape {input_x_dtensor.grad.shape} "
                f"but expected shape {d_input_x_expected_dtensor.shape}"
            )
            assert input_x_dtensor.grad.stride() == d_input_x_expected_dtensor.stride(), (
                f"Input DTensor grad has stride {input_x_dtensor.grad.stride()} "
                f"but expected stride {d_input_x_expected_dtensor.stride()}"
            )

            assert_no_percentile_upshift(
                output_dtensor_result.to_local(),
                output_expected_dtensor.to_local(),
                output_fp32_dtensor.to_local(),
                perc=perc,
                names_input=("output_cp_fp32", "output_serial_fp64", "output_serial_fp32"),
            )
            assert_no_percentile_upshift(
                input_x_dtensor.grad.to_local(),
                d_input_x_expected_dtensor.to_local(),
                d_input_x_fp32_dtensor.to_local(),
                perc=perc,
                names_input=("d_input_x_cp_fp32", "d_input_x_serial_fp64", "d_input_x_serial_fp32"),
            )

            # Check parameter gradients error histograms
            for name, grad_param_expected_global in grad_params_expected_global_host.items():
                grad_param_result_global = get_param_by_key(module, name).grad.full_tensor().cpu()
                assert_no_percentile_upshift(
                    grad_param_result_global,
                    grad_param_expected_global.to(dtype=grad_param_result_global.dtype),
                    grad_params_fp32_global_host[name],
                    perc=perc,
                    names_input=(f"d_{name}_cp_fp32", f"d_{name}_serial_fp64", f"d_{name}_serial_fp32"),
                )
        else:
            # Dtype-aware tolerances for fp32 vs fp64.
            # fp64: use torch defaults (~1.3e-6 rtol, 1e-5 atol).
            # fp32: for N=8 tokens (cp=2) and c_hidden=4, the softmax+matmul attention chain
            # has worst-case relative error O(N * eps_fp32) = 8 * 1.2e-7 ≈ 1e-6 forward.
            # Backward adds one more matmul reduction, giving ~1e-5.
            # We use 10x margin: atol=1e-5, rtol=1e-4 forward; atol=1e-4, rtol=1e-3 backward.
            if dtype == torch.float64:
                fwd_atol, fwd_rtol = None, None  # use torch defaults (tight)
                bwd_atol, bwd_rtol = None, None
            else:
                fwd_atol, fwd_rtol = 1e-5, 1e-4
                bwd_atol, bwd_rtol = 1e-4, 1e-3

            # Forward pass
            output_dtensor_result = module(input_x_dtensor, mask_dtensor, triattn_backend=triattn_backend)

            # Check output DTensor metadata
            assert output_dtensor_result.shape == output_expected_dtensor.shape, (
                f"Output DTensor has shape {output_dtensor_result.shape} "
                f"but expected shape {output_expected_dtensor.shape}"
            )
            assert output_dtensor_result.stride() == output_expected_dtensor.stride(), (
                f"Output DTensor has stride {output_dtensor_result.stride()} "
                f"but expected stride {output_expected_dtensor.stride()}"
            )

            # Verify inputs weren't modified
            assert_tensors_identical(
                input_x_dtensor_copy.to_local(),
                input_x_dtensor.to_local(),
                check_grad=False,
                check_grad_fn=False,
            )
            if mask_dtensor is not None:
                assert_tensors_identical(mask_dtensor_copy.to_local(), mask_dtensor.to_local())

            # Backward pass
            d_output_expected_dtensor_copy = d_output_expected_dtensor.detach().clone()
            output_dtensor_result.backward(d_output_expected_dtensor)

            assert input_x_dtensor.grad.shape == d_input_x_expected_dtensor.shape, (
                f"Input DTensor grad has shape {input_x_dtensor.grad.shape} "
                f"but expected shape {d_input_x_expected_dtensor.shape}"
            )
            assert input_x_dtensor.grad.stride() == d_input_x_expected_dtensor.stride(), (
                f"Input DTensor grad has stride {input_x_dtensor.grad.stride()} "
                f"but expected stride {d_input_x_expected_dtensor.stride()}"
            )

            # Verify upstream gradient wasn't modified
            assert_tensors_identical(
                d_output_expected_dtensor_copy.to_local(),
                d_output_expected_dtensor.to_local(),
            )

            # Verify gradients are non-zero and finite (guards against vacuous pass)
            assert input_x_dtensor.grad.to_local().abs().max() > 0, "Input gradient is all zeros"
            assert output_dtensor_result.to_local().abs().max() > 0, "Output is all zeros"
            assert output_dtensor_result.to_local().isfinite().all(), "Output has non-finite values"
            assert input_x_dtensor.grad.to_local().isfinite().all(), "Input gradient has non-finite values"

            if triattn_backend == TriAttnBackend.REFERENCE:
                # REFERENCE: tight numerical parity against same-kernel serial reference.
                torch.testing.assert_close(
                    output_dtensor_result.to_local(),
                    output_expected_dtensor.to_local(),
                    atol=fwd_atol,
                    rtol=fwd_rtol,
                )
                torch.testing.assert_close(
                    input_x_dtensor.grad.to_local(),
                    d_input_x_expected_dtensor.to_local(),
                    atol=bwd_atol,
                    rtol=bwd_rtol,
                )
                # Full tensor gathering - verify distributed results match serial
                output_global_result_host = output_dtensor_result.full_tensor().cpu()
                d_input_x_global_result_host = input_x_dtensor.grad.full_tensor().cpu()
                torch.testing.assert_close(
                    output_global_result_host,
                    output_expected_global_host.to(dtype=dtype),
                    atol=fwd_atol,
                    rtol=fwd_rtol,
                )
                torch.testing.assert_close(
                    d_input_x_global_result_host,
                    d_input_x_expected_global_host.to(dtype=dtype),
                    atol=bwd_atol,
                    rtol=bwd_rtol,
                )
            # CuEQ: skip cross-kernel numerical parity (CuEQ kernel + TF32 backward differs
            # fundamentally from REFERENCE serial). The test verifies the code path completes,
            # outputs are finite/non-zero, and shape/stride metadata is correct.

            # Test parameter gradients
            grad_params_result_dtensors = {}
            for name, param in module.named_parameters():
                if param.grad is not None:
                    if name not in grad_params_expected_global_host:
                        raise ValueError(
                            f"Parameter {name} has a resulting gradient but is not in the reference module"
                        )
                    grad_params_result_dtensors[name] = param.grad

            for name, grad_param_expected_global_host in grad_params_expected_global_host.items():
                assert (
                    name in grad_params_result_dtensors
                ), f"Parameter {name}'s gradient is not found in result gradients"
                grad_params_result = grad_params_result_dtensors[name]
                grad_params_result_global = grad_params_result.full_tensor()
                # Verify param grads are finite and non-zero
                assert grad_params_result_global.isfinite().all(), f"Parameter {name} gradient has non-finite values"
                assert grad_params_result_global.abs().max() > 0, f"Parameter {name} gradient is all zeros"
                if triattn_backend == TriAttnBackend.REFERENCE:
                    torch.testing.assert_close(
                        grad_params_result_global.cpu(),
                        grad_param_expected_global_host.to(dtype=dtype),
                        atol=bwd_atol,
                        rtol=bwd_rtol,
                    )
                # Replicated parameter gradients must be identical across CP ranks
                assert_all_identical(grad_params_result_global, cp_group)

        DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, check_error_hist, mask_mode",
    (
        params_test := [
            # mask_mode (4th element) selects the attention mask passed to BOTH the
            # serial reference and the distributed module:
            #   "random" -> a random 0/1 mask with emulated padding blocks (the
            #               wrapper's ``mask is not None`` arm; mask is resharded).
            #   "none"   -> mask=None (the wrapper's ELSE arm: it builds an all-ones
            #               mask locally and the ending node SKIPS the mask-reshard
            #               a2a).  These are DISTINCT code paths, so each needs its
            #               own row -- a random mask does not subsume mask=None.
            #
            # cp=1 single-rank serial-math anchor (CPU/gloo, runs locally with no
            # GPU).  At cp_size==1 every collective short-circuits to its no-op
            # fast-path, so this row pins the LOCAL fused math against the serial
            # reference; the cp>1 rows below gate the distributed collectives.
            # The sharding-active assert is guarded by ``cp_size > 1`` (a 1-rank
            # mesh has local==global, nothing to assert) -- non-vacuity is held
            # by the gradient-non-zero / output-non-zero checks that run for
            # every cp.  The cp=1 row is run with BOTH mask modes so the
            # serial-math anchor covers the else-arm too (subsumes the retired
            # cp=1 mask=None smoke).
            (((1, 1), True, "cpu", "ENV"), torch.float64, False, "random"),
            (((1, 1), True, "cpu", "ENV"), torch.float64, False, "none"),
            (((1, 2), True, "cuda", "ENV"), torch.float32, True, "random"),
            # cp>1 mask=None coverage on GPU: the ending-node else-arm skips the
            # mask-reshard a2a, a distributed path the random-mask rows never hit.
            (((1, 2), True, "cuda", "ENV"), torch.float32, False, "none"),
            (((2, 2), True, "cuda", "ENV"), torch.float32, True, "random"),
            (((1, 3), True, "cuda", "ENV"), torch.float32, False, "random"),
            (((1, 3), True, "cuda", "ENV"), torch.float64, False, "random"),
            (((1, 4), True, "cuda", "ENV"), torch.float32, False, "random"),
        ]
    ),
    indirect=["setup_env"],
    ids=[
        f"dp:{x[0][0][0]}, cp:{x[0][0][1]}, device_type:{x[0][2]}, dtype:{x[1]}, "
        f"check_error_hist:{x[2]}, mask:{x[3]}"
        for x in params_test
    ],
)
@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
@pytest.mark.parametrize(
    "triattn_backend",
    [TriAttnBackend.REFERENCE, TriAttnBackend.CUEQ],
    ids=lambda x: x.value,
)
def test_triangle_attention_1d_parallel(setup_env, dtype, check_error_hist, mask_mode, starting, triattn_backend):
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    use_mask = mask_mode == "random"

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    if triattn_backend == TriAttnBackend.CUEQ:
        if device_type != "cuda":
            pytest.skip("CuEQ requires CUDA")
        if not cueq_is_installed:
            pytest.skip("cuequivariance_torch not installed")
        if dtype == torch.float64:
            pytest.skip("CuEQ does not support fp64")

    if check_error_hist and grid_group_sizes["dp"] > 1:
        pytest.skip("skip error histogram check for dp > 1 to save test time")
    if check_error_hist and triattn_backend == TriAttnBackend.CUEQ:
        pytest.skip(
            "CuEQ uses TF32 matmul in backward, producing fundamentally different "
            "accumulation errors than the FP32 serial reference — error histogram "
            "comparison is not meaningful across kernel backends"
        )

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    # For float64 and error histogram check, use realistic model and input size
    # with heavier computation to test numerical stability. Smaller dims for
    # other cases to detect logical bugs inexpensively.
    test_large_model = check_error_hist or dtype == torch.float64
    if test_large_model:
        N = cp_size * 64
        c_in = 128
        c_hidden = 32
        no_heads = 4
        # Narrow init range to keep d_layer_norm.bias gradient within the
        # 4e-5 * max(1, cp_size/2) fp32 histogram tolerance.
        # At ±5e-2, the 75th-percentile gradient was ~1.22e-4 (≈3× over).
        # Gradient through two weight matrices scales as weight_scale²; reducing
        # ±5e-2 → ±1e-2 (5× narrower) drops it by ~25×, leaving ample margin
        # while keeping gradients non-zero (≫ fp32 epsilon ≈ 1.2e-7).
        min_val_init = -1e-2
        max_val_init = 1e-2
    else:
        N = cp_size * 4
        c_in = 16
        c_hidden = 4
        no_heads = 2
        min_val_init = -0.5
        max_val_init = 0.5

    seed = 42
    seed_by_rank(0, seed=seed)

    # Compute reference results with FP64
    input_x_global_fp64 = torch.empty((B, N, N, c_in), dtype=torch.float64, requires_grad=True, device=device_type)
    if use_mask:
        mask_global_fp64 = torch.randint(0, 2, (B, N, N), dtype=torch.float64, requires_grad=False, device=device_type)
        # Emulate blocks of pure padding
        mask_global_fp64[0, N // cp_size :, :] = 0
        mask_global_fp64[0, :, N // cp_size :] = 0
    else:
        # mask_mode == "none": exercise the wrapper's mask-is-None else-arm.
        mask_global_fp64 = None

    if starting:
        reference_module = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
    else:
        reference_module = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)

    init_tensors_uniform([input_x_global_fp64], low=min_val_init, high=max_val_init)
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)

    layer_state_dict_fp64 = reference_module.state_dict()
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).train()

    # Run forward pass in fp64 to get reference
    output_expected_global_fp64 = reference_module(input_x_global_fp64, mask_global_fp64)
    d_output_expected_global_fp64 = torch.rand_like(output_expected_global_fp64)
    output_expected_global_fp64.backward(d_output_expected_global_fp64)

    grad_params_fp64_expected_global_host = {
        name: param.grad.detach().clone().cpu() for name, param in reference_module.named_parameters()
    }

    if check_error_hist:
        # Compute serial fp32 reference for error histogram comparison
        input_x_global_fp32 = input_x_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        mask_global_fp32 = (
            mask_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(False) if use_mask else None
        )

        if starting:
            reference_module_fp32 = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
        else:
            reference_module_fp32 = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)

        reference_module_fp32.load_state_dict(layer_state_dict_fp64)
        reference_module_fp32 = reference_module_fp32.to(dtype=torch.float32, device=device_type).train()

        output_global_fp32 = reference_module_fp32(input_x_global_fp32, mask_global_fp32)
        d_output_expected_global_fp32 = d_output_expected_global_fp64.to(dtype=torch.float32)
        output_global_fp32.backward(d_output_expected_global_fp32)

        output_global_fp32_host = output_global_fp32.detach().clone().cpu()
        d_input_x_global_fp32_host = input_x_global_fp32.grad.detach().clone().cpu()
        grad_params_fp32_global_host = {
            name: param.grad.detach().clone().cpu() for name, param in reference_module_fp32.named_parameters()
        }
    else:
        output_global_fp32_host = None
        d_input_x_global_fp32_host = None
        grad_params_fp32_global_host = None

    if dtype == torch.float64:
        # Use fp64 reference directly
        ref_state_dict = layer_state_dict_fp64
        ref_input = input_x_global_fp64.detach().clone().cpu()
        ref_mask = mask_global_fp64.detach().clone().cpu() if use_mask else None
        ref_output = output_expected_global_fp64.detach().clone().contiguous().cpu()
        ref_d_output = d_output_expected_global_fp64.detach().clone().contiguous().cpu()
        ref_d_input = input_x_global_fp64.grad.detach().clone().contiguous().cpu()
        ref_grad_params = grad_params_fp64_expected_global_host
    else:
        # Compute fp32 reference to avoid cross-precision tolerance blow-up
        if check_error_hist:
            # Reuse the fp32 reference already computed above
            ref_input = input_x_global_fp32.detach().clone().cpu()
            ref_mask = mask_global_fp32.detach().clone().cpu() if use_mask else None
            ref_output = output_global_fp32.detach().clone().contiguous().cpu()
            ref_d_output = d_output_expected_global_fp32.detach().clone().contiguous().cpu()
            ref_d_input = input_x_global_fp32.grad.detach().clone().contiguous().cpu()
            ref_grad_params = {
                name: param.grad.detach().clone().cpu() for name, param in reference_module_fp32.named_parameters()
            }
        else:
            input_x_global_fp32 = input_x_global_fp64.detach().clone().to(dtype=dtype).requires_grad_(True)
            mask_global_fp32 = mask_global_fp64.detach().clone().to(dtype=dtype) if use_mask else None
            if starting:
                ref_module_fp32 = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
            else:
                ref_module_fp32 = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)
            ref_module_fp32.load_state_dict(layer_state_dict_fp64)
            ref_module_fp32 = ref_module_fp32.to(dtype=dtype, device=device_type).train()
            output_fp32 = ref_module_fp32(input_x_global_fp32, mask_global_fp32)
            d_output_fp32 = d_output_expected_global_fp64.to(dtype=dtype)
            output_fp32.backward(d_output_fp32)

            ref_input = input_x_global_fp32.detach().clone().cpu()
            ref_mask = mask_global_fp32.detach().clone().cpu() if use_mask else None
            ref_output = output_fp32.detach().clone().contiguous().cpu()
            ref_d_output = d_output_fp32.detach().clone().contiguous().cpu()
            ref_d_input = input_x_global_fp32.grad.detach().clone().contiguous().cpu()
            ref_grad_params = {
                name: param.grad.detach().clone().cpu() for name, param in ref_module_fp32.named_parameters()
            }
        ref_state_dict = layer_state_dict_fp64

    # Launch parallel test across all processes
    spawn_multiprocessing(
        parallel_assert_triangle_attention_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        c_in,
        c_hidden,
        no_heads,
        starting,
        ref_state_dict,
        ref_input,
        ref_mask,
        # The serial ending node output may be non-contiguous (internal transpose).
        # Make contiguous so distribute_tensor produces matching DTensor strides.
        ref_output,
        ref_d_output,
        ref_d_input,
        ref_grad_params,
        triattn_backend,
        output_global_fp32_host,
        d_input_x_global_fp32_host,
        grad_params_fp32_global_host,
    )


# ---------------------------------------------------------------------------
# test_triangle_attention_parallel_sm100f_1d
# Mirrors 2D test_triangle_attention_parallel_sm100f on the 1D starting-node
# DAP all-gather path.  See parallel_assert_sm100f_bwd_warning (2D) for the
# reference implementation.
# ---------------------------------------------------------------------------

SM100F_BWD_WARNING_SUBSTR = "SM100f kernel expects bias to be of the same dtype as q"


def parallel_assert_sm100f_bwd_warning_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    c_in,
    c_hidden,
    no_heads,
    layer_state_dict,
    input_x_global_host,
    mask_global_host,
    d_output_global_host,
    expect_warning,
    mock_util_always_false,
):
    """Worker: run 1D distributed forward+backward and check SM100f warning.

    Mirrors parallel_assert_sm100f_bwd_warning from test_dtensor_triangle_attention.py
    on the 1D starting-node DAP all-gather path with (Shard(0), Shard(1)) placements.
    """
    import sys

    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    if mock_util_always_false:
        triattn_mod = sys.modules["boltz.distributed.model.layers.triangular_attention"]
        monkeypatch.setattr(triattn_mod, "can_run_cueq_triattn_sm100f", lambda *_args, **_kw: False)

    dtype = torch.bfloat16
    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    cp_group = manager.group["cp"]
    device_mesh = manager.device_mesh

    module_serial = TriangleAttentionStartingNode(c_in, c_hidden, no_heads, inf=1e9)
    module_serial = module_serial.to(dtype=dtype)
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.to(device=manager.device)

    module = TriangleAttentionStartingNode1D(module_serial, device_mesh, cp_group)
    module = module.to(device=manager.device).train()

    # 1D CP placements: (Shard(0), Shard(1)) on 2D mesh (dp, cp)
    placements = (Shard(0), Shard(1))

    input_x = distribute_tensor(
        input_x_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements,
    ).requires_grad_(True)
    mask = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements,
    )
    d_output = distribute_tensor(
        d_output_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements,
    )

    output = module(input_x, mask, triattn_backend=TriAttnBackend.CUEQ)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output.backward(d_output)

    sm100f_msgs = [w for w in caught if SM100F_BWD_WARNING_SUBSTR in str(w.message)]
    if expect_warning:
        assert sm100f_msgs, f"Rank {rank}: expected SM100f bwd warning but none was emitted"
    else:
        assert not sm100f_msgs, f"Rank {rank}: SM100f bwd warning(s) emitted ({len(sm100f_msgs)}): " + "; ".join(
            str(w.message) for w in sm100f_msgs
        )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 4), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=["dp:1-cp:4-cuda"],
)
@pytest.mark.parametrize(
    "use_util_to_condition_fp32_cast",
    [True, False],
    ids=["util_active", "util_mocked_false"],
)
def test_triangle_attention_parallel_sm100f_1d(setup_env, use_util_to_condition_fp32_cast):
    """Assert SM100f backward warning fires on the 1D starting-node DAP path when util returns True, absent when mocked False."""
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not cueq_is_installed:
        pytest.skip("cuequivariance_torch is not installed")
    device_cc = torch.cuda.get_device_capability()
    if device_cc not in ((10, 0), (10, 3)):
        pytest.skip(f"GPU compute capability {device_cc} is not SM100/SM103")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    N = cp_size * 8
    c_in = 8
    c_hidden = 8
    no_heads = 2
    seed = 42
    seed_by_rank(0, seed=seed)

    input_x_global = torch.empty((B, N, N, c_in), dtype=torch.float64, device="cuda")
    mask_global = torch.ones((B, N, N), dtype=torch.float64, device="cuda")
    init_tensors_uniform([input_x_global], low=-0.5, high=0.5)

    reference_module = TriangleAttentionStartingNode(c_in, c_hidden, no_heads, inf=1e9)
    init_module_params_uniform(reference_module, low=-0.5, high=0.5)
    reference_module = reference_module.to(dtype=torch.float64, device="cuda")
    layer_state_dict = reference_module.state_dict()

    d_output_global = torch.rand((B, N, N, c_in), dtype=torch.float64, device="cuda")

    # True: real util pre-casts bias to q.dtype -> cuEq sees correct dtype -> no warning.
    # False: util mocked -> bias cast to fp32 -> cuEq internally detects SM100f and warns.
    mock_util_always_false = not use_util_to_condition_fp32_cast
    expect_warning = mock_util_always_false

    spawn_multiprocessing(
        parallel_assert_sm100f_bwd_warning_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        c_in,
        c_hidden,
        no_heads,
        layer_state_dict,
        input_x_global.detach().clone().cpu(),
        mask_global.detach().clone().cpu(),
        d_output_global.detach().clone().cpu(),
        expect_warning,
        mock_util_always_false,
    )


# ---------------------------------------------------------------------------
# test_all_gather_starting_node_1d_gloo_and_memory_budget
# Drives _AllGatherTriangleAttentionStartingNode1DImpl directly on CPU/Gloo.
# Two things at once:
#   1. Memory-budget invariant: the saved tensors must hold only the local
#      bias slab [..., N/cp, N], NOT the gathered bias [..., N, N].
#      Saving the full bias would violate the O(N^2/cp) backward budget.
#   2. Gloo backend fallback: the backward reduce-scatter must use the
#      all_reduce + slice path on Gloo (which lacks reduce_scatter).
# Also asserts forward parity vs serial REFERENCE attention and finite
# nonzero gradients.
# ---------------------------------------------------------------------------


def parallel_assert_all_gather_starting_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    q_global_host,
    k_global_host,
    v_global_host,
    mask_bias_global_host,
    triangle_bias_global_host,
    d_output_global_host,
    out_serial_host,
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
    cp_group = manager.group["cp"]
    cp_size = grid_group_sizes["cp"]
    cp_rank = dist.get_rank(cp_group)

    # Confirm we are on the Gloo backend (this test's whole point).
    assert dist.get_backend(cp_group) == "gloo", f"Rank {rank}: expected gloo backend, got {dist.get_backend(cp_group)}"

    B, _, no_heads, N, c_hidden = q_global_host.shape
    assert N % cp_size == 0, f"N={N} not divisible by cp_size={cp_size}"
    n_local = N // cp_size
    i_start = cp_rank * n_local
    i_end = i_start + n_local

    dtype = torch.float32
    device = manager.device

    # Local shards along I (== J_q == N).
    q_local = q_global_host[:, i_start:i_end].to(dtype=dtype, device=device).clone().requires_grad_(True)
    k_local = k_global_host[:, i_start:i_end].to(dtype=dtype, device=device).clone().requires_grad_(True)
    v_local = v_global_host[:, i_start:i_end].to(dtype=dtype, device=device).clone().requires_grad_(True)
    mask_bias_local = mask_bias_global_host[:, i_start:i_end].to(dtype=dtype, device=device).clone()
    triangle_bias_local = (
        triangle_bias_global_host[..., i_start:i_end, :].to(dtype=dtype, device=device).clone().requires_grad_(True)
    )

    q_scale = c_hidden**-0.5
    output = _AllGatherTriangleAttentionStartingNode1DImpl.apply(
        q_local * q_scale,
        k_local,
        v_local,
        mask_bias_local,
        triangle_bias_local,
        cp_group,
        True,  # apply_scale (pre-scaled above)
        q_scale,
    )

    # Memory-budget invariant on the autograd graph: only the local slab
    # should be saved for backward, NOT the gathered [..., N, N] bias.
    saved = output.grad_fn.saved_tensors
    # Last saved is triangle_bias_local per the impl's save_for_backward order.
    saved_bias = saved[-1]
    expected_slab_shape = (B, 1, no_heads, n_local, N)
    assert saved_bias.shape == expected_slab_shape, (
        f"Rank {rank}: saved bias shape {saved_bias.shape} != expected slab shape "
        f"{expected_slab_shape}.  bias_full would be {(B, 1, no_heads, N, N)} — saving "
        f"it would violate the O(N^2/cp) backward budget."
    )

    # Forward parity vs serial REFERENCE.
    out_serial_local = out_serial_host[:, i_start:i_end].to(dtype=dtype, device=device)
    torch.testing.assert_close(output.detach(), out_serial_local, atol=1e-5, rtol=1e-4)

    # Backward exercises the Gloo reduce-scatter fallback (all_reduce + slice).
    d_out_local = d_output_global_host[:, i_start:i_end].to(dtype=dtype, device=device).clone()
    output.backward(d_out_local)

    for name, tensor in [
        ("q_local", q_local),
        ("k_local", k_local),
        ("v_local", v_local),
        ("triangle_bias_local", triangle_bias_local),
    ]:
        assert tensor.grad is not None, f"Rank {rank}: {name}.grad is None"
        assert tensor.grad.isfinite().all(), f"Rank {rank}: {name}.grad has non-finite values"
        assert tensor.grad.abs().max() > 0, f"Rank {rank}: {name}.grad is all zeros"

    # dbias_local must have the slab shape (reduce-scatter / slice produces
    # the per-rank slab, not the full bias).
    assert (
        triangle_bias_local.grad.shape == expected_slab_shape
    ), f"Rank {rank}: dbias shape {triangle_bias_local.grad.shape} != {expected_slab_shape}"

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [((1, 3), True, "cpu", "ENV")],
    indirect=["setup_env"],
    ids=["dp:1-cp:3-cpu"],
)
def test_all_gather_starting_node_1d_gloo_and_memory_budget(setup_env):
    """Memory-budget invariant + Gloo reduce-scatter fallback for the all-gather
    starting-node implementation.

    Drives _AllGatherTriangleAttentionStartingNode1DImpl.apply() directly with
    CPU/Gloo at cp=3.  Asserts:
      1. ctx.saved_tensors holds the local bias slab [B,1,H,N/cp,N], NOT the
         gathered bias [B,1,H,N,N] — protects the O(N^2/cp) backward budget.
      2. Backward completes correctly on Gloo (exercises the
         all_reduce + slice fallback in the reduce-scatter path).
      3. Forward output matches the serial REFERENCE attention at the local
         slab.

    The standard `test_triangle_attention_1d_parallel` covers the CUDA/NCCL
    reduce_scatter path and end-to-end module parity; this test isolates the
    memory invariant and the Gloo branch which is not exercised elsewhere.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    cp_size = grid_group_sizes["cp"]
    B = 2
    no_heads = 2
    c_hidden = 4
    N = cp_size * 4
    seed = 42
    seed_by_rank(0, seed=seed)

    g = torch.Generator().manual_seed(seed)
    # Pre-scaled q convention — caller scales before passing to the impl.
    q_global = torch.empty((B, N, no_heads, N, c_hidden), dtype=torch.float64).uniform_(-0.1, 0.1, generator=g)
    k_global = torch.empty((B, N, no_heads, N, c_hidden), dtype=torch.float64).uniform_(-0.1, 0.1, generator=g)
    v_global = torch.empty((B, N, no_heads, N, c_hidden), dtype=torch.float64).uniform_(-0.1, 0.1, generator=g)
    mask_bias_global = torch.zeros(B, N, 1, 1, N, dtype=torch.float64)
    triangle_bias_global = torch.empty((B, 1, no_heads, N, N), dtype=torch.float64).uniform_(-0.1, 0.1, generator=g)
    d_output_global = torch.empty((B, N, no_heads, N, c_hidden), dtype=torch.float64).uniform_(-1.0, 1.0, generator=g)

    # Build the serial REFERENCE expected output (without any kernel) so we
    # can verify forward parity at the per-rank slab.
    q_scaled = q_global.to(torch.float32) * (c_hidden**-0.5)
    k32 = k_global.to(torch.float32)
    v32 = v_global.to(torch.float32)
    mb32 = mask_bias_global.to(torch.float32)
    tb32 = triangle_bias_global.to(torch.float32)
    attn = torch.matmul(q_scaled, k32.transpose(-1, -2)) + mb32 + tb32
    attn = torch.softmax(attn, dim=-1)
    out_serial = torch.matmul(attn, v32)

    spawn_multiprocessing(
        parallel_assert_all_gather_starting_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        q_global,
        k_global,
        v_global,
        mask_bias_global,
        triangle_bias_global,
        d_output_global,
        out_serial,
    )


# ---------------------------------------------------------------------------
# test_triangle_attention_cueq_vs_reference_parity_1d
# Cross-kernel numerical-parity check: same module + same inputs run through
# both REFERENCE and CUEQ backends, comparing forward output and input grads.
#
# Why: parallel_assert_triangle_attention_1d only asserts CUEQ via "kernel
# doesn't crash + finite/non-zero outputs + shape contract".  A broken slice
# inside the CUEQ branch could still produce finite outputs and pass.  This
# test cross-validates the two backends against each other on the DAP path
# (starting-node + ending-node).
#
# Tolerance derivation (kernel-difference budget):
#   - REFERENCE runs plain FP32 matmul (TF32 disabled).
#   - CUEQ requires TF32 matmul in backward (full-FP32 backward is not
#     implemented).  TF32 keeps 10 mantissa bits of the inputs and accumulates
#     in FP32, so per-matmul relative error is ~2^-10 ~= 1e-3.
#   - The softmax+matmul attention chain has worst-case relative error
#     O(N * tf32_eps) where tf32_eps ~= 1e-3.  At N=8 (cp=2 with N/cp=4),
#     that gives ~1e-2 relative on the forward.  We use atol=2e-2, rtol=2e-2
#     forward (2x margin) and atol=5e-2, rtol=5e-2 backward (one extra matmul
#     pass).
# ---------------------------------------------------------------------------


def parallel_assert_triangle_attention_cueq_vs_reference_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    c_in,
    c_hidden,
    no_heads,
    starting,
    layer_state_dict,
    input_x_global_host,
    mask_global_host,
    d_output_global_host,
):
    """Worker: run both REFERENCE and CUEQ backends on the same module/inputs
    and compare outputs + input gradients within a TF32-kernel-difference budget.

    Parameters
    ----------
    layer_state_dict
        FP32 reference state dict — both backends load this verbatim.
    input_x_global_host, mask_global_host, d_output_global_host
        Identical inputs / upstream grad for both backends.
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    dtype = torch.float32
    # Tolerance derivation -- TF32 (CUEQ) vs FP32 (REFERENCE) on the same module.
    # TF32 keeps 10 mantissa bits of inputs and accumulates in FP32, so each
    # matmul has ~2^-10 ~= 1e-3 relative error.  The softmax+matmul attention
    # chain at N<=8 accumulates to:
    #   - forward output: empirically max_abs ~1.5e-4 at ref_max ~0.4
    #     (~4e-4 relative).  Bound 1e-3 gives ~3x margin.
    #   - input grad: one extra matmul pass; max_abs ~1e-3 at ref_max ~1
    #     (~1e-3 relative).  Bound 5e-3 gives ~5x margin.
    #   - param grad: layer_norm.bias is the worst-case accumulator (small
    #     bias absorbing the full gradient sum); max_abs ~6e-3 at ref_max ~7
    #     (~1e-3 relative).  Bound 1e-2 + rtol 5e-3 covers it with ~2x margin.
    # The mask is all-ones to isolate kernel-math differences from the
    # boolean-vs-additive mask divergence between backends (mask-equivalence
    # belongs in a separate test).
    fwd_atol, fwd_rtol = 1e-3, 1e-3
    bwd_atol_input, bwd_rtol_input = 5e-3, 5e-3
    bwd_atol_param, bwd_rtol_param = 1e-2, 5e-3

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()
    cp_group = manager.group["cp"]
    device_mesh = manager.device_mesh

    # 1D CP placements: (Shard(0), Shard(1)) on 2D mesh (dp, cp)
    placements_z = (Shard(0), Shard(1))

    # Build a single serial module and load state dict; both backends share
    # the same wrapped distributed module (state is identical across runs).
    if starting:
        module_serial = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
    else:
        module_serial = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.to(dtype=dtype, device=manager.device)

    if starting:
        module = TriangleAttentionStartingNode1D(module_serial, device_mesh, cp_group)
    else:
        module = TriangleAttentionEndingNode1D(module_serial, device_mesh, cp_group)
    module = module.train()

    def _make_input_dtensors():
        x_dt = distribute_tensor(
            input_x_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
        ).requires_grad_(True)
        mask_dt = distribute_tensor(
            mask_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
        )
        d_out_dt = distribute_tensor(
            d_output_global_host.to(dtype=dtype, device=manager.device),
            device_mesh=device_mesh,
            placements=placements_z,
        )
        return x_dt, mask_dt, d_out_dt

    # --- REFERENCE pass (TF32 disabled per Precision.FP32) ---
    x_ref, mask_ref, d_out_ref = _make_input_dtensors()
    with setup_tf32_env(Precision.FP32):
        out_ref = module(x_ref, mask_ref, triattn_backend=TriAttnBackend.REFERENCE)
        out_ref.backward(d_out_ref)
    ref_param_grads = {name: param.grad.full_tensor().detach().clone() for name, param in module.named_parameters()}
    # Zero param grads before the CUEQ pass so they don't accumulate.
    for param in module.parameters():
        if param.grad is not None:
            param.grad = None

    # --- CUEQ pass (TF32 enabled per Precision.TF32; required for CUEQ bwd) ---
    x_cueq, mask_cueq, d_out_cueq = _make_input_dtensors()
    with setup_tf32_env(Precision.TF32):
        out_cueq = module(x_cueq, mask_cueq, triattn_backend=TriAttnBackend.CUEQ)
        out_cueq.backward(d_out_cueq)
    cueq_param_grads = {name: param.grad.full_tensor().detach().clone() for name, param in module.named_parameters()}

    # --- Cross-kernel comparison ---
    # Forward output: full-tensor reassembly for a global comparison.
    out_ref_global = out_ref.full_tensor().cpu()
    out_cueq_global = out_cueq.full_tensor().cpu()
    assert (
        out_ref_global.shape == out_cueq_global.shape
    ), f"Rank {rank}: output shape mismatch REF={out_ref_global.shape} CUEQ={out_cueq_global.shape}"
    assert out_ref_global.isfinite().all(), f"Rank {rank}: REFERENCE output has non-finite values"
    assert out_cueq_global.isfinite().all(), f"Rank {rank}: CUEQ output has non-finite values"
    assert out_ref_global.abs().max() > 0, f"Rank {rank}: REFERENCE output is all zeros (vacuous test)"
    assert out_cueq_global.abs().max() > 0, f"Rank {rank}: CUEQ output is all zeros (vacuous test)"

    torch.testing.assert_close(
        out_cueq_global,
        out_ref_global,
        atol=fwd_atol,
        rtol=fwd_rtol,
    )

    # Input gradient parity.
    grad_ref_global = x_ref.grad.full_tensor().cpu()
    grad_cueq_global = x_cueq.grad.full_tensor().cpu()
    assert grad_ref_global.isfinite().all(), f"Rank {rank}: REFERENCE input grad has non-finite values"
    assert grad_cueq_global.isfinite().all(), f"Rank {rank}: CUEQ input grad has non-finite values"
    assert grad_ref_global.abs().max() > 0, f"Rank {rank}: REFERENCE input grad is all zeros (vacuous test)"
    assert grad_cueq_global.abs().max() > 0, f"Rank {rank}: CUEQ input grad is all zeros (vacuous test)"

    torch.testing.assert_close(
        grad_cueq_global,
        grad_ref_global,
        atol=bwd_atol_input,
        rtol=bwd_rtol_input,
    )

    # Parameter-gradient parity.  Replicated params have Partial(Sum) grads
    # that full_tensor() reduces; compare globally.
    for name in ref_param_grads:
        assert name in cueq_param_grads, f"Rank {rank}: param {name} missing from CUEQ grads"
        ref_g = ref_param_grads[name].cpu()
        cueq_g = cueq_param_grads[name].cpu()
        assert ref_g.isfinite().all(), f"Rank {rank}: REFERENCE grad {name} has non-finite values"
        assert cueq_g.isfinite().all(), f"Rank {rank}: CUEQ grad {name} has non-finite values"
        torch.testing.assert_close(
            cueq_g,
            ref_g,
            atol=bwd_atol_param,
            rtol=bwd_rtol_param,
        )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 1), True, "cuda", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=["dp:1-cp:1-cuda", "dp:1-cp:2-cuda"],
)
@pytest.mark.parametrize("starting", [True, False], ids=["starting", "ending"])
def test_triangle_attention_cueq_vs_reference_parity_1d(setup_env, starting):
    """Cross-kernel CUEQ-vs-REFERENCE parity on the DAP 1D path.

    Same module, same inputs, both backends — compares forward output and
    input/parameter gradients within a TF32-kernel-difference budget.  Closes
    the coverage gap where `parallel_assert_triangle_attention_1d` skips
    cross-kernel numerical parity for CUEQ (todo.md item, refs 66dd5fd3).

    Parametrized over `starting` (True=all-gather DAP, False=col-slab DAP) and
    cp ∈ {1, 2}.  CUEQ requires CUDA + cuequivariance_torch.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type != "cuda":
        pytest.skip("CUEQ requires CUDA")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not cueq_is_installed:
        pytest.skip("cuequivariance_torch not installed")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]
    # Small dims for fast cross-kernel parity.  N=cp*4 mirrors the existing
    # parallel_assert_triangle_attention_1d small-model config.
    N = cp_size * 4
    c_in = 16
    c_hidden = 4
    no_heads = 2

    seed = 42
    seed_by_rank(0, seed=seed)

    g = torch.Generator(device="cpu").manual_seed(seed)
    input_x = torch.empty((B, N, N, c_in), dtype=torch.float32).uniform_(-0.5, 0.5, generator=g)
    # All-ones mask: REFERENCE uses inf*(mask-1) = 0 (no masking),
    # CUEQ uses mask.bool() = all-True.  Both paths produce the same masking
    # semantics with no large negative bias overflow risk -- isolates the
    # kernel-math difference (TF32 vs FP32) from mask-handling differences.
    mask = torch.ones((B, N, N), dtype=torch.float32)
    d_output = torch.empty((B, N, N, c_in), dtype=torch.float32).uniform_(-1.0, 1.0, generator=g)

    if starting:
        reference_module = TriangleAttentionStartingNode(c_in, c_hidden, no_heads)
    else:
        reference_module = TriangleAttentionEndingNode(c_in, c_hidden, no_heads)
    init_module_params_uniform(reference_module, low=-0.5, high=0.5)
    layer_state_dict = reference_module.state_dict()

    spawn_multiprocessing(
        parallel_assert_triangle_attention_cueq_vs_reference_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        c_in,
        c_hidden,
        no_heads,
        starting,
        layer_state_dict,
        input_x,
        mask,
        d_output,
    )


# ---------------------------------------------------------------------------
# Single-rank gloo helpers: adversarial dtype/device asserts (Item #2)
# and saved_tensors gating introspection (Item #3).  Both use cp=1 gloo so
# the cp_size==1 fast-path makes every collective a no-op and the tests
# run on the 2-GPU desktop without NCCL.
# ---------------------------------------------------------------------------


def _init_single_rank_gloo(monkeypatch):
    """Set up a single-rank gloo PG on a free port and return ``cp_group``.

    Mirrors the setup in ``test_dap_endingnode_1d_single_rank_sanity``.
    Returns ``cp_group`` (also installs the ``dist.get_backend`` monkeypatch
    so the ending-node NCCL guard accepts the gloo PG).  Caller is responsible
    for ``dist.destroy_process_group()`` in a ``finally``.

    Note: the ``dist.get_backend`` monkeypatch is no longer strictly required
    for cp_size==1 callers — the end-node DAP NCCL guard at
    ``_DAPTriangleAttentionEndingNode1DImpl.forward`` now gates on
    ``cp_size > 1`` and is a no-op for the single-rank degenerate path.  Kept
    here for explicitness and to keep this helper's contract identical across
    cp_size==1 and any hypothetical future cp_size>1 gloo caller.
    """
    import os
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    cp_group = dist.group.WORLD

    real_get_backend = dist.get_backend

    def _fake_get_backend(group=None):
        if group is cp_group:
            return "nccl"
        return real_get_backend(group)

    monkeypatch.setattr(dist, "get_backend", _fake_get_backend)
    return cp_group


def _starting_node_apply(cp_group, q, k, v, mask_bias, triangle_bias):
    """Call ``_AllGatherTriangleAttentionStartingNode1DImpl.apply`` at cp=1."""
    return _AllGatherTriangleAttentionStartingNode1DImpl.apply(
        q,
        k,
        v,
        mask_bias,
        triangle_bias,
        cp_group,
        TriAttnBackend.REFERENCE,
        True,  # apply_scale (q pre-scaled by caller)
        1.0,  # q_scale unused when apply_scale=True
    )


def _ending_node_apply(cp_group, q, k, v, mask_bias, triangle_bias, no_heads, c_hidden):
    """Call ``_DAPTriangleAttentionEndingNode1DImpl.apply`` at cp=1."""
    return _DAPTriangleAttentionEndingNode1DImpl.apply(
        q,
        k,
        v,
        mask_bias,
        triangle_bias,
        cp_group,
        no_heads,
        c_hidden,
        TriAttnBackend.REFERENCE,
        True,
        1.0,
    )


def _make_qkv_starting(dtype=torch.float32, device="cpu"):
    """Row-slab inputs for the starting-node impl at cp=1.

    Shapes (cp=1):
        q,k,v       [B, I=N, H, J=N, D]   (I/cp == N when cp=1)
        mask_bias   [B, I=N, 1, 1, J=N]
        triangle_b  [B, 1, H, I=N, J=N]
    """
    B, H, N, D = 1, 2, 4, 2
    gen = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, N, H, N, D, generator=gen, dtype=dtype).to(device=device)
    k = torch.randn(B, N, H, N, D, generator=gen, dtype=dtype).to(device=device)
    v = torch.randn(B, N, H, N, D, generator=gen, dtype=dtype).to(device=device)
    mask_bias = torch.zeros(B, N, 1, 1, N, dtype=dtype, device=device)
    triangle_bias = torch.randn(B, 1, H, N, N, generator=gen, dtype=dtype).to(device=device)
    return q, k, v, mask_bias, triangle_bias, (B, H, N, D)


def _make_qkv_ending(dtype=torch.float32, device="cpu"):
    """Col-slab (z-first) inputs for the ending-node impl at cp=1.

    The z-first end-node fn consumes col-slab tensors (q/k/v projected on z_col)
    and the bias PROJECTION (not the gathered operand).  Shapes (cp=1, J/cp == N):
        q,k,v         [B, J/cp=N, H, I=N, D]
        mask_bias     [B, J/cp=N, 1, 1, I_k=N]
        bias_col_proj [B, I=row=N, J/cp=col=N, H]   (the projection; the fn
                       all-gathers its col axis -> full [B,1,H,I_q=N,I_k=N])

    Note: D=4 (not 2) so the CUEQ fp32 kernel's hidden_dim % 4 == 0 and <= 32
    contract is satisfied for the saved_tensors gating test.
    """
    B, H, N, D = 1, 2, 4, 4
    gen = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, N, H, N, D, generator=gen, dtype=dtype).to(device=device)
    k = torch.randn(B, N, H, N, D, generator=gen, dtype=dtype).to(device=device)
    v = torch.randn(B, N, H, N, D, generator=gen, dtype=dtype).to(device=device)
    mask_bias = torch.zeros(B, N, 1, 1, N, dtype=dtype, device=device)
    # z-first bias PROJECTION layout [B, I=row=N, J/cp=col=N, H] (NOT the gathered
    # [B, 1, H, N, N] operand -- the fn builds that via _build_bias_full_from_proj).
    bias_col_proj = torch.randn(B, N, N, H, generator=gen, dtype=dtype).to(device=device)
    return q, k, v, mask_bias, bias_col_proj, (B, H, N, D)


@pytest.mark.parametrize("impl", ["starting", "ending"])
@pytest.mark.parametrize("mismatch", ["dtype", "device"])
def test_triattn_1d_qkv_dtype_device_assert(monkeypatch, impl, mismatch):
    """Adversarial test: q/k/v with a dtype OR device mismatch must trip
    the new asserts in both DAP forwards (Item #2 vacuous-pass guard).

    Uses a single-rank gloo PG so cp_size==1 makes every collective a
    no-op fast-path; the asserts fire at the very top of forward before
    any collective is issued.
    """
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized; not safe to clobber")

    if mismatch == "device" and not torch.cuda.is_available():
        pytest.skip("device-mismatch adversarial requires CUDA for the other-device side")

    cp_group = _init_single_rank_gloo(monkeypatch)
    try:
        if impl == "starting":
            q, k, v, mask_bias, tb, shape = _make_qkv_starting()
        else:
            q, k, v, mask_bias, tb, shape = _make_qkv_ending()
        _, H, _, D = shape

        if mismatch == "dtype":
            # bf16 q, fp32 k -- valid shape on both, mismatched dtype.
            q = q.to(dtype=torch.bfloat16)
            k = k.to(dtype=torch.float32)
            expected_substr = "dtype mismatch"
        else:
            # CPU q, CUDA k -- valid shape on both, mismatched device.
            q = q.cpu()
            k = k.cuda()
            v = v.cuda()
            mask_bias = mask_bias.cuda()
            tb = tb.cuda()
            expected_substr = "device mismatch"

        with pytest.raises(AssertionError, match=expected_substr):
            if impl == "starting":
                _starting_node_apply(cp_group, q, k, v, mask_bias, tb)
            else:
                _ending_node_apply(cp_group, q, k, v, mask_bias, tb, H, D)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "triattn_backend",
    [TriAttnBackend.REFERENCE, TriAttnBackend.CUEQ],
    ids=lambda x: x.value,
)
def test_dap_endingnode_1d_o_col_save_gating(monkeypatch, triattn_backend):
    """Algorithmic-claim guard for Item #3: assert that the last saved
    tensor in ``_DAPTriangleAttentionEndingNode1DImpl`` is ``o_col`` on
    REFERENCE and ``None`` on CUEQ.

    Mirrors the saved-tensors introspection pattern from
    ``test_all_gather_starting_node_1d_gloo_and_memory_budget``.

    CUEQ branch requires CUDA + cuequivariance_torch; skipped otherwise.
    At cp_size=1 every collective in the impl is a no-op fast-path, so
    the test runs on the 2-GPU desktop without NCCL.
    """
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized; not safe to clobber")

    if triattn_backend == TriAttnBackend.CUEQ:
        if not torch.cuda.is_available():
            pytest.skip("CUEQ requires CUDA")
        if not cueq_is_installed:
            pytest.skip("cuequivariance_torch not installed")
        device = "cuda"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float64

    cp_group = _init_single_rank_gloo(monkeypatch)
    try:
        q, k, v, mask_bias, tb, (B, H, N, D) = _make_qkv_ending(dtype=dtype, device=device)
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)
        tb = tb.requires_grad_(True)

        # CUEQ requires TF32 backward; enable so backward graph is consistent
        # with production.  REFERENCE works under either precision; pin FP32.
        precision = Precision.TF32 if triattn_backend == TriAttnBackend.CUEQ else Precision.FP32
        with setup_tf32_env(precision):
            output = _DAPTriangleAttentionEndingNode1DImpl.apply(
                q,
                k,
                v,
                mask_bias,
                tb,
                cp_group,
                H,
                D,
                triattn_backend,
                True,
                1.0,
            )

        saved = output.grad_fn.saved_tensors
        # z-first forward saves q_col, k_col, v_col, mask_col, bias_col_proj, o_col.
        assert (
            len(saved) == 6
        ), f"Expected 6 saved tensors (q/k/v/mask/bias_col_proj/o_col), got {len(saved)} for backend={triattn_backend}"
        # saved[4] is the O(N^2/cp) col-slab bias PROJECTION [B, I=N, J/cp, H],
        # NOT the gathered O(N^2) operand (budget guard).
        saved_bias_proj = saved[4]
        assert saved_bias_proj.shape == (B, N, N, H), (
            f"saved bias_col_proj shape {saved_bias_proj.shape} != expected [B,I=N,J/cp=N,H] "
            f"{(B, N, N, H)} (must be the projection, not the gathered [B,1,H,N,N] operand)"
        )
        saved_o_col = saved[-1]

        if triattn_backend == TriAttnBackend.REFERENCE:
            assert saved_o_col is not None, "REFERENCE: o_col must be saved (consumed by backward d = vecdot(do, o_c))"
            expected_shape = (B, N, H, N, D)  # [B, J/cp=N, H, I=N, D] at cp=1
            assert (
                saved_o_col.shape == expected_shape
            ), f"REFERENCE: saved o_col shape {saved_o_col.shape} != expected {expected_shape}"
        else:
            assert saved_o_col is None, (
                f"CUEQ: o_col must be None (CUEQ backward recomputes o via forward kernel), "
                f"got {type(saved_o_col).__name__}"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_ending_node_bias_index_semantics_perturbation_trace():
    """ANCHOR: pin the ending-node triangle-bias index semantics (serial, CPU).

    This is the durable ground-truth anchor for the z-first ending-node fusion's
    bias-handling design (sichu/triattn-fuse).  The whole design hinges on ONE
    fact that is genuinely error-prone to derive by reading shapes (the column
    axis J appears in BOTH the attention batch position AND, positionally, the
    query position) -- so we pin it by an EXECUTABLE PERTURBATION TRACE rather
    than prose (see project-memory ``settle-sharded-semantics-by-trace``).

    Claim under test: the ending-node triangle bias is ``b[i_q, i_k]`` where the
    attention QUERY axis ``i_q`` == the original COLUMN axis ``J``, and the KEY
    axis ``i_k`` == the original ROW axis ``I``.  The bias's dim-1 singleton
    broadcasts the SAME ``[i_q=N, i_k=N]`` face over every batch column -- so the
    bias is a full O(N^2) object that z-first must ALL-GATHER on the column axis
    (which it shards), NOT a per-column-shardable O(N^2/cp) tensor.
    """
    import torch

    from boltz.model.layers.triangular_attention.attention import TriangleAttentionEndingNode
    from boltz.model.layers.triangular_attention.utils import permute_final_dims

    torch.manual_seed(0)
    B, N, C, H, Dh = 1, 6, 8, 2, 4
    module = TriangleAttentionEndingNode(C, Dh, H).to(dtype=torch.float64).eval()

    x = torch.randn(B, N, N, C, dtype=torch.float64)

    # Reproduce the serial bias construction EXACTLY (attention.py forward):
    #   ending node transposes I<->J, LN, then bias = permute_final_dims(linear(x),(2,0,1)).
    x_t = x.transpose(-2, -3)  # [B, J, I, C]
    x_n = module.layer_norm(x_t)
    # [B, J, I, H] -> permute_final_dims((2,0,1)) -> [B, H, J, I] -> unsqueeze -> [B, 1, H, J, I]
    bias = permute_final_dims(module.linear(x_n), (2, 0, 1)).unsqueeze(-4)  # [B, 1, H, J, I]

    # The score a (in _attention) is [B, J_batch, H, I_q, I_k]; a += bias broadcasts
    # bias's dim-1 singleton over J_batch, aligning bias dim3 (=J) with I_q and
    # bias dim4 (=I) with I_k.  Perturb a single (dim3=i_q0, dim4=i_k0) bias element.
    iq0, ik0, h0 = 3, 1, 1
    bias_perturbed = bias.clone()
    bias_perturbed[0, 0, h0, iq0, ik0] += 1.0
    delta = (bias_perturbed - bias)[0]  # [1, H, J=N, I=N]

    # (1) Exactly one bias element differs, at (h0, dim3=iq0, dim4=ik0).
    nz = torch.nonzero(delta, as_tuple=False)
    assert nz.shape[0] == 1, f"expected 1 perturbed bias element, got {nz.shape[0]}"
    assert tuple(nz[0].tolist()) == (0, h0, iq0, ik0), f"perturbation at wrong index {nz[0].tolist()}"

    # (2) The bias add to score[b, jb, h, i_q, i_k] uses bias[b,0,h,i_q,i_k] for ALL
    #     batch columns jb -> the SAME [i_q,i_k] face is added to every column, so
    #     the bias is O(N^2) (full i_q x i_k), NOT shardable by the column batch.
    score = torch.zeros(B, N, H, N, N, dtype=torch.float64)  # [B, J_batch, H, I_q, I_k]
    added = score + bias  # broadcast bias [B,1,H,N,N] over J_batch
    for jb in range(1, N):
        torch.testing.assert_close(added[:, jb], added[:, 0])
    # the perturbed element lands at score[:, every_jb, h0, i_q=iq0, i_k=ik0]:
    added_p = score + bias_perturbed
    diff = (added_p - added)[0]  # [J_batch, H, I_q, I_k]
    nz_score = torch.nonzero(diff.abs() > 0, as_tuple=False)
    assert nz_score.shape[0] == N, f"expected {N} nonzeros (one per batch col), got {nz_score.shape[0]}"
    for row in nz_score:
        assert tuple(row[1:].tolist()) == (h0, iq0, ik0), f"score sensitivity at wrong (h,i_q,i_k): {row.tolist()}"
