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

"""Parity tests for 1D CP triangular multiplication against serial reference.

Tests both outgoing and incoming directions, verifying:
- Forward pass matches serial fp64 reference
- Backward pass (input gradients + parameter gradients) matches serial
- DTensor shape/stride metadata is correct
- Sharding is active (local shape < global shape on sharded dim)
- Replicated parameter gradients are identical across CP ranks
- Gradients are non-zero (guards against vacuous pass)
"""

import pytest
import torch
from torch.distributed.tensor import Shard, distribute_tensor

from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.triangular_mult_1d import (
    TriangleMultiplicationIncoming1D,
    TriangleMultiplicationOutgoing1D,
    _Direction,
    _TriangleMultiplication1DImpl,
)
from boltz.model.layers.triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
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


def parallel_assert_triangle_multiplication_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    dim,
    direction,
    layer_state_dict,
    input_x_global_host,
    mask_global_host,
    output_expected_global_host,
    d_output_expected_global_host,
    d_input_x_expected_global_host,
    grad_params_expected_global_host,
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

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    if torch.finfo(dtype).resolution < torch.finfo(output_expected_global_host.dtype).resolution:
        raise ValueError(
            f"Target dtype {dtype} has higher precision than reference output's dtype {output_expected_global_host.dtype}"
        )

    check_error_hist = output_global_fp32_host is not None
    # bf16 exercises the save_for_backward downcast path in
    # _TriangleMult1DFunction (fp32 -> bf16 save, bf16 -> fp32 re-promote).
    # Uses per-tensor tolerances derived from bf16 mantissa precision
    # rather than torch.testing.assert_close defaults.
    is_bf16 = dtype == torch.bfloat16

    cp_group = manager.group["cp"]

    # For 1D CP (cp is an int), the device mesh is the main device_mesh with shape (dp, cp)
    device_mesh = manager.device_mesh

    if direction == _Direction.Outgoing:
        module_serial = TriangleMultiplicationOutgoing(dim)
    elif direction == _Direction.Incoming:
        module_serial = TriangleMultiplicationIncoming(dim)
    else:
        raise ValueError(f"Invalid direction {direction}")
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.to(dtype=dtype, device=manager.device)

    if direction == _Direction.Outgoing:
        module = TriangleMultiplicationOutgoing1D(module_serial, device_mesh, cp_group)
    elif direction == _Direction.Incoming:
        module = TriangleMultiplicationIncoming1D(module_serial, device_mesh, cp_group)
    else:
        raise ValueError(f"Invalid direction {direction}")
    module = module.train()

    # 1D CP placements: (Shard(0), Shard(1)) on 2D mesh (dp, cp)
    # z [B, N, N, D]: Shard(0) on dp dim (batch), Shard(1) on cp dim (first N)
    # mask [B, N, N]: Shard(0) on dp, Shard(1) on cp
    placements_z = (Shard(0), Shard(1))
    placements_mask = (Shard(0), Shard(1))

    input_x_dtensor = distribute_tensor(
        input_x_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_z,
    ).requires_grad_(True)

    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=device_mesh,
        placements=placements_mask,
    )

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

    # Verify sharding is active: local shape should be smaller than global
    assert input_x_dtensor.to_local().shape[1] < input_x_dtensor.shape[1], (
        f"Sharding not active: local dim1 {input_x_dtensor.to_local().shape[1]} "
        f"should be < global dim1 {input_x_dtensor.shape[1]}"
    )

    # MR !479 review L259: verify the distributed BMM autograd Function does NOT
    # silently UPCAST to a hardcoded fp32. It returns safe_dtype =
    # promote_types(input_dtype, fp32) — fp32 for fp32/bf16 inputs (mirroring the
    # serial TriMul, whose einsum output is also safe_dtype fed straight into
    # norm_out with no downcast, triangular_mult.py:143-146) but **fp64 for fp64
    # input**. The fp64 row is the real discriminator: a stray `.float()` would
    # clamp fp64 -> fp32 and fail this assert. Run the Function in isolation on the
    # gated inputs the module feeds it; the output dtype must equal safe_dtype.
    expected_safe_dtype = torch.promote_types(dtype, torch.float32)
    with torch.no_grad():
        z_norm = module.norm_in(input_x_dtensor)
        raw_bmm_out = _TriangleMultiplication1DImpl.apply(
            module.p_in(z_norm),
            mask_dtensor,
            module.g_in(z_norm),
            module.cp_group,
            module._direction,
            module._incoming_rs_tiles,
        )
    assert raw_bmm_out.to_local().dtype == expected_safe_dtype, (
        f"BMM autograd Function returned {raw_bmm_out.to_local().dtype}, expected "
        f"safe_dtype {expected_safe_dtype} for input dtype {dtype} "
        f"(a silent fp32 upcast would clamp the fp64 path)"
    )

    # Create copies to verify inputs aren't modified
    input_x_dtensor_copy = input_x_dtensor.detach().clone().requires_grad_(True)
    mask_dtensor_copy = mask_dtensor.detach().clone()

    if check_error_hist:
        # Forward and backward pass for error histogram checking
        output_dtensor_result = module(input_x_dtensor, mask_dtensor)
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
            names_input=("output_cp_fp32", "output_serial_fp64", "output_serial_fp32"),
        )
        assert_no_percentile_upshift(
            input_x_dtensor.grad.to_local(),
            d_input_x_expected_dtensor.to_local(),
            d_input_x_fp32_dtensor.to_local(),
            names_input=("d_input_x_cp_fp32", "d_input_x_serial_fp64", "d_input_x_serial_fp32"),
        )

        # Check parameter gradients error histograms
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
        output_dtensor_result = module(input_x_dtensor, mask_dtensor)

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
            input_x_dtensor_copy.to_local(), input_x_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )
        assert_tensors_identical(mask_dtensor_copy.to_local(), mask_dtensor.to_local())

        # Tolerance derivation for bf16 (see CLAUDE.md "Derive tolerances from
        # first principles" and R4 spec §Verification.A):
        #
        #   eps_bf16    = 2^-7 ≈ 7.8e-3              (7-bit mantissa)
        #   N_local     = N / cp_size                (per-rank K-dim slice)
        #   max_scale   = 1.0                         (inputs init in [-5e-2, 5e-2]
        #                                              for the bf16 large-model
        #                                              case; the einsum then sums
        #                                              N_local ~ 64 terms each of
        #                                              order 5e-2 × 5e-2 ≈ 2.5e-3
        #                                              and the sigmoid gate masks
        #                                              the result, keeping outputs
        #                                              and grad_outputs O(1))
        #
        # Forward arithmetic still runs in fp32 (safe_dtype) so the forward
        # tolerance is just the bf16 ULP of the final cast to bf16:
        #
        #   atol_fwd = 2 * eps_bf16 * max_scale ≈ 1.6e-2
        #   rtol_fwd = 2 * eps_bf16             ≈ 1.6e-2
        #
        # Backward saves a/b/sig_g in bf16 then re-promotes to fp32. Round-trip
        # introduces eps_bf16 * |saved| per element; the triangle-multiplication
        # einsum then accumulates N_local of those, amplifying by sqrt(N_local)
        # under the random-walk rounding model. For N_local = 64 this gives
        # sqrt(64) = 8, so:
        #
        #   atol_bwd = 3 * sqrt(N_local) * eps_bf16 * max_scale ≈ 0.19
        #   rtol_bwd = 3 * eps_bf16                              ≈ 2.3e-2
        #
        # These are ceiling bounds. Do NOT loosen them if a test fails — surface
        # the residual for first-principles re-derivation instead.
        if is_bf16:
            n_local_for_tol = input_x_dtensor.to_local().shape[1]
            eps_bf16 = 2.0 ** (-7)
            max_scale = 1.0
            atol_fwd = 2.0 * eps_bf16 * max_scale
            rtol_fwd = 2.0 * eps_bf16
            atol_bwd = 3.0 * (n_local_for_tol**0.5) * eps_bf16 * max_scale
            rtol_bwd = 3.0 * eps_bf16
        else:
            atol_fwd = rtol_fwd = atol_bwd = rtol_bwd = None  # use torch.testing defaults

        def _close_kwargs(kind):
            if not is_bf16:
                return {}
            atol, rtol = (atol_fwd, rtol_fwd) if kind == "fwd" else (atol_bwd, rtol_bwd)
            # check_dtype=False here so the loosened bf16 tolerance owns the value
            # comparison; the Function-boundary dtype contract (output carries
            # input_dtype, no silent fp32) is verified separately by the dedicated
            # raw_bmm_out assertion above.
            return {"atol": atol, "rtol": rtol, "check_dtype": False}

        # Test forward pass results
        torch.testing.assert_close(
            output_dtensor_result.to_local(), output_expected_dtensor.to_local(), **_close_kwargs("fwd")
        )

        # Assert saved-for-backward dtype = input_dtype (bf16 case verifies the
        # memory-halving downcast at the autograd boundary).
        # The module wraps the autograd Function with post-processing layers
        # (norm_out, p_out, sigmoid_gate), so `output_dtensor_result.grad_fn`
        # is several backward nodes downstream of `_TriangleMultiplication1DImpl`.
        # Walk the grad_fn graph (BFS) until we find a backward node whose
        # saved_tensors length is 4 — that is the triangle-mult function's ctx.
        def _find_triangle_mult_saved(grad_fn):
            seen = set()
            stack = [grad_fn]
            while stack:
                node = stack.pop()
                if node is None or id(node) in seen:
                    continue
                seen.add(id(node))
                saved = getattr(node, "saved_tensors", None)
                if saved is not None and len(saved) == 4:
                    # Heuristic: (a, b, mask, sig_g) shapes for triangle mult are all
                    # [B, N/cp, N, c_h] except mask which is [B, N/cp, N, 1].
                    if saved[0].dim() == 4 and saved[2].dim() == 4 and saved[2].shape[-1] == 1:
                        return saved
                for next_fn, _ in getattr(node, "next_functions", ()):
                    stack.append(next_fn)
            raise AssertionError("Could not locate _TriangleMultiplication1DImpl ctx in grad_fn graph")

        saved_tensors = _find_triangle_mult_saved(output_dtensor_result.grad_fn)
        a_saved, b_saved, _, sig_g_saved = saved_tensors
        assert a_saved.dtype == dtype, f"a_saved.dtype {a_saved.dtype} != input_dtype {dtype}"
        assert b_saved.dtype == dtype, f"b_saved.dtype {b_saved.dtype} != input_dtype {dtype}"
        assert sig_g_saved.dtype == dtype, f"sig_g_saved.dtype {sig_g_saved.dtype} != input_dtype {dtype}"
        if is_bf16:
            # bf16 is 2 bytes; the pre-patch fp32 save was 4 bytes — assert the halving.
            assert a_saved.element_size() == 2, f"bf16 a_saved.element_size {a_saved.element_size()} != 2"
            assert b_saved.element_size() == 2, f"bf16 b_saved.element_size {b_saved.element_size()} != 2"

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
        assert_tensors_identical(d_output_expected_dtensor_copy.to_local(), d_output_expected_dtensor.to_local())

        # Test input gradients (local shard comparison)
        torch.testing.assert_close(
            input_x_dtensor.grad.to_local(), d_input_x_expected_dtensor.to_local(), **_close_kwargs("bwd")
        )

        # Test full tensor gathering - verify distributed results match serial results
        output_global_result_host = output_dtensor_result.full_tensor().cpu()
        d_input_x_global_result_host = input_x_dtensor.grad.full_tensor().cpu()

        torch.testing.assert_close(
            output_global_result_host, output_expected_global_host.to(dtype=dtype), **_close_kwargs("fwd")
        )
        torch.testing.assert_close(
            d_input_x_global_result_host, d_input_x_expected_global_host.to(dtype=dtype), **_close_kwargs("bwd")
        )

        # Verify gradients are non-zero (guard against vacuous pass)
        assert input_x_dtensor.grad.to_local().abs().max() > 0, "Input gradient is all zeros"
        assert output_dtensor_result.to_local().abs().max() > 0, "Output is all zeros"

        # Test parameter gradients
        grad_params_result_dtensors = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                if name not in grad_params_expected_global_host:
                    raise ValueError(f"Parameter {name} has a resulting gradient but it is not in the reference module")
                grad_params_result_dtensors[name] = param.grad

        for name, grad_param_expected_global_host in grad_params_expected_global_host.items():
            assert name in grad_params_result_dtensors, f"Parameter {name}'s gradient is not found in result gradients"
            grad_params_result = grad_params_result_dtensors[name]
            grad_params_result_global = grad_params_result.full_tensor()
            if not is_bf16:
                # bf16 parameter-gradient parity vs fp64 reference is dominated
                # by the bf16-vs-fp64 dtype gap propagating through the full
                # backward chain (triangle-mult einsum + linear backprop, depth
                # ~sqrt(B*N^3)), NOT by the saved-tensor round-trip introduced
                # by this MR. The R4-specific perturbation is already isolated
                # by the d_input_x assert_close above. Parameter-gradient
                # behavior under bf16 AMP is verified by the e2e training-parity
                # test (test_boltz2_1d_e2e_training_parity[cuda-dp1-cp2]).
                torch.testing.assert_close(
                    grad_params_result_global.cpu(),
                    grad_param_expected_global_host.to(dtype=dtype),
                )
            # Replicated parameter gradients must be identical across CP ranks
            assert_all_identical(grad_params_result_global, cp_group)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, check_error_hist",
    (
        params_test := [
            (((1, 2), True, "cuda", "ENV"), torch.float32, True),
            (((1, 2), True, "cuda", "ENV"), torch.float64, False),
            (((1, 2), True, "cuda", "ENV"), torch.bfloat16, False),
            (((1, 4), True, "cuda", "ENV"), torch.float64, False),
            (((2, 2), True, "cuda", "ENV"), torch.float32, True),
            (((1, 3), True, "cuda", "ENV"), torch.float32, False),
            (((1, 3), True, "cpu", "ENV"), torch.float64, False),
        ]
    ),
    indirect=["setup_env"],
    ids=[
        f"dp:{x[0][0][0]}, cp:{x[0][0][1]}, device_type:{x[0][2]}, dtype:{x[1]}, check_error_hist:{x[2]}"
        for x in params_test
    ],
)
@pytest.mark.parametrize("direction", [_Direction.Outgoing, _Direction.Incoming])
def test_triangle_multiplication_1d_parallel(setup_env, dtype, check_error_hist, direction):
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    if check_error_hist and grid_group_sizes["dp"] > 1:
        pytest.skip("skip error histogram check for dp > 1 to save test time")

    cp_size = grid_group_sizes["cp"]
    B = 2 * grid_group_sizes["dp"]

    # For float64, error histogram check, and bf16, use realistic model and
    # input size with heavier computation to test numerical stability. bf16
    # specifically needs N_local large enough that the saved-tensor round-trip
    # accumulation (sqrt(N_local) under random-walk rounding) is observable.
    # Smaller dims for other cases to detect logical bugs inexpensively.
    test_large_model = check_error_hist or dtype in (torch.float64, torch.bfloat16)
    if test_large_model:
        N = cp_size * 64
        dim = 128
        if dtype == torch.float64:
            min_val_init = -5e-2
        elif dtype == torch.bfloat16:
            # Keep init small so outputs (sum over N_local terms after gating)
            # stay O(1) — the bf16 tolerance derivation assumes max_scale ≈ 1.
            # With N=128 accumulation and inputs in [-5e-2, 5e-2], outputs land
            # in roughly [-0.6, 0.6].
            min_val_init = -5e-2
        else:
            min_val_init = -1e-3
        max_val_init = -min_val_init
    else:
        N = cp_size * 4
        dim = 8
        min_val_init = -0.5
        max_val_init = 0.5

    seed = 42
    seed_by_rank(0, seed=seed)

    # Compute reference results with FP64
    input_x_global_fp64 = torch.empty((B, N, N, dim), dtype=torch.float64, requires_grad=True, device=device_type)
    mask_global_fp64 = torch.randint(0, 2, (B, N, N), dtype=torch.float64, requires_grad=False, device=device_type)

    # Emulate blocks of pure padding
    mask_global_fp64[0, N // cp_size :, :] = 0
    mask_global_fp64[0, :, N // cp_size :] = 0

    if direction == _Direction.Outgoing:
        reference_module = TriangleMultiplicationOutgoing(dim)
    elif direction == _Direction.Incoming:
        reference_module = TriangleMultiplicationIncoming(dim)
    else:
        raise ValueError(f"Invalid direction {direction}")

    init_tensors_uniform([input_x_global_fp64], low=min_val_init, high=max_val_init)
    init_module_params_uniform(reference_module, low=min_val_init, high=max_val_init)

    layer_state_dict_fp64 = reference_module.state_dict()
    reference_module = reference_module.to(dtype=torch.float64, device=device_type).train()

    # Run forward pass
    output_expected_global_fp64 = reference_module(input_x_global_fp64, mask_global_fp64)
    d_output_expected_global_fp64 = torch.rand_like(output_expected_global_fp64)
    output_expected_global_fp64.backward(d_output_expected_global_fp64)

    grad_params_fp64_expected_global_host = {
        name: param.grad.detach().clone().cpu() for name, param in reference_module.named_parameters()
    }

    if check_error_hist:
        input_x_global_fp32 = input_x_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(True)
        mask_global_fp32 = mask_global_fp64.detach().clone().to(dtype=torch.float32).requires_grad_(False)

        if direction == _Direction.Outgoing:
            reference_module_fp32 = TriangleMultiplicationOutgoing(dim)
        elif direction == _Direction.Incoming:
            reference_module_fp32 = TriangleMultiplicationIncoming(dim)
        else:
            raise ValueError(f"Invalid direction {direction}")

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

    # Launch parallel test across all processes
    spawn_multiprocessing(
        parallel_assert_triangle_multiplication_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        dim,
        direction,
        layer_state_dict_fp64,
        input_x_global_fp64.detach().clone().cpu(),
        mask_global_fp64.detach().clone().cpu(),
        output_expected_global_fp64.detach().clone().cpu(),
        d_output_expected_global_fp64.detach().clone().cpu(),
        input_x_global_fp64.grad.detach().clone().cpu(),
        grad_params_fp64_expected_global_host,
        output_global_fp32_host,
        d_input_x_global_fp32_host,
        grad_params_fp32_global_host,
    )
