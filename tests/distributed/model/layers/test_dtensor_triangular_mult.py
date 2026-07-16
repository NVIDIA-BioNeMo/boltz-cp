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


import pytest
import torch
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from boltz.distributed.comm import Ring2DComm
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming as DistributedTriangleMultiplicationIncoming,
)
from boltz.distributed.model.layers.triangular_mult import (
    TriangleMultiplicationOutgoing as DistributedTriangleMultiplicationOutgoing,
)
from boltz.distributed.model.layers.triangular_mult import (
    _Direction,
    _distributed_bmm,
    _TriangleMultiplicationImpl,
    _XposeArgs,
)
from boltz.distributed.utils import update_exhaustive_strides
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


def parallel_assert_triangle_multiplication(
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
    output_global_fp32_host: torch.Tensor | None = None,
    d_input_x_global_fp32_host: torch.Tensor | None = None,
    grad_params_fp32_global_host: dict[str, torch.Tensor] | None = None,
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

    if ((output_global_fp32_host is None) != (d_input_x_global_fp32_host is None)) or (
        (output_global_fp32_host is not None) != (grad_params_fp32_global_host is not None)
    ):
        raise ValueError(
            "output_global_fp32_host, d_input_x_global_fp32_host, and grad_params_fp32_global_host must be either all None or all not None"
        )

    check_error_hist = output_global_fp32_host is not None

    layout_map = manager.layout_subgroups["cp"]
    ring_comm = Ring2DComm(manager.group["cp"], manager.subgroups["cp"][0], layout_map)

    if direction == _Direction.Outgoing:
        module_serial = TriangleMultiplicationOutgoing(dim)
    elif direction == _Direction.Incoming:
        module_serial = TriangleMultiplicationIncoming(dim)
    else:
        raise ValueError(f"Invalid direction {direction}")
    module_serial.load_state_dict(layer_state_dict)
    module_serial = module_serial.to(dtype=dtype, device=manager.device)

    if direction == _Direction.Outgoing:
        module = DistributedTriangleMultiplicationOutgoing(module_serial, manager.device_mesh_subgroups, ring_comm)
    elif direction == _Direction.Incoming:
        module = DistributedTriangleMultiplicationIncoming(module_serial, manager.device_mesh_subgroups, ring_comm)
    else:
        raise ValueError(f"Invalid direction {direction}")
    module = module.train()

    # Input tensors have the same sharding pattern:
    # x: (B, N, N, D) - sharded on dims 1 and 2 (N and N)
    # mask: (B, N, N) - sharded on dims 1 and 2 (N and N)
    placements = (Shard(0), Shard(1), Shard(2))

    # Distribute input tensors
    input_x_dtensor = distribute_tensor(
        input_x_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
    ).requires_grad_(True)

    mask_dtensor = distribute_tensor(
        mask_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
    )

    # Distribute expected outputs
    d_output_expected_dtensor = distribute_tensor(
        d_output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
    )
    output_expected_dtensor = distribute_tensor(
        output_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
        src_data_rank=None,
    )
    d_input_x_expected_dtensor = distribute_tensor(
        d_input_x_expected_global_host.to(dtype=dtype, device=manager.device),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
        src_data_rank=None,
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
            device_mesh=manager.device_mesh_subgroups,
            placements=placements,
            src_data_rank=None,
        )

        d_input_x_fp32_dtensor = distribute_tensor(
            d_input_x_global_fp32_host.to(device=manager.device),
            device_mesh=manager.device_mesh_subgroups,
            placements=placements,
            src_data_rank=None,
        )

        # Check that the output tensor has the correct shape
        assert (
            output_dtensor_result.shape == output_expected_dtensor.shape
        ), f"Output DTensor has shape {output_dtensor_result.shape} but expected shape {output_expected_dtensor.shape}"

        # Check that the output tensor has the correct shape
        assert (
            output_dtensor_result.stride() == output_expected_dtensor.stride()
        ), f"Output DTensor has stride {output_dtensor_result.stride()} but expected stride {output_expected_dtensor.stride()}"

        assert (
            input_x_dtensor.grad.shape == d_input_x_expected_dtensor.shape
        ), f"Input DTensor grad has shape {input_x_dtensor.grad.shape} but expected shape {d_input_x_expected_dtensor.shape}"

        assert (
            input_x_dtensor.grad.stride() == d_input_x_expected_dtensor.stride()
        ), f"Input DTensor grad has stride {input_x_dtensor.grad.stride()} but expected stride {d_input_x_expected_dtensor.stride()}"

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

        # Check that the output tensor has the correct shape
        assert (
            output_dtensor_result.shape == output_expected_dtensor.shape
        ), f"Output DTensor has shape {output_dtensor_result.shape} but expected shape {output_expected_dtensor.shape}"

        # Check that the output tensor has the correct shape
        assert (
            output_dtensor_result.stride() == output_expected_dtensor.stride()
        ), f"Output DTensor has stride {output_dtensor_result.stride()} but expected stride {output_expected_dtensor.stride()}"

        # Verify inputs weren't modified
        assert_tensors_identical(
            input_x_dtensor_copy.to_local(), input_x_dtensor.to_local(), check_grad=False, check_grad_fn=False
        )
        assert_tensors_identical(mask_dtensor_copy.to_local(), mask_dtensor.to_local())

        # Test forward pass results
        torch.testing.assert_close(output_dtensor_result.to_local(), output_expected_dtensor.to_local())

        # Backward pass
        d_output_expected_dtensor_copy = d_output_expected_dtensor.detach().clone()
        output_dtensor_result.backward(d_output_expected_dtensor)

        assert (
            input_x_dtensor.grad.shape == d_input_x_expected_dtensor.shape
        ), f"Input DTensor grad has shape {input_x_dtensor.grad.shape} but expected shape {d_input_x_expected_dtensor.shape}"

        assert (
            input_x_dtensor.grad.stride() == d_input_x_expected_dtensor.stride()
        ), f"Input DTensor grad has stride {input_x_dtensor.grad.stride()} but expected stride {d_input_x_expected_dtensor.stride()}"

        # Verify upstream gradient wasn't modified
        assert_tensors_identical(d_output_expected_dtensor_copy.to_local(), d_output_expected_dtensor.to_local())

        # Test input gradients
        torch.testing.assert_close(input_x_dtensor.grad.to_local(), d_input_x_expected_dtensor.to_local())

        # Test full tensor gathering - verify distributed results match serial results
        output_global_result_host = output_dtensor_result.full_tensor().cpu()
        d_input_x_global_result_host = input_x_dtensor.grad.full_tensor().cpu()

        # Verify full tensors match expected results
        torch.testing.assert_close(output_global_result_host, output_expected_global_host.to(dtype=dtype))
        torch.testing.assert_close(d_input_x_global_result_host, d_input_x_expected_global_host.to(dtype=dtype))

        # Test parameter gradients
        grad_params_result_dtensors = {}
        for name, param in module.named_parameters():
            if param.grad is not None:
                if name not in grad_params_expected_global_host:
                    # do an extra check here to make sure the parallel computation don't result in extra gradients
                    raise ValueError(f"Parameter {name} has a resulting gradient but it is not in the reference module")
                grad_params_result_dtensors[name] = param.grad

        for name, grad_param_expected_global_host in grad_params_expected_global_host.items():
            assert name in grad_params_result_dtensors, f"Parameter {name}'s gradient is not found in result gradients"
            grad_params_result = grad_params_result_dtensors[name]
            grad_params_result_global = grad_params_result.full_tensor()
            torch.testing.assert_close(grad_params_result_global.cpu(), grad_param_expected_global_host.to(dtype=dtype))
            assert_all_identical(grad_params_result_global, manager.group["cp"])

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env, dtype, check_error_hist",
    (
        params_test := [
            (((1, (2, 2)), True, "cuda", "ENV"), torch.float32, True),
            (((1, (2, 2)), True, "cuda", "ENV"), torch.float64, False),
            (((2, (2, 2)), True, "cuda", "ENV"), torch.float32, True),
            (((1, (3, 3)), True, "cuda", "ENV"), torch.float32, False),
            (((1, (3, 3)), True, "cpu", "ENV"), torch.float32, False),
        ]
    ),
    indirect=["setup_env"],
    ids=[
        f"dp:{x[0][0][0]}, cp:{x[0][0][1]}, specify_method:{x[0][1]}, device_type:{x[0][2]}, method_init:{x[0][3]}, "
        f"dtype:{x[1]}, check_error_hist:{x[2]}"
        for x in params_test
    ],
)
@pytest.mark.parametrize("direction", [_Direction.Outgoing, _Direction.Incoming])
def test_triangle_multiplication_parallel(setup_env, dtype, check_error_hist, direction):
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    # dtype is the dtype used by the parallel computation
    # check_error_hist determine whether to compare the error histograms between
    # (CP_in_FP32, serial_in_FP64) and (serial_in_FP32, serial_in_FP64)
    # Typically, check_error_hist will use large input dimensions to emulate
    # the real-world use cases. Same with dtype==torch.float64.

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() != {world_size}")

    if check_error_hist:
        if grid_group_sizes["dp"] > 1:
            pytest.skip("skip error histogram check for dp > 1 to save test time")

    # For float64 and error histogram check, we use a realistic model and input size
    # with heavier computation to test the numerical stability. On the other hand,
    # a smaller model and input size incur less numerical error accumulation to allow
    # a larger range of input values to detect logical bugs inexpensively by using
    # smaller dimensions.
    test_large_model = check_error_hist or dtype == torch.float64

    size_ring = grid_group_sizes["cp"][0]
    B = 2 * grid_group_sizes["dp"]
    if test_large_model:
        N = size_ring * 128  # Number of tokens
        dim = 128  # Hidden dimension
        min_val_init = -5e-2 if dtype == torch.float64 else -1e-3
        max_val_init = -min_val_init
    else:
        N = size_ring * 4  # Number of tokens
        dim = 8  # Hidden dimension
        min_val_init = -0.5
        max_val_init = 0.5

    seed = 42
    seed_by_rank(0, seed=seed)

    # compute reference results with FP64
    input_x_global_fp64 = torch.empty((B, N, N, dim), dtype=torch.float64, requires_grad=True, device=device_type)
    mask_global_fp64 = torch.randint(0, 2, (B, N, N), dtype=torch.float64, requires_grad=False, device=device_type)

    # emulate blocks of pure padding
    mask_global_fp64[0, N // size_ring :, :] = 0
    mask_global_fp64[0, :, N // size_ring :] = 0

    # Create reference serial module
    if direction == _Direction.Outgoing:
        reference_module = TriangleMultiplicationOutgoing(dim)
    elif direction == _Direction.Incoming:
        reference_module = TriangleMultiplicationIncoming(dim)
    else:
        raise ValueError(f"Invalid direction {direction}")

    # The output activation and gradient of the layer weights typically increase by 2 to 3 orders of magnitude,
    # where the ULP would be too large and numerical error distribution becomes very wide, i.e., we would have
    # very unpredictable numerical errors. That would make the test results very noisy and not very useful to
    # detect logical bugs in the code. To avoid this, we use a smaller range for the input and layer weights.
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
        parallel_assert_triangle_multiplication,
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


# ---------------------------------------------------------------------------
# Dtype-invariance tests for ``_TriangleMultiplicationImpl`` and
# ``_distributed_bmm``.
#
# Motivation
# ----------
# Under ``bf16-mixed`` precision (production training, see
# ``scripts/train/configs/structurev2.yaml`` and ``structurev2_cp.yaml``)
# the inputs reaching ``_TriangleMultiplicationImpl.apply`` are heterogeneous:
#
# * ``x`` and ``g``: BF16 — outputs of ``LinearParamsReplicated`` autocast.
# * ``mask``: FP32 — ``pair_mask = feats["token_pair_pad_mask"].to(z.dtype)``
#   where ``z.dtype`` is at least FP32 (mirroring the serial trunk's
#   ``compute_dtype = promote_types(s_init.dtype, FP32)``).
#
# Out-of-place ``BF16 * FP32`` in ``forward`` would type-promote ``x_local``
# to FP32 and cascade through the saved tensors, producing a FP32 gradient at
# ``g`` in backward — a silent precision/memory regression versus the serial
# reference whose final trimul output is BF16 under autocast.  These tests
# verify that the autograd function preserves ``x.dtype`` end-to-end.
# ---------------------------------------------------------------------------


def parallel_assert_triangle_multiplication_impl_dtype_invariance(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    direction,
    x_dtype,
    mask_dtype,
):
    """Verify forward/backward dtype invariants of ``_TriangleMultiplicationImpl``.

    The autograd.Function must produce outputs and gradients matching
    ``x.dtype`` regardless of ``mask.dtype``.  This guards against:

    * Forward: out-of-place ``x_local = x.to_local() * mask_local`` promoting
      to ``mask_local.dtype`` when it is wider, which then propagates through
      ``_distributed_bmm``'s ``zeros_like(lhs)`` accumulator and produces a
      forward output at the wider dtype.
    * Backward: the FP32-cascaded ``x_masked_gated_local`` saved tensor
      promoting ``dg_local = dab_local * x_masked_gated_local`` to FP32 even
      though ``dab_local`` is at the matmul dtype.
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
    layout_map = manager.layout_subgroups["cp"]
    ring_comm = Ring2DComm(manager.group["cp"], manager.subgroups["cp"][0], layout_map)

    size_ring = grid_group_sizes["cp"][0]
    B = 2 * grid_group_sizes["dp"]
    N = size_ring * 4
    dim = 8  # last-axis size; must be even (chunked into halves inside the impl)
    placements = (Shard(0), Shard(1), Shard(2))

    seed_by_rank(rank, seed=42)

    shape_xg = (B, N, N, dim)
    shape_mask = (B, N, N)
    n_local = N // size_ring

    x_local = torch.randn(B, n_local, n_local, dim, dtype=x_dtype, device=manager.device)
    g_local = torch.randn(B, n_local, n_local, dim, dtype=x_dtype, device=manager.device)
    # mask is binary in production; we draw a 0/1 pattern then cast to mask_dtype.
    mask_local = (torch.rand(B, n_local, n_local, device=manager.device) > 0.5).to(dtype=mask_dtype)

    stride_xg = update_exhaustive_strides(x_local.shape, x_local.stride(), shape_xg)
    stride_mask = update_exhaustive_strides(mask_local.shape, mask_local.stride(), shape_mask)

    x_dt = DTensor.from_local(
        x_local.detach().clone(),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
        shape=shape_xg,
        stride=stride_xg,
    ).requires_grad_(True)
    g_dt = DTensor.from_local(
        g_local.detach().clone(),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
        shape=shape_xg,
        stride=stride_xg,
    ).requires_grad_(True)
    mask_dt = DTensor.from_local(
        mask_local.detach().clone(),
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
        shape=shape_mask,
        stride=stride_mask,
    )

    out_dt = _TriangleMultiplicationImpl.apply(x_dt, mask_dt, g_dt, ring_comm, direction)

    # Forward invariant: output dtype follows x's dtype regardless of mask dtype.
    assert out_dt.dtype == x_dtype, (
        f"_TriangleMultiplicationImpl forward output dtype {out_dt.dtype} "
        f"does not match x_dtype {x_dtype} (mask_dtype={mask_dtype}, direction={direction})"
    )
    assert out_dt.to_local().dtype == x_dtype, (
        f"_TriangleMultiplicationImpl forward output local dtype "
        f"{out_dt.to_local().dtype} does not match x_dtype {x_dtype}"
    )

    grad_out_local = torch.randn_like(out_dt.to_local()).contiguous()
    grad_out_dt = DTensor.from_local(
        grad_out_local,
        device_mesh=manager.device_mesh_subgroups,
        placements=placements,
        shape=out_dt.shape,
        stride=out_dt.stride(),
    )
    out_dt.backward(grad_out_dt)

    # Backward invariants: both dx and dg must match x_dtype.
    # ``dx`` is incidentally preserved by the in-place ``dx_local *= mask_local``
    # which keeps the LHS dtype, so the historical bug surfaces in ``dg``.
    assert x_dt.grad.dtype == x_dtype, (
        f"x.grad dtype {x_dt.grad.dtype} does not match x_dtype {x_dtype} "
        f"(mask_dtype={mask_dtype}, direction={direction})"
    )
    assert g_dt.grad.dtype == x_dtype, (
        f"g.grad dtype {g_dt.grad.dtype} does not match x_dtype {x_dtype} "
        f"(mask_dtype={mask_dtype}, direction={direction})"
    )
    assert x_dt.grad.to_local().dtype == x_dtype
    assert g_dt.grad.to_local().dtype == x_dtype

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, (2, 2)), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=["dp:1, cp:(2,2), cuda, ENV"],
)
@pytest.mark.parametrize("direction", [_Direction.Outgoing, _Direction.Incoming])
@pytest.mark.parametrize(
    "x_dtype, mask_dtype",
    [
        # Production scenario: BF16 activations + FP32 pair_mask.
        # Before the fix this case produced FP32 forward output and FP32 dg.
        (torch.bfloat16, torch.float32),
        # Sanity: all BF16 — no upcast risk, must remain BF16 end-to-end.
        (torch.bfloat16, torch.bfloat16),
        # Sanity: all FP32 — matches existing parity tests, must remain FP32.
        (torch.float32, torch.float32),
    ],
    ids=[
        "x=bf16,mask=fp32",
        "x=bf16,mask=bf16",
        "x=fp32,mask=fp32",
    ],
)
def test_triangle_multiplication_impl_dtype_invariance(setup_env, direction, x_dtype, mask_dtype):
    """Guard ``_TriangleMultiplicationImpl`` fwd/bwd against silent dtype cascades.

    The autograd function must produce a forward output and both backward
    gradients (``dx``, ``dg``) at ``x.dtype`` regardless of ``mask.dtype``.

    The historical bug: under ``bf16-mixed`` autocast the mask reaches the
    impl as FP32 while ``x``/``g`` are BF16; the out-of-place
    ``x.to_local() * mask_local`` then upcasts ``x_local`` to FP32, which
    cascades into the ``_distributed_bmm`` accumulator and into the saved
    ``x_masked_gated_local``, ultimately producing a FP32 dg.  See the
    docstring of ``_TriangleMultiplicationImpl.forward`` for the full trace.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("skip cuda test because torch.cuda.is_available == False")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        parallel_assert_triangle_multiplication_impl_dtype_invariance,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        direction,
        x_dtype,
        mask_dtype,
    )


def parallel_assert_distributed_bmm_dtype_invariance(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    use_autocast,
):
    """Verify ``_distributed_bmm`` preserves the input dtype end-to-end.

    Production usage in ``_TriangleMultiplicationImpl`` always passes ``lhs``
    and ``rhs`` at the same dtype (both shards of the masked-gated ``x``).
    Under that contract, the partial-sum accumulator is seeded by
    ``zeros_like(lhs)`` and every ``torch.matmul(lhs, rhs)`` returns the same
    dtype, so the final output dtype must equal ``lhs.dtype``.

    This test is also run under ``torch.autocast`` to confirm that
    ``_distributed_bmm`` does **not** downcast to the autocast dtype when
    inputs are at a wider dtype — the accumulator dtype acts as a floor.
    This matches the serial ``torch.einsum`` behaviour for same-dtype inputs.
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
    layout_map = manager.layout_subgroups["cp"]
    ring_comm = Ring2DComm(manager.group["cp"], manager.subgroups["cp"][0], layout_map)

    seed_by_rank(rank, seed=42)

    # Shapes mirror the Outgoing call pattern in
    # ``_TriangleMultiplicationImpl.forward``: lhs (B, n, k, D), rhs (B, m, k, D).
    B = 1
    n_local = 2
    m_local = 2
    k_local = 2
    feat = 4

    lhs_local = torch.randn(B, n_local, k_local, feat, dtype=dtype, device=manager.device)
    rhs_local = torch.randn(B, m_local, k_local, feat, dtype=dtype, device=manager.device)

    def _run_bmm() -> torch.Tensor:
        return _distributed_bmm(
            lhs_local,
            rhs_local,
            ring_comm,
            permute_lhs=(0, 3, 1, 2),  # (B, n, k, D) -> (B, D, n, k)
            permute_rhs=(0, 3, 2, 1),  # (B, m, k, D) -> (B, D, k, m)
            permute_out=(0, 2, 3, 1),  # (B, D, n, m) -> (B, n, m, D)
            xpose_args=_XposeArgs.rhs,
        )

    if use_autocast:
        # bf16-mixed mirrors production training.  When inputs are already at
        # ``dtype``, autocast must not change the output dtype: the accumulator
        # ``zeros_like(lhs)`` is at ``dtype`` and ``torch.matmul`` returns
        # ``dtype`` for same-dtype same-precision inputs.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = _run_bmm()
    else:
        out = _run_bmm()

    assert out.dtype == dtype, (
        f"_distributed_bmm output dtype {out.dtype} does not match input dtype {dtype} "
        f"(use_autocast={use_autocast})"
    )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, (2, 2)), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=["dp:1, cp:(2,2), cuda, ENV"],
)
@pytest.mark.parametrize(
    "dtype, use_autocast",
    [
        (torch.bfloat16, False),
        (torch.float32, False),
        # bf16 inputs under bf16 autocast — output stays bf16.
        (torch.bfloat16, True),
        # fp32 inputs under bf16 autocast — autocast must not silently
        # downcast.  This pins the "accumulator dtype is a floor" contract.
        (torch.float32, True),
    ],
    ids=[
        "dtype=bf16,autocast=False",
        "dtype=fp32,autocast=False",
        "dtype=bf16,autocast=bf16",
        "dtype=fp32,autocast=bf16",
    ],
)
def test_distributed_bmm_dtype_invariance(setup_env, dtype, use_autocast):
    """Guard ``_distributed_bmm`` against silent dtype shifts.

    The invariant verified here:
    ``_distributed_bmm(lhs, rhs, ...).dtype == lhs.dtype == rhs.dtype`` when
    ``lhs`` and ``rhs`` share a dtype, regardless of autocast state.  This is
    the only configuration that ``_TriangleMultiplicationImpl`` exercises in
    production after the mask cast in ``forward``; callers that mix dtypes
    are out of scope (and would hit ``torch.matmul`` promotion rules whose
    behaviour varies across PyTorch versions).
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("skip cuda test because torch.cuda.is_available == False")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        parallel_assert_distributed_bmm_dtype_invariance,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        use_autocast,
    )


# ---------------------------------------------------------------------------
# End-to-end dtype-invariance test for the full distributed
# ``TriangleMultiplication`` wrapper (``LayerNormParamsReplicated`` →
# ``LinearParamsReplicated`` × 4 → ``_TriangleMultiplicationImpl`` →
# ``LayerNormParamsReplicated`` → ``LinearParamsReplicated`` → ``sigmoid_gate``)
# under ``bf16-mixed`` autocast — the production setup.
#
# This mirrors the autocast-driven dtype regression pattern in
# ``tests/distributed/test_dtensor_linear.py::test_dtensor_linear_bf16_gradient_promotion``
# but skips numerical verification — only the dtype contract is asserted.
# ---------------------------------------------------------------------------


def parallel_assert_triangle_multiplication_dtype_invariance(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    direction,
    mask_dtype,
):
    """Verify dtype invariants of the distributed ``TriangleMultiplication``
    under ``bf16-mixed`` autocast.

    Production configuration mirrored here (distributed trunk in
    ``src/boltz/distributed/model/models/boltz2.py``):

    * Parameters are FP32 (``module.to(dtype=fp32)``).
    * Activations (``x``) arrive at the wrapper as FP32.  Although
      Lightning's ``bf16-mixed`` plugin wraps ``forward`` in
      ``torch.autocast(dtype=bf16)`` so every Linear produces BF16,
      ``z_init`` becomes FP32 because of additive terms not eligible
      for autocast — most notably ``self.token_bonds_type(...)`` which is
      an ``EmbeddingParamsReplicated`` wrapping ``F.embedding`` (a gather,
      not a matmul; not autocast-eligible) and therefore returns at the
      embedding weight dtype FP32.  ``elementwise_op(BF16, FP32, SUM)``
      then promotes ``z_init`` to FP32, and the pairformer residual
      ``z = z + tri_mul(z, pair_mask)`` keeps ``z`` at FP32 via
      ``FP32 + BF16 → FP32`` type promotion.  This matches the serial
      reference whose ``compute_dtype = promote_types(s_init.dtype, FP32)``
      pin is FP32 for the same reason (autocast Linear outputs are BF16
      but ``compute_dtype`` clamps to ≥ FP32).
    * ``pair_mask = feats["token_pair_pad_mask"].to(dtype=z.dtype)`` is
      therefore FP32 in production — this is the regression case the mask
      cast in ``_TriangleMultiplicationImpl.forward`` is designed to handle.
      Parametrised here over ``mask_dtype`` to keep the contract explicit.
    * The forward call is wrapped in ``torch.autocast(dtype=bf16)`` to
      match how the Lightning plugin invokes the training step.

    Invariants verified after one forward + backward step:

    * ``output.dtype == bf16`` — under bf16-mixed autocast the first
      autocast-eligible op (``F.linear`` inside ``g_in``/``p_in``) casts the
      FP32 input to BF16; every downstream sub-layer uses
      ``cast_params_dtype_to_x=True`` so the chain stays BF16 even with
      FP32 parameters.  This catches gross autocast-wiring regressions —
      e.g. a sub-layer that disables autocast or unconditionally upcasts
      to FP32.  Note that this assertion is *not* sensitive to the FP32
      mask cascade fix inside ``_TriangleMultiplicationImpl`` because the
      downstream ``p_out`` Linear under autocast re-casts the impl's output
      to BF16; that regression is covered by
      ``test_triangle_multiplication_impl_dtype_invariance`` which tests
      the impl in isolation.
    * ``x.grad.dtype == x.dtype (= FP32)`` — autograd inserts an inverse
      cast at the same boundary where autocast cast FP32→BF16 in forward,
      so the gradient is cast back to FP32 before reaching ``AccumulateGrad``.
      The post-``AccumulateGrad`` ``x.grad`` is therefore FP32.  This
      mirrors the production contract: the trunk holds ``z`` at FP32, so
      ``z.grad`` is FP32 and feeds into the FP32-promoted upstream
      parameter updates.
    * ``param.grad.dtype == param.dtype (= FP32)`` for every parameter —
      ``AccumulateGrad`` casts the backward-returned grad to ``param.dtype``
      (FP32) before storing.  The optimizer therefore always sees FP32
      gradients regardless of the BF16 compute dtype used in backward.
    """
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    try:
        DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
        manager = DistributedManager()
        layout_map = manager.layout_subgroups["cp"]
        ring_comm = Ring2DComm(manager.group["cp"], manager.subgroups["cp"][0], layout_map)

        # Production setup: FP32 parameters + FP32 activations arriving at
        # the wrapper.  ``z`` is FP32 in the trunk because
        # ``self.token_bonds_type`` (Embedding, not autocast-eligible)
        # contributes FP32 to ``z_init`` and the pairformer residual
        # ``z + tri_mul(z, pair_mask)`` keeps ``z`` at FP32 via
        # ``FP32 + BF16 → FP32`` promotion.
        param_dtype = torch.float32
        x_dtype = torch.float32
        expected_compute_dtype = torch.bfloat16

        # Same dimensions as the small-model branch of
        # ``test_triangle_multiplication_parallel`` to keep collective
        # traffic cheap while still exercising the (B, N, N, D) → sharded
        # ring pattern.
        size_ring = grid_group_sizes["cp"][0]
        B = 2 * grid_group_sizes["dp"]
        N = size_ring * 4
        dim = 8

        seed_by_rank(0, seed=42)

        if direction == _Direction.Outgoing:
            module_serial = TriangleMultiplicationOutgoing(dim)
        elif direction == _Direction.Incoming:
            module_serial = TriangleMultiplicationIncoming(dim)
        else:
            raise ValueError(f"Invalid direction {direction}")
        init_module_params_uniform(module_serial, low=-0.5, high=0.5)
        module_serial = module_serial.to(dtype=param_dtype, device=manager.device)

        if direction == _Direction.Outgoing:
            module = DistributedTriangleMultiplicationOutgoing(module_serial, manager.device_mesh_subgroups, ring_comm)
        elif direction == _Direction.Incoming:
            module = DistributedTriangleMultiplicationIncoming(module_serial, manager.device_mesh_subgroups, ring_comm)
        else:
            raise ValueError(f"Invalid direction {direction}")
        module = module.train()

        for name, param in module.named_parameters():
            assert (
                param.dtype == param_dtype
            ), f"Parameter '{name}' dtype {param.dtype} does not match param_dtype {param_dtype}"

        placements = (Shard(0), Shard(1), Shard(2))

        input_x_global = torch.empty((B, N, N, dim), dtype=x_dtype, device=manager.device)
        init_tensors_uniform([input_x_global], low=-0.5, high=0.5)
        mask_global = torch.randint(0, 2, (B, N, N), device=manager.device).to(dtype=mask_dtype)

        input_x_dtensor = distribute_tensor(
            input_x_global,
            device_mesh=manager.device_mesh_subgroups,
            placements=placements,
        ).requires_grad_(True)
        mask_dtensor = distribute_tensor(
            mask_global,
            device_mesh=manager.device_mesh_subgroups,
            placements=placements,
        )

        with torch.autocast(device_type=device_type, dtype=expected_compute_dtype, enabled=True):
            output_dtensor = module(input_x_dtensor, mask_dtensor)

        # Forward dtype invariant: autocast must produce BF16 output.
        # This catches gross autocast-wiring regressions across the chain;
        # the impl-internal FP32-mask cascade is masked here by the
        # downstream ``p_out`` Linear that re-casts to BF16 under autocast
        # — that regression is covered by
        # ``test_triangle_multiplication_impl_dtype_invariance``.
        assert output_dtensor.dtype == expected_compute_dtype, (
            f"Forward output dtype {output_dtensor.dtype} does not match expected "
            f"{expected_compute_dtype} (mask_dtype={mask_dtype}, direction={direction})"
        )
        assert output_dtensor.to_local().dtype == expected_compute_dtype, (
            f"Forward output local dtype {output_dtensor.to_local().dtype} does not match " f"{expected_compute_dtype}"
        )

        # Upstream adjoint at the forward-output (BF16) dtype.  ``custom_bwd``
        # restores autocast on CUDA so backward operates in BF16 between the
        # cast boundary and the output; autograd then casts back to FP32 at
        # the same boundary where forward autocast cast FP32→BF16.
        grad_output_local = torch.randn(
            B, N // size_ring, N // size_ring, dim, dtype=expected_compute_dtype, device=manager.device
        )
        grad_output_dtensor = DTensor.from_local(
            grad_output_local,
            device_mesh=manager.device_mesh_subgroups,
            placements=placements,
            shape=output_dtensor.shape,
            stride=output_dtensor.stride(),
        )
        output_dtensor.backward(grad_output_dtensor)

        # ``x.grad`` is post-``AccumulateGrad`` so it is always at ``x.dtype``
        # (FP32 here) — the BF16 backward chain is cast back to FP32 at the
        # autocast boundary before reaching the leaf.
        assert input_x_dtensor.grad is not None, "Backward did not populate input_x_dtensor.grad"
        assert input_x_dtensor.grad.dtype == x_dtype, (
            f"x.grad dtype {input_x_dtensor.grad.dtype} does not match x.dtype "
            f"{x_dtype} (mask_dtype={mask_dtype}, direction={direction})"
        )
        assert input_x_dtensor.grad.to_local().dtype == x_dtype, (
            f"x.grad local dtype {input_x_dtensor.grad.to_local().dtype} does not match " f"{x_dtype}"
        )

        # Parameter gradients accumulate at ``param.dtype`` (FP32) — the
        # optimizer always sees FP32 grads.  We walk ``named_parameters``
        # so the error message identifies the offending sub-layer.
        n_params_checked = 0
        for name, param in module.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' did not receive a gradient in backward"
            assert param.grad.dtype == param_dtype, (
                f"Parameter '{name}' grad dtype {param.grad.dtype} does not match param_dtype "
                f"{param_dtype} (mask_dtype={mask_dtype}, direction={direction})"
            )
            assert param.grad.to_local().dtype == param_dtype, (
                f"Parameter '{name}' grad local dtype {param.grad.to_local().dtype} does not match " f"{param_dtype}"
            )
            n_params_checked += 1
        # Non-vacuous: 4 Linear (weight+bias) + 2 LayerNorm (weight+bias) = 12 params.
        # Use >= 8 to be robust against optional biases.
        assert n_params_checked >= 8, (
            f"Expected at least 8 parameter grads to inspect, only saw {n_params_checked}. "
            "Parameter enumeration may have silently skipped sub-layers."
        )
    finally:
        DistributedManager.cleanup()
        monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, (2, 2)), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=["dp:1, cp:(2,2), cuda, ENV"],
)
@pytest.mark.parametrize("direction", [_Direction.Outgoing, _Direction.Incoming])
@pytest.mark.parametrize(
    "mask_dtype",
    [
        # Production mask dtype: ``pair_mask = feats[...].to(dtype=z.dtype)``
        # in the distributed trunk is FP32 because ``z`` is FP32 (see worker
        # docstring — ``token_bonds_type`` Embedding contributes FP32 to
        # ``z_init`` which the pairformer residual then preserves).  This
        # is also the regression case for the mask cast in
        # ``_TriangleMultiplicationImpl.forward``.
        torch.float32,
    ],
    ids=["mask=fp32"],
)
def test_triangle_multiplication_dtype_invariance(setup_env, direction, mask_dtype):
    """End-to-end dtype-invariance test for distributed ``TriangleMultiplication``
    under ``bf16-mixed`` autocast.

    Companion to ``test_triangle_multiplication_parallel`` (numerical parity
    against the serial FP64 reference) and to
    ``test_triangle_multiplication_impl_dtype_invariance``
    (sub-layer dtype contract).  This test pins the production
    forward/backward dtype invariants for the full wrapper:

    * forward output is BF16 (autocast compute dtype);
    * ``x.grad`` is ``x.dtype`` (FP32) via the autograd cast at the
      autocast boundary plus ``AccumulateGrad``;
    * ``param.grad`` is ``param.dtype`` (FP32) via ``AccumulateGrad``.

    Modelled on ``test_dtensor_linear_bf16_gradient_promotion`` in
    ``tests/distributed/test_dtensor_linear.py`` adapted for the distributed
    Boltz2 trunk's actual production dtype: ``z`` is FP32 (the
    ``token_bonds_type`` Embedding contributes FP32 to ``z_init`` and the
    pairformer residual preserves it), so the input arriving at TriMul is
    FP32 and ``pair_mask`` follows.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env

    if not torch.cuda.is_available():
        pytest.skip("skip cuda test because torch.cuda.is_available == False")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    spawn_multiprocessing(
        parallel_assert_triangle_multiplication_dtype_invariance,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        direction,
        mask_dtype,
    )
