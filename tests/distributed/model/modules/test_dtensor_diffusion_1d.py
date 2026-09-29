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

"""Tests for 1D CP DiffusionTransformerLayer1D, DiffusionTransformer1D,
AtomDiffusion1D helpers, sample(), and compute_loss().

Verifies forward and backward parity against serial Boltz-2 modules on
a 2D mesh ``(dp, cp)`` with 2-element placements ``(Shard(0), Shard(1))``.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

import boltz.distributed.model.modules.diffusion_1d as diffusion_1d_module
import boltz.model.modules.diffusionv2 as serial_diffusion_v2_module
from boltz.data import const as boltz_const
from boltz.distributed.data.module.placements_1d import TRAINING_FEATURE_PLACEMENTS_1D
from boltz.distributed.data.utils import distribute_features
from boltz.distributed.manager import DistributedManager
from boltz.distributed.model.layers.elementwise_op import ElementwiseOp, scalar_tensor_op
from boltz.distributed.model.modules.diffusion_1d import (
    AtomDiffusion1D,
    DiffusionTransformer1D,
    DiffusionTransformerLayer1D,
)
from boltz.distributed.model.modules.diffusion_conditioning_1d import (
    DiffusionConditioning1D,
)
from boltz.distributed.port_utils import find_free_port
from boltz.model.modules.diffusion_conditioning import (
    DiffusionConditioning as SerialDiffusionConditioning,
)
from boltz.model.modules.diffusionv2 import AtomDiffusion as SerialAtomDiffusionV2
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer as SerialDiffusionTransformerV2,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformerLayer as SerialDiffusionTransformerLayerV2,
)
from boltz.testing.utils import (
    SetModuleInfValues,
    assert_all_identical,
    assert_tensors_identical,
    get_param_by_key,
    init_module_params_uniform,
    init_tensors_uniform,
    random_features,
    seed_by_rank,
    skip_if_cuda_not_avail_or_device_count_less_than_word_size,
    spawn_multiprocessing,
)

# Symmetric init range keeps outputs O(1), enabling meaningful tolerances.
# Uniform(0, 1) produces all-positive weights that compound through matmuls,
# inflating outputs to O(1e3)–O(1e4) and requiring vacuously loose atol.
# (-0.2, 0.2) matches the 2D CP DiffusionTransformerLayer test.
_INIT_LOW, _INIT_HIGH = -0.2, 0.2

# ---------------------------------------------------------------------------
# DiffusionTransformerLayer1D parity test
# ---------------------------------------------------------------------------


def parallel_assert_dtl_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    heads,
    dim,
    dim_single_cond,
    post_layer_norm,
    multiplicity,
    B,
    N,
    layer_state_dict,
    a_global_host,
    s_global_host,
    z_global_host,
    mask_global_host,
    d_out_global_host,
    out_expected_global_host,
    d_a_expected_global_host,
    d_s_expected_global_host,
    d_z_expected_global_host,
    expected_param_grads_host,
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

    # Guard: if NCCL init failed silently (e.g. insufficient GPUs for cp=3),
    # DistributedManager may be "initialized" but with no process groups.
    try:
        cp_group = manager.group["cp"]
    except KeyError:
        DistributedManager.cleanup()
        return

    # Recreate serial module from state dict
    module_serial = SerialDiffusionTransformerLayerV2(
        heads=heads,
        dim=dim,
        dim_single_cond=dim_single_cond,
        post_layer_norm=post_layer_norm,
    )
    # Cast to target dtype BEFORE load_state_dict to avoid FP64->FP32->FP64
    # round-trip precision loss from default FP32 parameter buffers
    # (CLAUDE.md: "Convert module to target dtype before `load_state_dict`").
    module_serial = module_serial.to(device=manager.device, dtype=dtype).train()
    module_serial.load_state_dict(layer_state_dict)
    module_serial.apply(SetModuleInfValues())

    # 1D CP uses 2D mesh (dp, cp). In DistributedManager, for 1D CP the mesh
    # is manager.device_mesh (2D), not device_mesh_subgroups (3D).
    device_mesh = manager.device_mesh

    module = DiffusionTransformerLayer1D(
        layer=module_serial,
        device_mesh=device_mesh,
        cp_group=cp_group,
    ).train()

    # 1D CP placements: (Shard(0), Shard(1))
    placements_single = (Shard(0), Shard(1))

    a_dt = distribute_tensor(
        a_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    s_dt = distribute_tensor(
        s_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    # z has shape [B, N, N, H] with placements (Shard(0), Shard(1)) = row-slab
    z_dt = distribute_tensor(
        z_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    mask_dt = distribute_tensor(
        mask_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(False)

    # Forward
    out_dt = module(a_dt, s_dt, z_dt, mask=mask_dt, multiplicity=multiplicity)

    # fp64 assert_close defaults (atol=1e-7, rtol=1.3e-6) for both fwd and bwd.
    _mask_full = mask_dt.full_tensor().unsqueeze(-1)
    if multiplicity > 1:
        _mask_full = _mask_full.repeat_interleave(multiplicity, 0)
    torch.testing.assert_close(
        out_dt.full_tensor() * _mask_full,
        out_expected_global_host.to(device=manager.device, dtype=dtype) * _mask_full,
    )

    # Backward
    d_out_dt = distribute_tensor(
        d_out_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    )
    out_dt.backward(d_out_dt)

    # Input grad parity
    torch.testing.assert_close(
        a_dt.grad.full_tensor().cpu(),
        d_a_expected_global_host.to(dtype=dtype),
    )
    torch.testing.assert_close(
        s_dt.grad.full_tensor().cpu(),
        d_s_expected_global_host.to(dtype=dtype),
    )
    torch.testing.assert_close(
        z_dt.grad.full_tensor().cpu(),
        d_z_expected_global_host.to(dtype=dtype),
    )

    # Parameter grad parity
    for name, grad_expected in expected_param_grads_host.items():
        grad_param = get_param_by_key(module, name).grad
        assert grad_param is not None, f"Missing grad for param {name}"
        if isinstance(grad_param, DTensor):
            grad_global = grad_param.full_tensor().cpu()
        else:
            grad_global = grad_param.detach().cpu()
        torch.testing.assert_close(grad_global, grad_expected.to(dtype=dtype))
        if isinstance(grad_param, DTensor):
            assert_all_identical(grad_param.full_tensor(), cp_group)


@pytest.mark.parametrize(
    "grid_group_sizes,device_type,backend",
    [
        pytest.param({"dp": 1, "cp": 2}, "cpu", "gloo", id="dp1_cp2_cpu"),
        pytest.param({"dp": 1, "cp": 3}, "cpu", "gloo", id="dp1_cp3_cpu"),
        # cp=2 CUDA exercises the 2-rank NCCL all-reduce path, which uses a
        # different collective code-path inside NCCL than cp=3 (ring-of-2
        # vs ring-of-3). Without this entry the module's cp=2 NCCL path
        # is only reached transitively through higher-level integration
        # tests (e.g. test_atom_diffusion_sample_1d), which is fragile.
        pytest.param({"dp": 1, "cp": 2}, "cuda", "nccl", id="dp1_cp2_cuda"),
        pytest.param({"dp": 1, "cp": 3}, "cuda", "nccl", id="dp1_cp3_cuda"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize("dim", [16])
@pytest.mark.parametrize("dim_single_cond", [16])
@pytest.mark.parametrize("post_layer_norm", [False, True])
@pytest.mark.parametrize("multiplicity", [1])
@pytest.mark.parametrize("B,N", [(1, 12)])
def test_diffusion_transformer_layer_1d(
    grid_group_sizes,
    device_type,
    backend,
    dtype,
    heads,
    dim,
    dim_single_cond,
    post_layer_norm,
    multiplicity,
    B,
    N,
):
    """Test DiffusionTransformerLayer1D forward/backward parity vs serial."""
    world_size = grid_group_sizes["dp"] * grid_group_sizes["cp"]
    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type, world_size)

    torch.manual_seed(42)

    # Create serial module
    module_serial = SerialDiffusionTransformerLayerV2(
        heads=heads,
        dim=dim,
        dim_single_cond=dim_single_cond,
        post_layer_norm=post_layer_norm,
    )
    module_serial = module_serial.to(dtype=dtype)
    init_module_params_uniform(module_serial, low=_INIT_LOW, high=_INIT_HIGH)
    module_serial.apply(SetModuleInfValues())

    # Create inputs with symmetric range to keep activations O(1)
    BM = B * multiplicity
    a_global = torch.empty(BM, N, dim, dtype=dtype, requires_grad=True)
    s_global = torch.empty(BM, N, dim_single_cond, dtype=dtype, requires_grad=True)
    z_global = torch.empty(B, N, N, heads, dtype=dtype, requires_grad=True)
    mask_global = torch.ones(B, N, dtype=dtype)
    d_out_global = torch.empty(BM, N, dim, dtype=dtype)
    init_tensors_uniform([a_global, s_global, z_global, d_out_global], low=_INIT_LOW, high=_INIT_HIGH)

    # Serial forward
    with torch.no_grad():
        out_serial = module_serial(
            a_global.detach().clone().requires_grad_(True),
            s_global.detach().clone().requires_grad_(True),
            z_global.detach().clone().requires_grad_(True),
            mask=mask_global,
            multiplicity=multiplicity,
        )

    # Serial backward
    a_serial = a_global.detach().clone().requires_grad_(True)
    s_serial = s_global.detach().clone().requires_grad_(True)
    z_serial = z_global.detach().clone().requires_grad_(True)
    out_serial2 = module_serial(a_serial, s_serial, z_serial, mask=mask_global, multiplicity=multiplicity)
    out_serial2.backward(d_out_global)

    expected_param_grads = {}
    for name, param in module_serial.named_parameters():
        if param.grad is not None:
            expected_param_grads[name] = param.grad.detach().cpu()

    world_size = grid_group_sizes["dp"] * grid_group_sizes["cp"]
    env_per_rank = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(find_free_port()),
        "RANK": "<INPUT_RANK>",
        "WORLD_SIZE": str(world_size),
    }

    spawn_multiprocessing(
        parallel_assert_dtl_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        heads,
        dim,
        dim_single_cond,
        post_layer_norm,
        multiplicity,
        B,
        N,
        module_serial.state_dict(),
        a_global.detach().cpu(),
        s_global.detach().cpu(),
        z_global.detach().cpu(),
        mask_global.detach().cpu(),
        d_out_global.detach().cpu(),
        out_serial.detach().cpu(),
        a_serial.grad.detach().cpu(),
        s_serial.grad.detach().cpu(),
        z_serial.grad.detach().cpu(),
        expected_param_grads,
    )


# ---------------------------------------------------------------------------
# DiffusionTransformer1D (multi-layer) parity test
# ---------------------------------------------------------------------------


def parallel_assert_dt_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    heads,
    dim,
    dim_single_cond,
    depth,
    multiplicity,
    B,
    N,
    state_dict,
    a_global_host,
    s_global_host,
    z_global_host,
    mask_global_host,
    d_out_global_host,
    out_expected_global_host,
    d_a_expected_global_host,
    d_s_expected_global_host,
    d_z_expected_global_host,
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

    # Guard: if NCCL init failed silently (e.g. insufficient GPUs for cp=3),
    # DistributedManager may be "initialized" but with no process groups.
    try:
        cp_group = manager.group["cp"]
    except KeyError:
        DistributedManager.cleanup()
        return

    module_serial = SerialDiffusionTransformerV2(
        depth=depth,
        heads=heads,
        dim=dim,
        dim_single_cond=dim_single_cond,
    )
    # Cast to target dtype BEFORE load_state_dict to avoid FP64->FP32->FP64
    # round-trip precision loss from default FP32 parameter buffers
    # (CLAUDE.md: "Convert module to target dtype before `load_state_dict`").
    module_serial = module_serial.to(device=manager.device, dtype=dtype).train()
    module_serial.load_state_dict(state_dict)
    module_serial.apply(SetModuleInfValues())

    device_mesh = manager.device_mesh

    module = DiffusionTransformer1D(
        layer=module_serial,
        device_mesh=device_mesh,
        cp_group=cp_group,
    ).train()

    placements_single = (Shard(0), Shard(1))

    a_dt = distribute_tensor(
        a_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    s_dt = distribute_tensor(
        s_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    z_dt = distribute_tensor(
        z_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(True)
    mask_dt = distribute_tensor(
        mask_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    ).requires_grad_(False)

    out_dt = module(a_dt, s_dt, z_dt, mask=mask_dt, multiplicity=multiplicity)

    # fp64 assert_close defaults (atol=1e-7, rtol=1.3e-6) for both fwd and bwd.
    _mask_full = mask_dt.full_tensor().unsqueeze(-1)
    if multiplicity > 1:
        _mask_full = _mask_full.repeat_interleave(multiplicity, 0)
    torch.testing.assert_close(
        out_dt.full_tensor() * _mask_full,
        out_expected_global_host.to(device=manager.device, dtype=dtype) * _mask_full,
    )

    # Backward
    d_out_dt = distribute_tensor(
        d_out_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    )
    out_dt.backward(d_out_dt)

    torch.testing.assert_close(
        a_dt.grad.full_tensor().cpu(),
        d_a_expected_global_host.to(dtype=dtype),
    )
    torch.testing.assert_close(
        s_dt.grad.full_tensor().cpu(),
        d_s_expected_global_host.to(dtype=dtype),
    )
    torch.testing.assert_close(
        z_dt.grad.full_tensor().cpu(),
        d_z_expected_global_host.to(dtype=dtype),
    )

    # Replicated parameter gradients must be identical across CP ranks
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        grad_param = param.grad
        if isinstance(grad_param, DTensor):
            assert_all_identical(grad_param.full_tensor(), cp_group)


@pytest.mark.parametrize(
    "grid_group_sizes,device_type,backend",
    [
        pytest.param({"dp": 1, "cp": 2}, "cpu", "gloo", id="dp1_cp2_cpu"),
        pytest.param({"dp": 1, "cp": 3}, "cpu", "gloo", id="dp1_cp3_cpu"),
        # cp=2 CUDA — see rationale on the corresponding layer-level entry above.
        pytest.param({"dp": 1, "cp": 2}, "cuda", "nccl", id="dp1_cp2_cuda"),
        pytest.param({"dp": 1, "cp": 3}, "cuda", "nccl", id="dp1_cp3_cuda"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float64])
@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize("dim", [16])
@pytest.mark.parametrize("dim_single_cond", [16])
@pytest.mark.parametrize("depth", [2])
@pytest.mark.parametrize("multiplicity", [1])
@pytest.mark.parametrize("B,N", [(1, 12)])
def test_diffusion_transformer_1d(
    grid_group_sizes,
    device_type,
    backend,
    dtype,
    heads,
    dim,
    dim_single_cond,
    depth,
    multiplicity,
    B,
    N,
):
    """Test DiffusionTransformer1D (multi-layer) forward/backward parity vs serial."""
    world_size = grid_group_sizes["dp"] * grid_group_sizes["cp"]
    skip_if_cuda_not_avail_or_device_count_less_than_word_size(device_type, world_size)

    torch.manual_seed(42)

    module_serial = SerialDiffusionTransformerV2(
        depth=depth,
        heads=heads,
        dim=dim,
        dim_single_cond=dim_single_cond,
    )
    module_serial = module_serial.to(dtype=dtype)
    init_module_params_uniform(module_serial, low=_INIT_LOW, high=_INIT_HIGH)
    module_serial.apply(SetModuleInfValues())

    # Create inputs with symmetric range to keep activations O(1)
    BM = B * multiplicity
    a_global = torch.empty(BM, N, dim, dtype=dtype, requires_grad=True)
    s_global = torch.empty(BM, N, dim_single_cond, dtype=dtype, requires_grad=True)
    # Boltz-2 bias has last dim = heads * depth, split across layers
    z_global = torch.empty(B, N, N, heads * depth, dtype=dtype, requires_grad=True)
    mask_global = torch.ones(B, N, dtype=dtype)
    d_out_global = torch.empty(BM, N, dim, dtype=dtype)
    init_tensors_uniform([a_global, s_global, z_global, d_out_global], low=_INIT_LOW, high=_INIT_HIGH)

    # Serial forward
    with torch.no_grad():
        out_serial = module_serial(
            a_global.detach().clone(),
            s_global.detach().clone(),
            bias=z_global.detach().clone(),
            mask=mask_global,
            multiplicity=multiplicity,
        )

    # Serial backward
    a_serial = a_global.detach().clone().requires_grad_(True)
    s_serial = s_global.detach().clone().requires_grad_(True)
    z_serial = z_global.detach().clone().requires_grad_(True)
    out_serial2 = module_serial(a_serial, s_serial, bias=z_serial, mask=mask_global, multiplicity=multiplicity)
    out_serial2.backward(d_out_global)

    world_size = grid_group_sizes["dp"] * grid_group_sizes["cp"]
    env_per_rank = {
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(find_free_port()),
        "RANK": "<INPUT_RANK>",
        "WORLD_SIZE": str(world_size),
    }

    spawn_multiprocessing(
        parallel_assert_dt_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        heads,
        dim,
        dim_single_cond,
        depth,
        multiplicity,
        B,
        N,
        module_serial.state_dict(),
        a_global.detach().cpu(),
        s_global.detach().cpu(),
        z_global.detach().cpu(),
        mask_global.detach().cpu(),
        d_out_global.detach().cpu(),
        out_serial.detach().cpu(),
        a_serial.grad.detach().cpu(),
        s_serial.grad.detach().cpu(),
        z_serial.grad.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# Helper: build feature placements for 1D CP
# ---------------------------------------------------------------------------

_selected_atom_keys = {
    "atom_pad_mask",
    "ref_pos",
    "ref_space_uid",
    "ref_charge",
    "ref_element",
    "ref_atom_name_chars",
    "atom_to_token",
    "atom_counts_per_token",
    "atom_resolved_mask",
    "plddt",
}
_selected_token_keys = {"token_pad_mask"}


def _build_feature_placements_1d(feat_keys):
    """Build per-feature placements for 2D mesh ``(dp, cp)`` from 1D CP registry.

    The 1D CP registry stores unbatched placements on the ``(cp,)`` sub-mesh.
    After batching, tensor dim 0 becomes the batch dimension (sharded on dp),
    so every ``Shard(d)`` in the CP placement must shift to ``Shard(d + 1)``
    to account for the new leading batch axis.  This mirrors the logic in
    :class:`CollateDTensor1D`.
    """
    placements = {}
    for key in feat_keys:
        if key in TRAINING_FEATURE_PLACEMENTS_1D:
            cp_placement = TRAINING_FEATURE_PLACEMENTS_1D[key]
            # Shift shard dims by 1 for the batch/dp dimension
            shifted = tuple(Shard(p.dim + 1) if isinstance(p, Shard) else p for p in cp_placement)
            placements[key] = (Shard(0),) + shifted
        else:
            # Default: replicate on cp, shard batch on dp
            placements[key] = (Shard(0), Replicate())
    return placements


def _make_noise_dtensors(noise_host_list, world_size, rank, device_mesh, device, dtype):
    """Convert host noise tensors to rank-local DTensors with ``(Shard(0), Shard(1))``."""
    placement = (Shard(0), Shard(1))
    dts = []
    for noise_host in noise_host_list:
        noise_global = noise_host.to(device=device, dtype=dtype)
        local_shard = noise_global.chunk(world_size, dim=1)[rank]
        dt = DTensor.from_local(
            local_shard,
            device_mesh=device_mesh,
            placements=placement,
            shape=noise_global.shape,
            stride=noise_global.stride(),
        )
        dts.append(dt)
    return dts


def _monkeypatch_deterministic_noise_1d(monkeypatch, init_noise_dt, step_noise_dts):
    """Monkeypatch diffusion_1d to use deterministic noise and no augmentation."""
    _orig = diffusion_1d_module._center_random_augmentation_1d

    def _centering_only(atom_coords, atom_mask, **kwargs):
        kwargs["augmentation"] = False
        kwargs["centering"] = True
        return _orig(atom_coords, atom_mask, **kwargs)

    _calls = []
    _sequence = [init_noise_dt] + list(step_noise_dts)

    def _fixed_randn(shape, device_mesh, placements, dtype=torch.float32, scale=1.0):
        idx = len(_calls)
        _calls.append(idx)
        noise_dt = _sequence[idx]
        if scale != 1.0:
            noise_dt = scalar_tensor_op(scale, noise_dt, ElementwiseOp.PROD)
        return noise_dt

    monkeypatch.setattr(diffusion_1d_module, "_center_random_augmentation_1d", _centering_only)
    monkeypatch.setattr(diffusion_1d_module, "create_distributed_randn", _fixed_randn)


# ---------------------------------------------------------------------------
# AtomDiffusion1D helpers parity test
# ---------------------------------------------------------------------------


def parallel_assert_atom_diffusion_helpers_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    atom_diffusion_state_dict,
    score_model_kwargs,
    sigma_global_host,
    c_skip_expected_host,
    c_out_expected_host,
    c_in_expected_host,
    c_noise_expected_host,
    loss_weight_expected_host,
    noise_dist_expected_host,
    sample_schedule_expected_host,
):
    """Parallel assertion for AtomDiffusion1D helper functions."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    serial = SerialAtomDiffusionV2(
        score_model_args=score_model_kwargs,
        coordinate_augmentation=False,
    )
    serial = serial.to(device=manager.device, dtype=dtype)
    serial.load_state_dict(atom_diffusion_state_dict)
    serial = serial.eval()

    device_mesh = manager.device_mesh
    cp_group = manager.group["cp"]

    module = AtomDiffusion1D(
        layer=serial,
        device_mesh=device_mesh,
        cp_group=cp_group,
    ).eval()

    # Distribute sigma: (Shard(0), Replicate()) = TIMES_PLACEMENTS_1D
    placements_scalar = (Shard(0), Replicate())
    sigma_dt = distribute_tensor(
        sigma_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_scalar,
    )

    # Test c_skip, c_out, c_in, c_noise, loss_weight
    for name, fn, expected_host in [
        ("c_skip", module.c_skip, c_skip_expected_host),
        ("c_out", module.c_out, c_out_expected_host),
        ("c_in", module.c_in, c_in_expected_host),
        ("c_noise", module.c_noise, c_noise_expected_host),
        ("loss_weight", module.loss_weight, loss_weight_expected_host),
    ]:
        result = fn(sigma_dt)
        expected = expected_host.to(device=manager.device, dtype=dtype)
        torch.testing.assert_close(result.full_tensor(), expected, msg=lambda m: f"{name}: {m}")

    # Test noise_distribution (stochastic — check shape, placement, and dtype)
    noise_dist = module.noise_distribution(sigma_dt.shape[0])
    assert (
        noise_dist.shape == sigma_dt.shape
    ), f"noise_distribution shape mismatch: {noise_dist.shape} vs {sigma_dt.shape}"
    assert noise_dist.placements == placements_scalar
    assert noise_dist.dtype == torch.float32, f"noise_distribution default dtype: {noise_dist.dtype} != float32"

    noise_dist_f64 = module.noise_distribution(sigma_dt.shape[0], dtype=torch.float64)
    assert noise_dist_f64.dtype == torch.float64, f"noise_distribution float64 dtype: {noise_dist_f64.dtype} != float64"

    # Test sample_schedule (deterministic, returns plain Tensor)
    schedule = module.sample_schedule(num_sampling_steps=5)
    expected_schedule = sample_schedule_expected_host.to(device=manager.device)
    torch.testing.assert_close(schedule, expected_schedule)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((2, 1), True, "cpu", "ENV"),
        ((2, 1), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=lambda x: f"dp:{x[0][0]}, cp:{x[0][1]}, device_type:{x[2]}",
)
def test_atom_diffusion_helpers_1d(setup_env):
    """Test AtomDiffusion1D scalar helper functions.

    Tests c_skip, c_out, c_in, c_noise, loss_weight, noise_distribution, sample_schedule.
    Uses dp=2, cp=1 since these functions don't involve CP atom sharding.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype = torch.float64

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    seed = 42
    seed_by_rank(0, seed=seed)

    B = 1 * grid_group_sizes["dp"]

    # Build a minimal serial AtomDiffusion (V2)
    score_model_kwargs = {
        "token_s": 4,
        "atom_s": 8,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "sigma_data": 16,
        "dim_fourier": 32,
        "atom_encoder_depth": 1,
        "atom_encoder_heads": 1,
        "token_transformer_depth": 1,
        "token_transformer_heads": 1,
        "atom_decoder_depth": 1,
        "atom_decoder_heads": 1,
        "conditioning_transition_layers": 1,
    }
    serial = SerialAtomDiffusionV2(
        score_model_args=score_model_kwargs,
        coordinate_augmentation=False,
    ).to(device=device_type, dtype=dtype)
    serial.eval()
    atom_diffusion_state_dict = serial.state_dict()

    # Generate test sigma values
    sigma = torch.tensor([0.5, 3.14, 16.0, 160.0], device=device_type, dtype=dtype)[:B]

    # Compute serial reference
    c_skip_ref = serial.c_skip(sigma)
    c_out_ref = serial.c_out(sigma)
    c_in_ref = serial.c_in(sigma)
    c_noise_ref = serial.c_noise(sigma)
    loss_weight_ref = serial.loss_weight(sigma)
    noise_dist_ref = serial.noise_distribution(B)
    sample_schedule_ref = serial.sample_schedule(num_sampling_steps=5)

    spawn_multiprocessing(
        parallel_assert_atom_diffusion_helpers_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        {k: v.cpu() for k, v in atom_diffusion_state_dict.items()},
        score_model_kwargs,
        sigma.cpu(),
        c_skip_ref.cpu(),
        c_out_ref.cpu(),
        c_in_ref.cpu(),
        c_noise_ref.cpu(),
        loss_weight_ref.cpu(),
        noise_dist_ref.cpu(),
        sample_schedule_ref.cpu(),
    )


# ---------------------------------------------------------------------------
# AtomDiffusion1D.sample() parity test
# ---------------------------------------------------------------------------


def parallel_assert_atom_diffusion_sample_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    multiplicity,
    num_sampling_steps,
    max_parallel_samples,
    alignment_reverse_diff,
    atom_diffusion_state_dict,
    conditioning_state_dict,
    score_model_kwargs,
    conditioning_kwargs,
    W,
    H,
    feats_global_host,
    s_inputs_global_host,
    s_trunk_global_host,
    z_trunk_global_host,
    rel_pos_enc_global_host,
    init_noise_global_host,
    step_noise_list_global_host,
    sample_coords_expected_global_host,
):
    """Parallel assertion for AtomDiffusion1D.sample()."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    serial_atom_diffusion = SerialAtomDiffusionV2(
        score_model_args=score_model_kwargs,
        coordinate_augmentation=False,
        alignment_reverse_diff=alignment_reverse_diff,
        num_sampling_steps=num_sampling_steps,
    )
    serial_atom_diffusion = serial_atom_diffusion.to(device=manager.device, dtype=dtype)
    serial_atom_diffusion.load_state_dict(atom_diffusion_state_dict)
    serial_atom_diffusion = serial_atom_diffusion.eval()

    device_mesh = manager.device_mesh
    cp_group = manager.group["cp"]

    module = AtomDiffusion1D(
        layer=serial_atom_diffusion,
        device_mesh=device_mesh,
        cp_group=cp_group,
    ).eval()

    # Distribute features via distribute_features (1D)
    feats_global = {
        k: v.to(device=manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in feats_global_host.items()
    }
    feature_placements = _build_feature_placements_1d(feats_global.keys())
    feats_dt = distribute_features(
        feats_global if rank == 0 else None,
        feature_placements,
        group=dist.group.WORLD,
        src_rank_global=0,
        device_mesh=device_mesh,
    )

    # Distribute token-level tensors
    placements_single = (Shard(0), Shard(1))
    placements_pair = (Shard(0), Shard(1))

    s_inputs_dt = distribute_tensor(
        s_inputs_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    )
    s_trunk_dt = distribute_tensor(
        s_trunk_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_single,
    )
    z_trunk_dt = distribute_tensor(
        z_trunk_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_pair,
    )
    rel_pos_enc_dt = distribute_tensor(
        rel_pos_enc_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_pair,
    )

    # Distribute noise tensors
    world_size = grid_group_sizes["dp"] * grid_group_sizes["cp"]
    all_noise_host = [init_noise_global_host] + list(step_noise_list_global_host)
    noise_dts = _make_noise_dtensors(
        all_noise_host,
        world_size,
        rank,
        device_mesh,
        manager.device,
        dtype,
    )
    _monkeypatch_deterministic_noise_1d(monkeypatch, noise_dts[0], noise_dts[1:])

    # Build conditioning
    serial_conditioning = SerialDiffusionConditioning(**conditioning_kwargs)
    serial_conditioning = serial_conditioning.to(device=manager.device, dtype=dtype)
    serial_conditioning.load_state_dict(conditioning_state_dict)
    serial_conditioning = serial_conditioning.eval()

    dtensor_conditioning = DiffusionConditioning1D(
        layer=serial_conditioning,
        device_mesh=device_mesh,
    ).eval()

    with torch.no_grad():
        q_dt, c_dt, enc_bias_dt, dec_bias_dt, trans_bias_dt = dtensor_conditioning(
            s_trunk=s_trunk_dt,
            z_trunk=z_trunk_dt,
            relative_position_encoding=rel_pos_enc_dt,
            feats=feats_dt,
        )

    network_condition_kwargs = {
        "s_inputs": s_inputs_dt,
        "s_trunk": s_trunk_dt,
        "feats": feats_dt,
        "diffusion_conditioning": {
            "q": q_dt,
            "c": c_dt,
            "atom_enc_bias": enc_bias_dt,
            "atom_dec_bias": dec_bias_dt,
            "token_trans_bias": trans_bias_dt,
        },
    }

    with torch.no_grad():
        out_dt = module.sample(
            atom_mask=feats_dt["atom_pad_mask"],
            multiplicity=multiplicity,
            max_parallel_samples=max_parallel_samples,
            **network_condition_kwargs,
        )

    # Compare: full_tensor on both sides. Tolerance is parametrize-conditional;
    # see comment block before @pytest.mark.parametrize on this test for the
    # full first-principles derivation.
    dt_full = out_dt["sample_atom_coords"].full_tensor()
    sample_expected = sample_coords_expected_global_host.to(device=manager.device, dtype=dtype)
    if alignment_reverse_diff:
        # weighted_rigid_align_1d engaged inside the sample loop. Upstream
        # cpu vs cuda model-forward fp divergence (cuBLAS/cuDNN vs MKL kernels)
        # delivers WRA inputs that already differ by max_abs ~1e+3; fp64 rigid
        # alignment suppresses this by ~2e+8x down to ~1e-5 — the working-
        # precision floor. Worst observed across the parametrize grid is
        # 1.19e-05 (mul=4 cpu); atol=5e-5 gives ~5x headroom over the worst cell.
        atol = 5e-5
        rtol = 1.3e-6  # torch.testing default fp64 rtol
    elif multiplicity == 4 and device_type == "cpu":
        # No weighted_rigid_align_1d, but the multiplicity=4 chunked path on
        # cpu accumulates a slightly larger sum-of-shards residual than fp64
        # default (2.94e-07 observed); ~5x bump over default fp64 atol=1e-7.
        atol = 5e-7
        rtol = 1.3e-6
    else:
        # Default fp64 tolerances suffice for all other cells.
        atol = None
        rtol = None
    torch.testing.assert_close(dt_full, sample_expected, atol=atol, rtol=rtol)

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.slow
# Tolerance derivation (first principles, per CLAUDE.md "Derive tolerances
# from first principles"):
#
# Mechanism: cpu vs cuda divergence is dominated by upstream model-forward
# floating-point differences (cuBLAS/cuDNN vs MKL GEMM, attention, layernorm,
# activation kernels) that accumulate through the denoiser and arrive at
# `weighted_rigid_align_1d` (`src/boltz/distributed/model/modules/diffusion_1d.py:1125`)
# at max_abs ~1e+3 on the (atom_coords_noisy, atom_coords_denoised) inputs.
# fp64 rigid alignment then suppresses this divergence by ~2e+8x down to a
# ~1e-5 residual on the rotated/translated output — the working-precision
# floor for fp64 SVD-based alignment of two structurally-close-but-not-bit-
# identical point clouds. When `alignment_reverse_diff=False`, WRA is not
# engaged and the only residual is per-platform fp accumulation on the
# multiplicity-chunked path (cpu mul=4 sees a ~2.94e-07 bump over fp64
# default; cuda mul=4 stays at default).
#
# WRA-internal contributions (verified non-mechanism by ULP trace at cp=2):
#   - cov_matrix all_reduce is bit-identical to a naive sum on both gloo
#     (cpu) and NCCL (cuda) — with cp=2 there is only one reduction order
#     (a+b), so the previously-hypothesised backend-dependent reduction-
#     order divergence does not apply here.
#   - post-all_reduce cov_matrix and SVD singular values `S` are bit-
#     identical across cp ranks of the same device — WRA does not introduce
#     intra-device divergence.
#   - cpu vs cuda divergence is already present in WRA's coord inputs and
#     in the per-rank pre-all_reduce cov_matrix (local einsum on platform-
#     divergent coords), confirming the divergence is upstream of WRA.
#
# Observed maxima (post-ac8d1c0c pack_atom_features fix; ~24000x reduction
# vs pre-fix):
#   align_rev=False mul=1 {cpu,cuda}: passes at default fp64 atol=1e-7
#   align_rev=False mul=4 cpu:        2.94e-07
#   align_rev=False mul=4 cuda:       passes at default
#   align_rev=True  mul=1 cpu:        2.96e-06
#   align_rev=True  mul=1 cuda:       1.10e-05
#   align_rev=True  mul=4 cpu:        1.19e-05
#   align_rev=True  mul=4 cuda:       1.10e-05
#
# Bound: atol=5e-5 for align_rev=True (5x headroom over worst-cell 1.19e-05);
# atol=5e-7 for align_rev=False mul=4 cpu (5x headroom over 2.94e-07); fp64
# default elsewhere. Ruled-out hypotheses (diagnostic 2026-05-15): fp32 SVD
# downcast, CUDA-kernel fp32 leak, fp64→fp32→fp64 load_state_dict round-trip,
# and (refuted by ULP trace) cp-axis reduction-order + SVD-driver divergence
# inside WRA.
@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cpu", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=["setup_env"],
    ids=lambda x: f"dp:{x[0][0]}, cp:{x[0][1]}, device_type:{x[2]}",
)
@pytest.mark.parametrize("multiplicity", [1, 4], ids=lambda x: f"mul:{x}")
@pytest.mark.parametrize("alignment_reverse_diff", [False, True], ids=lambda x: f"align_rev:{x}")
def test_atom_diffusion_sample_1d(setup_env, multiplicity, alignment_reverse_diff):
    """Test AtomDiffusion1D.sample() (inference) with 1D CP.

    Determinism is achieved via monkeypatching with pre-generated non-zero noise
    tensors. Serial uses mocked torch.randn; DTensor uses the same noise
    distributed via DTensor.from_local with (Shard(0), Shard(1)).
    Uses num_sampling_steps=2 to limit numerical error accumulation.
    Exercises max_parallel_samples chunking (max_parallel_samples=2 when multiplicity=4).
    The alignment_reverse_diff=True case exercises the weighted_rigid_align_1d
    callsite at diffusion_1d.py:1125 inside the sample-loop.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    dtype = torch.float64

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    seed = 42
    seed_by_rank(0, seed=seed)

    size_cp = grid_group_sizes["cp"]
    B = 1 * grid_group_sizes["dp"]
    W = 32
    H = 128
    val_init_min_max = (-0.5, 0.5)
    num_sampling_steps = 2
    max_parallel_samples = 2 if multiplicity > 2 else multiplicity

    n_atoms_per_token_min = 8
    n_atoms_per_token_max = 20
    N_tokens = 30 * size_cp
    N_atoms_raw = N_tokens * n_atoms_per_token_max
    N_atoms = ((N_atoms_raw + W - 1) // W) * W
    N_msa = 1

    atom_s = 8
    token_s = 4
    token_z = 4
    atom_z = 8
    atom_encoder_depth = 2
    atom_encoder_heads = 2
    token_transformer_depth = 2
    token_transformer_heads = 2
    atom_decoder_depth = 2
    atom_decoder_heads = 2
    conditioning_transition_layers = 1

    atom_feature_dim = 3 + 1 + boltz_const.num_elements + 4 * 64

    selected_keys = list(_selected_atom_keys | _selected_token_keys)
    feats = random_features(
        size_batch=B,
        n_tokens=N_tokens,
        n_atoms=N_atoms,
        n_msa=N_msa,
        atom_counts_per_token_range=(n_atoms_per_token_min, n_atoms_per_token_max),
        device=torch.device(device_type),
        float_value_range=val_init_min_max,
        selected_keys=selected_keys,
    )
    feats = {k: v.to(dtype=dtype) if v.dtype == torch.float64 else v for k, v in feats.items()}

    s_inputs_dim = token_s
    s_inputs = torch.empty((B, N_tokens, s_inputs_dim), device=device_type, dtype=dtype)
    s_trunk = torch.empty((B, N_tokens, token_s), device=device_type, dtype=dtype)
    z_trunk = torch.empty((B, N_tokens, N_tokens, token_z), device=device_type, dtype=dtype)
    rel_pos_enc = torch.empty((B, N_tokens, N_tokens, token_z), device=device_type, dtype=dtype)
    init_tensors_uniform([s_inputs, s_trunk, z_trunk, rel_pos_enc], low=val_init_min_max[0], high=val_init_min_max[1])

    # Build serial modules (V2 only for 1D)
    score_model_kwargs = {
        "token_s": token_s,
        "atom_s": atom_s,
        "atoms_per_window_queries": W,
        "atoms_per_window_keys": H,
        "sigma_data": 16,
        "dim_fourier": 32,
        "atom_encoder_depth": atom_encoder_depth,
        "atom_encoder_heads": atom_encoder_heads,
        "token_transformer_depth": token_transformer_depth,
        "token_transformer_heads": token_transformer_heads,
        "atom_decoder_depth": atom_decoder_depth,
        "atom_decoder_heads": atom_decoder_heads,
        "conditioning_transition_layers": conditioning_transition_layers,
    }
    conditioning_kwargs = {
        "token_s": token_s,
        "token_z": token_z,
        "atom_s": atom_s,
        "atom_z": atom_z,
        "atoms_per_window_queries": W,
        "atoms_per_window_keys": H,
        "atom_encoder_depth": atom_encoder_depth,
        "atom_encoder_heads": atom_encoder_heads,
        "token_transformer_depth": token_transformer_depth,
        "token_transformer_heads": token_transformer_heads,
        "atom_decoder_depth": atom_decoder_depth,
        "atom_decoder_heads": atom_decoder_heads,
        "atom_feature_dim": atom_feature_dim,
        "conditioning_transition_layers": conditioning_transition_layers,
    }
    serial_conditioning = SerialDiffusionConditioning(**conditioning_kwargs).to(device=device_type, dtype=dtype)
    serial_conditioning.train()
    init_module_params_uniform(serial_conditioning, low=val_init_min_max[0], high=val_init_min_max[1])
    serial_conditioning.apply(SetModuleInfValues())
    conditioning_state_dict = serial_conditioning.state_dict()

    serial_atom_diffusion = SerialAtomDiffusionV2(
        score_model_args=score_model_kwargs,
        coordinate_augmentation=False,
        alignment_reverse_diff=alignment_reverse_diff,
        num_sampling_steps=num_sampling_steps,
    ).to(device=device_type, dtype=dtype)
    serial_atom_diffusion.train()
    init_module_params_uniform(serial_atom_diffusion, low=val_init_min_max[0], high=val_init_min_max[1])
    serial_atom_diffusion.apply(SetModuleInfValues())
    serial_atom_diffusion.eval()
    atom_diffusion_state_dict = serial_atom_diffusion.state_dict()

    N_atoms_actual = feats["atom_pad_mask"].shape[1]
    _B_M = B * multiplicity

    # Pre-generate non-zero noise tensors for deterministic comparison
    init_noise = torch.empty((_B_M, N_atoms_actual, 3), device=device_type, dtype=dtype)
    step_noise_list = [
        torch.empty((_B_M, N_atoms_actual, 3), device=device_type, dtype=dtype) for _ in range(num_sampling_steps)
    ]
    init_tensors_uniform([init_noise, *step_noise_list], low=val_init_min_max[0], high=val_init_min_max[1])

    # Serial sample (with monkeypatched determinism)
    def _identity_compute_random_augmentation(multiplicity_arg, device=None, dtype=None):
        R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(_B_M, -1, -1)
        tr = torch.zeros(_B_M, 1, 3, device=device, dtype=dtype)
        return R, tr

    _serial_randn_calls = []
    _serial_randn_sequence = [init_noise] + step_noise_list

    def _fixed_randn(*args, **kwargs):
        idx = len(_serial_randn_calls)
        _serial_randn_calls.append(idx)
        return _serial_randn_sequence[idx].clone()

    _monkeypatch = pytest.MonkeyPatch()
    _monkeypatch.setattr(
        serial_diffusion_v2_module, "compute_random_augmentation", _identity_compute_random_augmentation
    )
    _monkeypatch.setattr(serial_diffusion_v2_module.torch, "randn", _fixed_randn)

    # V2: compute conditioning then sample
    q_cond, c_cond, to_keys, enc_bias, dec_bias, trans_bias = serial_conditioning(
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        relative_position_encoding=rel_pos_enc,
        feats={k: v.detach() for k, v in feats.items()},
    )
    with torch.no_grad():
        out_serial = serial_atom_diffusion.sample(
            atom_mask=feats["atom_pad_mask"],
            multiplicity=multiplicity,
            max_parallel_samples=max_parallel_samples,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            feats={k: v.clone() for k, v in feats.items()},
            diffusion_conditioning={
                "q": q_cond,
                "c": c_cond,
                "to_keys": to_keys,
                "atom_enc_bias": enc_bias,
                "atom_dec_bias": dec_bias,
                "token_trans_bias": trans_bias,
            },
        )

    _monkeypatch.undo()

    spawn_multiprocessing(
        parallel_assert_atom_diffusion_sample_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        multiplicity,
        num_sampling_steps,
        max_parallel_samples,
        alignment_reverse_diff,
        {k: v.cpu() for k, v in atom_diffusion_state_dict.items()},
        {k: v.cpu() for k, v in conditioning_state_dict.items()},
        score_model_kwargs,
        conditioning_kwargs,
        W,
        H,
        {k: v.cpu() for k, v in feats.items()},
        s_inputs.cpu(),
        s_trunk.cpu(),
        z_trunk.cpu(),
        rel_pos_enc.cpu(),
        init_noise.cpu(),
        [n.cpu() for n in step_noise_list],
        out_serial["sample_atom_coords"].cpu(),
    )


# ---------------------------------------------------------------------------
# AtomDiffusion1D.compute_loss() parity test
# ---------------------------------------------------------------------------


def parallel_assert_atom_diffusion_compute_loss_1d(
    rank,
    grid_group_sizes,
    device_type,
    backend,
    env_per_rank,
    dtype,
    multiplicity,
    add_smooth_lddt_loss,
    nucleotide_loss_weight,
    ligand_loss_weight,
    filter_by_plddt,
    use_triton_kernel,
    feats_host,
    atom_diffusion_state_dict,
    score_model_kwargs,
    denoised_atom_coords_global_host,
    aligned_true_atom_coords_global_host,
    sigma_global_host,
    expected_total_loss_global_host,
    expected_mse_loss_global_host,
    expected_smooth_lddt_loss_global_host,
    expected_denoised_atom_coords_grad_global_host,
):
    """Parallel assertion for AtomDiffusion1D.compute_loss()."""
    monkeypatch = pytest.MonkeyPatch()
    if env_per_rank is not None:
        for var_name, value in env_per_rank.items():
            if value == "<INPUT_RANK>":
                monkeypatch.setenv(var_name, f"{rank}")
                continue
            monkeypatch.setenv(var_name, value)

    DistributedManager.initialize(grid_group_sizes, device_type=device_type, backend=backend)
    manager = DistributedManager()

    serial_atom_diffusion = SerialAtomDiffusionV2(
        score_model_args=score_model_kwargs,
    )
    serial_atom_diffusion = serial_atom_diffusion.to(device=manager.device, dtype=dtype)
    serial_atom_diffusion.load_state_dict(atom_diffusion_state_dict)
    serial_atom_diffusion = serial_atom_diffusion.train()

    device_mesh = manager.device_mesh
    cp_group = manager.group["cp"]

    module = AtomDiffusion1D(
        layer=serial_atom_diffusion,
        device_mesh=device_mesh,
        cp_group=cp_group,
    ).train()

    # Distribute features
    feats_global = {
        k: v.to(device=manager.device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
        for k, v in feats_host.items()
        if k != "coords"
    }
    feature_placements = _build_feature_placements_1d(feats_global.keys())
    feats_dt = distribute_features(
        feats_global if rank == 0 else None,
        feature_placements,
        group=dist.group.WORLD,
        src_rank_global=0,
        device_mesh=device_mesh,
    )

    # Distribute compute_loss-specific tensors via distribute_tensor
    placements_atom = (Shard(0), Shard(1))
    placements_scalar = (Shard(0), Replicate())

    denoised_atom_coords_dtensor = distribute_tensor(
        denoised_atom_coords_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_atom,
    ).requires_grad_(True)

    aligned_true_atom_coords_dtensor = distribute_tensor(
        aligned_true_atom_coords_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_atom,
    )

    sigma_dtensor = distribute_tensor(
        sigma_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_scalar,
    )

    d_r_update_expected_dtensor = distribute_tensor(
        expected_denoised_atom_coords_grad_global_host.to(device=manager.device, dtype=dtype),
        device_mesh,
        placements_atom,
    )

    input_dict = {
        "denoised_atom_coords": denoised_atom_coords_dtensor,
        "sigmas": sigma_dtensor,
        "aligned_true_atom_coords": aligned_true_atom_coords_dtensor,
    }
    input_dict_clone = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in input_dict.items()}

    output_dict = module.compute_loss(
        feats=feats_dt,
        out_dict=input_dict,
        add_smooth_lddt_loss=add_smooth_lddt_loss,
        nucleotide_loss_weight=nucleotide_loss_weight,
        ligand_loss_weight=ligand_loss_weight,
        multiplicity=multiplicity,
        filter_by_plddt=filter_by_plddt,
        use_triton_kernel=use_triton_kernel,
    )

    # Ensure input dict is not modified by compute_loss
    for k in input_dict.keys():
        assert_tensors_identical(
            input_dict[k],
            input_dict_clone[k],
            check_grad=False,
            check_grad_fn=False,
            check_storage_offset=True,
            check_storage_pointer=False,
        )

    total_loss_dtensor = output_dict["loss"]
    mse_loss_dtensor = output_dict["loss_breakdown"]["mse_loss"]
    smooth_lddt_loss_dtensor = output_dict["loss_breakdown"]["smooth_lddt_loss"]

    # 1D losses should be Replicated on both mesh dims
    for name, loss_dt in [
        ("total_loss", total_loss_dtensor),
        ("mse_loss", mse_loss_dtensor),
        ("smooth_lddt_loss", smooth_lddt_loss_dtensor),
    ]:
        assert all(
            isinstance(p, Replicate) for p in loss_dt.placements
        ), f"{name} placements {loss_dt.placements} are not fully replicated"

    total_loss = total_loss_dtensor.full_tensor().cpu()
    mse_loss = mse_loss_dtensor.full_tensor().cpu()
    smooth_lddt_loss = smooth_lddt_loss_dtensor.full_tensor().cpu()

    assert not (mse_loss == 0.0).all(), "mse_loss should not be 0"
    if add_smooth_lddt_loss:
        assert not (smooth_lddt_loss == 0.0).all(), "smooth_lddt_loss should not be 0"
    assert not (total_loss == 0.0).all(), "total_loss should not be 0"
    torch.testing.assert_close(mse_loss, expected_mse_loss_global_host)
    torch.testing.assert_close(smooth_lddt_loss, expected_smooth_lddt_loss_global_host)
    torch.testing.assert_close(total_loss, expected_total_loss_global_host)

    total_loss_dtensor_clone = total_loss_dtensor.detach().clone().requires_grad_(total_loss_dtensor.requires_grad)
    mse_loss_dtensor_clone = mse_loss_dtensor.detach().clone().requires_grad_(mse_loss_dtensor.requires_grad)
    smooth_lddt_loss_dtensor_clone = (
        smooth_lddt_loss_dtensor.detach().clone().requires_grad_(smooth_lddt_loss_dtensor.requires_grad)
    )

    total_loss_dtensor.backward()

    assert_tensors_identical(
        total_loss_dtensor,
        total_loss_dtensor_clone,
        check_grad=False,
        check_grad_fn=False,
        check_storage_offset=True,
        check_storage_pointer=False,
    )
    assert_tensors_identical(
        mse_loss_dtensor,
        mse_loss_dtensor_clone,
        check_grad=False,
        check_grad_fn=False,
        check_storage_offset=True,
        check_storage_pointer=False,
    )
    assert_tensors_identical(
        smooth_lddt_loss_dtensor,
        smooth_lddt_loss_dtensor_clone,
        check_grad=False,
        check_grad_fn=False,
        check_storage_offset=True,
        check_storage_pointer=False,
    )

    torch.testing.assert_close(
        denoised_atom_coords_dtensor.grad.full_tensor(), d_r_update_expected_dtensor.full_tensor()
    )

    DistributedManager.cleanup()
    monkeypatch.undo()


@pytest.mark.parametrize(
    "setup_env",
    [
        ((1, 2), True, "cpu", "ENV"),
        ((1, 2), True, "cuda", "ENV"),
    ],
    indirect=("setup_env",),
    ids=lambda x: f"dp={x[0][0]}, cp={x[0][1]}, device_type={x[2]}",
)
@pytest.mark.parametrize(
    "loss_config",
    [
        (1, True, False, 0.0),
        (4, True, True, 0.5),
        (1, True, False, 0.0),
        (4, True, True, 0.0),
    ],
    ids=lambda x: f"mul={x[0]}, lddt={x[1]}, triton={x[2]}, plddt={x[3]:.1f}",
)
def test_atom_diffusion_compute_loss_1d(
    setup_env,
    loss_config,
    nucleotide_loss_weight: float = 5.0,
    ligand_loss_weight: float = 10.0,
    dtype: torch.dtype = torch.float32,
):
    """Test AtomDiffusion1D.compute_loss() with 1D CP.

    Compares DTensor AtomDiffusion1D.compute_loss() against serial V2
    for forward and backward.
    """
    grid_group_sizes, world_size, device_type, backend, _, env_per_rank = setup_env
    multiplicity, add_smooth_lddt_loss, use_triton_kernel, filter_by_plddt = loss_config

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("skip cuda test because torch.cuda.is_available == False")
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"skip cuda test because torch.cuda.device_count() < {world_size}")

    if not add_smooth_lddt_loss and use_triton_kernel:
        pytest.skip("use_triton_kernel requires add_smooth_lddt_loss=True")

    if use_triton_kernel and device_type == "cpu":
        pytest.skip("triton requires CUDA")

    seed = 42
    seed_by_rank(0, seed=seed)

    size_cp = grid_group_sizes["cp"]
    B = 1 * grid_group_sizes["dp"]
    W = 32
    H = 128
    val_init_min_max = (-1.0, 1.0)

    n_atoms_per_token_min = 8
    n_atoms_per_token_max = 20
    N_tokens = 10 * size_cp
    N_atoms_raw = N_tokens * n_atoms_per_token_max
    N_atoms = ((N_atoms_raw + W - 1) // W) * W
    N_msa = 1

    atom_s = 8
    token_s = 4
    token_z = 4

    atom_encoder_depth = 2
    atom_encoder_heads = 2
    token_transformer_depth = 2
    token_transformer_heads = 2
    atom_decoder_depth = 2
    atom_decoder_heads = 2
    conditioning_transition_layers = 1

    compute_loss_selected_keys = {
        "atom_resolved_mask",
        "mol_type",
        "atom_to_token",
        "atom_counts_per_token",
        "plddt",
    }
    selected_keys = list(_selected_atom_keys | _selected_token_keys | compute_loss_selected_keys)
    feats = random_features(
        size_batch=B,
        n_tokens=N_tokens,
        n_atoms=N_atoms,
        n_msa=N_msa,
        atom_counts_per_token_range=(n_atoms_per_token_min, n_atoms_per_token_max),
        device=torch.device(device_type),
        float_value_range=val_init_min_max,
        selected_keys=selected_keys,
    )
    feats = {k: v.to(dtype=dtype) if v.dtype == torch.float64 else v for k, v in feats.items()}

    # V2 s_inputs has token_s dim
    s_inputs = torch.empty((B, N_tokens, token_s), device=device_type, dtype=dtype, requires_grad=True)
    s_trunk = torch.empty((B, N_tokens, token_s), device=device_type, dtype=dtype, requires_grad=True)
    z_trunk = torch.empty((B, N_tokens, N_tokens, token_z), device=device_type, dtype=dtype)
    rel_pos_enc = torch.empty((B, N_tokens, N_tokens, token_z), device=device_type, dtype=dtype)
    init_tensors_uniform([s_inputs, s_trunk, z_trunk, rel_pos_enc], low=val_init_min_max[0], high=val_init_min_max[1])

    # V2 serial model
    score_model_kwargs = {
        "token_s": token_s,
        "atom_s": atom_s,
        "atoms_per_window_queries": W,
        "atoms_per_window_keys": H,
        "sigma_data": 16,
        "dim_fourier": 32,
        "atom_encoder_depth": atom_encoder_depth,
        "atom_encoder_heads": atom_encoder_heads,
        "token_transformer_depth": token_transformer_depth,
        "token_transformer_heads": token_transformer_heads,
        "atom_decoder_depth": atom_decoder_depth,
        "atom_decoder_heads": atom_decoder_heads,
        "conditioning_transition_layers": conditioning_transition_layers,
    }

    serial_model = SerialAtomDiffusionV2(
        score_model_args=score_model_kwargs,
        coordinate_augmentation=False,
    ).to(device=device_type, dtype=dtype)
    init_module_params_uniform(serial_model, low=-0.1, high=0.1)
    serial_model.apply(SetModuleInfValues())
    module_state_dict = serial_model.state_dict()
    serial_model = serial_model.to(device=device_type, dtype=dtype)

    N_atoms_actual = feats["atom_pad_mask"].shape[1]
    denoised_atom_coords = torch.empty(
        (B * multiplicity, N_atoms_actual, 3), device=device_type, dtype=dtype, requires_grad=True
    )
    init_tensors_uniform([denoised_atom_coords], low=val_init_min_max[0], high=val_init_min_max[1])
    aligned_true_atom_coords = torch.empty_like(denoised_atom_coords)
    init_tensors_uniform([aligned_true_atom_coords], low=val_init_min_max[0], high=val_init_min_max[1])
    sigma = serial_model.noise_distribution(B * multiplicity).to(device=device_type, dtype=dtype)
    denoised_atom_coords.requires_grad = True

    input_dict = {
        "denoised_atom_coords": denoised_atom_coords,
        "sigmas": sigma,
        "aligned_true_atom_coords": aligned_true_atom_coords,
    }
    feats["coords"] = aligned_true_atom_coords

    extra_kwargs = {}
    if filter_by_plddt > 0:
        extra_kwargs["filter_by_plddt"] = filter_by_plddt

    output_dict = serial_model.compute_loss(
        feats=feats,
        out_dict=input_dict,
        add_smooth_lddt_loss=add_smooth_lddt_loss,
        nucleotide_loss_weight=nucleotide_loss_weight,
        ligand_loss_weight=ligand_loss_weight,
        multiplicity=multiplicity,
        **extra_kwargs,
    )
    output_dict["loss"].backward()

    feats_host = {k: v.detach().to(device="cpu", copy=True) if torch.is_tensor(v) else v for k, v in feats.items()}

    spawn_multiprocessing(
        parallel_assert_atom_diffusion_compute_loss_1d,
        world_size,
        grid_group_sizes,
        device_type,
        backend,
        env_per_rank,
        dtype,
        multiplicity,
        add_smooth_lddt_loss,
        nucleotide_loss_weight,
        ligand_loss_weight,
        filter_by_plddt,
        use_triton_kernel,
        feats_host,
        {k: v.cpu() for k, v in module_state_dict.items()},
        score_model_kwargs,
        denoised_atom_coords.detach().clone().cpu(),
        aligned_true_atom_coords.detach().clone().cpu(),
        sigma.detach().clone().cpu(),
        output_dict["loss"].detach().clone().cpu(),
        output_dict["loss_breakdown"]["mse_loss"].detach().clone().cpu(),
        output_dict["loss_breakdown"]["smooth_lddt_loss"].detach().clone().cpu(),
        denoised_atom_coords.grad.detach().clone().cpu(),
    )
