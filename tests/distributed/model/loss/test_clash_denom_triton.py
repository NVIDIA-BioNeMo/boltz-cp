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

"""Tests for the clash_denom_grouped Triton kernel.

Tests:
  - test_clash_denom_grouped_correctness: parametrized correctness test
    comparing Triton kernel output against a PyTorch reference across
    shapes, dtypes, and chain/entity configurations.
  - test_clash_denom_grouped_validation: input validation checks.
  - test_no_register_spilling: PTX-level check that the kernel does not
    spill registers.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import torch

from boltz.distributed.model.loss.triton.clash_denom import clash_denom_grouped

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def clash_denom_reference(
    coords: torch.Tensor,
    mask: torch.Tensor,
    chain_id: torch.Tensor,
    entity_id: torch.Tensor,
    multiplicity: int,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch O(N^2) reference for clash_denom_grouped."""
    B_mul, N, _ = coords.shape
    compute_dtype = torch.promote_types(coords.dtype, torch.float32)
    coords_c = coords.to(compute_dtype)

    mask_exp = mask.repeat_interleave(multiplicity, dim=0).to(compute_dtype)
    chain_exp = chain_id.repeat_interleave(multiplicity, dim=0)
    entity_exp = entity_id.repeat_interleave(multiplicity, dim=0)

    dists = torch.cdist(coords_c, coords_c)

    valid = mask_exp[:, :, None] * mask_exp[:, None, :]
    diag = torch.eye(N, device=coords.device, dtype=compute_dtype)[None]
    active = valid * (1.0 - diag) * (dists < cutoff).to(compute_dtype)

    same_chain = (chain_exp[:, :, None] == chain_exp[:, None, :]).to(compute_dtype)
    same_entity = (entity_exp[:, :, None] == entity_exp[:, None, :]).to(compute_dtype)

    total_denom = active.sum(dim=-1)
    chain_denom = (active * same_chain).sum(dim=-1)
    entity_denom = (active * same_entity).sum(dim=-1)

    return total_denom, chain_denom, entity_denom


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("B, multiplicity", [(1, 1), (2, 4)], ids=["B1M1", "B2M4"])
@pytest.mark.parametrize("N", [32, 100], ids=lambda x: f"N{x}")
@pytest.mark.parametrize(
    "num_chains, num_entities",
    [(2, 1), (4, 2), (5, 5)],
    ids=["2ch_1ent", "4ch_2ent", "5ch_5ent"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
    ids=["f32", "f64"],
)
def test_clash_denom_grouped_correctness(B, multiplicity, N, num_chains, num_entities, dtype):
    device = torch.device("cuda")
    B_mul = B * multiplicity
    rng = torch.Generator(device=device).manual_seed(42)

    coords = torch.randn(B_mul, N, 3, device=device, dtype=dtype, generator=rng)
    mask = torch.randint(0, 2, (B, N), device=device, generator=rng).to(dtype)

    chain_id = torch.randint(0, num_chains, (B, N), device=device, generator=rng)
    chain_to_entity = torch.arange(num_chains, device=device) % num_entities
    entity_id = chain_to_entity[chain_id]

    cutoff = 1.5

    total_tri, chain_tri, entity_tri = clash_denom_grouped(
        coords=coords,
        mask=mask,
        chain_id=chain_id,
        entity_id=entity_id,
        multiplicity=multiplicity,
        cutoff=cutoff,
    )

    total_ref, chain_ref, entity_ref = clash_denom_reference(
        coords=coords,
        mask=mask,
        chain_id=chain_id,
        entity_id=entity_id,
        multiplicity=multiplicity,
        cutoff=cutoff,
    )

    torch.testing.assert_close(total_tri, total_ref, msg="total_denom mismatch")
    torch.testing.assert_close(chain_tri, chain_ref, msg="chain_denom mismatch")
    torch.testing.assert_close(entity_tri, entity_ref, msg="entity_denom mismatch")

    assert total_ref.sum() > 0, "Reference total_denom is all zeros — test geometry is degenerate"


# ---------------------------------------------------------------------------
# Input validation test
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        ({}, None),
        ({"coord_dim": 2}, "Coordinate dimension must be 3"),
        ({"bad_bmul": True}, "must be divisible by multiplicity"),
    ],
    ids=["valid_inputs", "coord_dim_not_3", "bmul_not_divisible"],
)
def validation_case(request):
    return request.param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_clash_denom_grouped_validation(validation_case):
    modifications, expected_error = validation_case
    device = torch.device("cuda")

    B, multiplicity, N = 2, 4, 32
    B_mul = B * multiplicity
    coord_dim = modifications.get("coord_dim", 3)
    if modifications.get("bad_bmul"):
        B_mul = B * multiplicity + 1

    inputs = {
        "coords": torch.randn(B_mul, N, coord_dim, device=device),
        "mask": torch.ones(B, N, device=device),
        "chain_id": torch.zeros(B, N, device=device, dtype=torch.long),
        "entity_id": torch.zeros(B, N, device=device, dtype=torch.long),
        "multiplicity": multiplicity,
        "cutoff": 1.5,
    }

    if expected_error is None:
        total, chain, entity = clash_denom_grouped(**inputs)
        assert total.shape == (B_mul, N)
        assert chain.shape == (B_mul, N)
        assert entity.shape == (B_mul, N)
    else:
        with pytest.raises(ValueError, match=expected_error):
            clash_denom_grouped(**inputs)


# ---------------------------------------------------------------------------
# Register spilling test
# ---------------------------------------------------------------------------


def assert_no_register_spilling(path_to_ptx_file: Path):
    ptx_code = path_to_ptx_file.read_text()

    sm_arch_match = re.search(r"\.target (sm_\w+)", ptx_code)
    if not sm_arch_match:
        raise RuntimeError(f"No .target directive found in {path_to_ptx_file}")
    sm_arch = sm_arch_match.group(1)

    ptxas_path = os.environ.get("TRITON_PTXAS_PATH", "ptxas")
    cmd = [ptxas_path, "-v", f"--gpu-name={sm_arch}", str(path_to_ptx_file)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stderr
        if "0 bytes spill stores, 0 bytes spill loads" not in output:
            raise RuntimeError(f"Register spilling detected in {path_to_ptx_file}:\n{output}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ptxas failed with error:\n{e.stderr}") from e


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_register_spilling(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_KERNEL_DUMP", "1")
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_PTXAS_PATH", os.environ.get("TRITON_PTXAS_PATH", "ptxas"))

    device = torch.device("cuda")
    # mul=16 must differ from correctness-test values (1, 4) so Triton
    # compiles a fresh kernel specialisation and dumps PTX to tmp_path.
    B, mul, N = 1, 16, 100
    B_mul = B * mul

    coords = torch.randn(B_mul, N, 3, device=device)
    mask = torch.ones(B, N, device=device)
    chain_id = torch.randint(0, 3, (B, N), device=device)
    entity_id = torch.randint(0, 2, (B, N), device=device)

    clash_denom_grouped(
        coords=coords,
        mask=mask,
        chain_id=chain_id,
        entity_id=entity_id,
        multiplicity=mul,
        cutoff=1.5,
    )

    ptx_files = list(tmp_path.glob("**/_clash_denom_grouped_kernel.ptx"))
    if not ptx_files:
        raise RuntimeError(f"No PTX file found in {tmp_path}")

    assert_no_register_spilling(ptx_files[0])
