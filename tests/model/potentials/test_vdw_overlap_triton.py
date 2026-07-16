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

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from boltz.data import const
from boltz.model.potentials.potentials import VDWOverlapPotential
from boltz.model.potentials.triton.vdw_overlap import (
    _vdw_overlap_kernel,
    vdw_overlap_energy,
    vdw_overlap_gradient,
)
from boltz.testing.utils import random_features


def _build_vdw_feats(coords, atom_pad_mask, atom_chain_id, connected_chain_matrix, ref_element):
    """Build feats dict for VDWOverlapPotential from raw test components.

    Uses identity atom_to_token (N_tokens == N_atoms), so asym_id ==
    atom_chain_id after the bmm derivation in _extract_vdw_features.
    """
    N = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    atom_to_token = torch.eye(N, dtype=dtype, device=device)
    asym_id = atom_chain_id.to(device=device, dtype=torch.long)

    # Extract connected chain edges from upper triangle (code symmetrizes both directions)
    off_diag = connected_chain_matrix.clone()
    off_diag.fill_diagonal_(False)
    upper = torch.triu(off_diag, diagonal=1)
    src, dst = upper.nonzero(as_tuple=True)
    connected_chain_index = torch.stack([src, dst], dim=0).to(device)

    return {
        "atom_to_token": atom_to_token.unsqueeze(0),
        "asym_id": asym_id.unsqueeze(0),
        "atom_pad_mask": atom_pad_mask.unsqueeze(0).to(device),
        "ref_element": ref_element.unsqueeze(0),
        "connected_chain_index": connected_chain_index.unsqueeze(0),
    }


def _vdw_reference(feats, parameters, coords_batched):
    """Compute VDW energy and gradient via the generic Potential pipeline (PyTorch).

    Monkeypatches ``_HAS_TRITON_VDW`` to ``False`` so that
    ``VDWOverlapPotential`` falls back to the base-class PyTorch path
    (O(N^2) pair indices via ``torch.triu_indices``) even on CUDA.
    This is the serial ground truth.

    Returns:
        energy: (B,) tensor
        gradient: (B, N, 3) tensor
    """
    import boltz.model.potentials.potentials as _pot_mod

    original = _pot_mod._HAS_TRITON_VDW
    _pot_mod._HAS_TRITON_VDW = False
    try:
        potential = VDWOverlapPotential()
        energy = potential.compute(coords_batched, feats, parameters)
        gradient = potential.compute_gradient(coords_batched, feats, parameters)
    finally:
        _pot_mod._HAS_TRITON_VDW = original
    return energy, gradient


def _extract_triton_args(feats, parameters):
    """Extract raw Triton kernel args from feats via VDWOverlapPotential._extract_vdw_features."""
    potential = VDWOverlapPotential()
    return potential._extract_vdw_features(feats, parameters)


def make_vdw_test_inputs(n_atoms, num_chains, dtype, device, size_batch=1, seed=42):
    """Generate random VDW overlap test inputs as (feats, parameters, coords_batched).

    Uses ``random_features`` with multi-atom tokens (1–4 atoms per token,
    ``n_atoms = 4 * n_tokens``) for ``atom_to_token`` and ``ref_element``,
    then overrides chain structure (``asym_id``, ``connected_chain_index``)
    and padding (``atom_pad_mask``) to guarantee test validity:

    - At least one single-atom chain (single-ion edge case).
    - At least one disconnected inter-chain pair (non-vacuous overlap).

    Features are generated with ``size_batch=1`` then ``.expand()``-ed so all
    batch elements share identical masks, radii, and chain IDs — matching the
    Triton kernel contract that only coords vary across the batch dimension.
    Coords are independently sampled per batch element.

    Returns:
        feats: dict compatible with VDWOverlapPotential (batch dim = size_batch)
        parameters: {"buffer": float}
        coords_batched: (size_batch, n_atoms, 3) coords tensor
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    n_tokens = n_atoms // 4

    # Generate with size_batch=1 and expand so all batch elements share
    # identical atom_to_token and ref_element (the CPU reference path always
    # extracts features from batch_idx=0).
    feats = random_features(
        size_batch=1,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        n_msa=1,
        atom_counts_per_token_range=(1, 4),
        device=torch.device("cpu"),
        float_value_range=(-1.0, 1.0),
        selected_keys=["atom_to_token", "ref_element"],
        rng=rng,
    )
    feats = {k: v.expand(size_batch, *v.shape[1:]) for k, v in feats.items()}

    # Override asym_id (per-token) for controlled chain structure.
    # Atom-level chain IDs are derived via bmm(atom_to_token, asym_id).
    token_chain_id = torch.randint(0, num_chains, (n_tokens,), generator=rng)
    # Force last token into a dedicated chain (single-atom chain guarantee)
    token_chain_id[-1] = num_chains - 1
    token_chain_id[:-1][token_chain_id[:-1] == num_chains - 1] = 0
    # Ensure chains 0 and 1 each have ≥2 tokens (multi-atom, not single-ion)
    if num_chains >= 2 and n_tokens >= 4:
        token_chain_id[0] = 0
        token_chain_id[1] = 0
        token_chain_id[2] = 1
        token_chain_id[3] = 1
    feats["asym_id"] = token_chain_id.unsqueeze(0).expand(size_batch, -1)

    # Override atom_pad_mask: ~80% real atoms
    atom_pad_mask = torch.rand(n_atoms, generator=rng) < 0.8
    # Ensure the dedicated chain's atom is valid (single-ion coverage)
    atom_pad_mask[-1] = True
    feats["atom_pad_mask"] = atom_pad_mask.unsqueeze(0).expand(size_batch, -1)

    # Build connected_chain_index: random ~20% connections, guarantee ≥1,
    # but never connect chains 0 and 1 (guarantees a disconnected multi-atom pair).
    src_all, dst_all = torch.triu_indices(num_chains, num_chains, offset=1)
    keep = torch.rand(src_all.shape[0], generator=rng) < 0.2
    if not keep.any():
        keep[0] = True
    # Remove 0↔1 connection to ensure non-trivial overlap
    keep &= ~((src_all == 0) & (dst_all == 1))
    if not keep.any():
        # Re-add a connection that doesn't involve 0↔1
        for i in range(len(src_all)):
            if not (src_all[i] == 0 and dst_all[i] == 1):
                keep[i] = True
                break
    cci = torch.stack([src_all[keep], dst_all[keep]], dim=0)
    feats["connected_chain_index"] = cci.unsqueeze(0).expand(size_batch, -1, -1)

    # Small scale coords so atoms overlap frequently
    coords = torch.randn(size_batch, n_atoms, 3, generator=rng)

    # Move to target device and dtype
    feats = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in feats.items()}
    coords = coords.to(device=device, dtype=dtype)

    return feats, {"buffer": 0.225}, coords


# ---------------------------------------------------------------------------
# Parity tests: energy and gradient together
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=lambda x: f"dtype:{x}")
@pytest.mark.parametrize("N_atoms", [32, 100, 256], ids=lambda x: f"N:{x}")
@pytest.mark.parametrize("size_batch", [1, 2], ids=lambda x: f"B:{x}")
def test_parity(dtype, N_atoms, size_batch):
    """Triton energy and gradient match VDWOverlapPotential reference."""
    device = torch.device("cuda")
    n_tokens = N_atoms // 4
    num_chains = min(5, max(3, n_tokens // 4))
    feats, parameters, coords_batched = make_vdw_test_inputs(N_atoms, num_chains, dtype, device, size_batch=size_batch)

    ref_energy, ref_gradient = _vdw_reference(feats, parameters, coords_batched)
    assert ref_energy.shape == (size_batch,)
    assert (ref_energy > 0).all(), "Reference energy must not be trivially zero"
    assert ref_gradient.shape == (size_batch, N_atoms, 3)
    for b in range(size_batch):
        assert not torch.all(ref_gradient[b] == 0.0), f"Reference gradient must not be trivially zero (batch {b})"

    triton_args = _extract_triton_args(feats, parameters)
    triton_energy = vdw_overlap_energy(coords_batched, *triton_args)
    triton_gradient = vdw_overlap_gradient(coords_batched, *triton_args)

    assert triton_energy.shape == (size_batch,)
    assert (triton_energy > 0).all(), "Triton energy must not be trivially zero"
    assert triton_gradient.shape == (size_batch, N_atoms, 3)
    for b in range(size_batch):
        assert not torch.all(triton_gradient[b] == 0.0), f"Triton gradient must not be trivially zero (batch {b})"

    torch.testing.assert_close(triton_energy, ref_energy)
    torch.testing.assert_close(triton_gradient, ref_gradient)


# ---------------------------------------------------------------------------
# Geometric edge cases: parametrized zero/non-zero tests
# ---------------------------------------------------------------------------


def _build_edge_case_inputs(case_name, device, dtype):
    """Build (feats, parameters, coords_batched) for a named edge case.

    Returns:
        feats: dict with batch dim = 1
        parameters: {"buffer": float}
        coords_batched: (1, N, 3) tensor
    """
    N = 64
    num_chains = 4
    buffer = 0.225
    element_idx = 6  # carbon

    coords = torch.randn(N, 3, device=device, dtype=dtype)
    ref_element = torch.zeros(N, const.num_elements, dtype=dtype, device=device)
    ref_element[:, element_idx] = 1.0
    atom_chain_id = torch.randint(0, num_chains, (N,), device=device)

    if case_name == "all_padded":
        atom_pad_mask = torch.zeros(N, device=device, dtype=torch.bool)
        connected_chain_matrix = torch.eye(num_chains, device=device, dtype=torch.bool)

    elif case_name == "single_atom":
        N = 1
        coords = torch.randn(1, 3, device=device, dtype=dtype)
        ref_element = torch.zeros(1, const.num_elements, dtype=dtype, device=device)
        ref_element[0, element_idx] = 1.0
        atom_pad_mask = torch.ones(1, device=device, dtype=torch.bool)
        atom_chain_id = torch.zeros(1, device=device, dtype=torch.long)
        num_chains = 1
        connected_chain_matrix = torch.eye(1, device=device, dtype=torch.bool)

    elif case_name == "all_ions":
        atom_pad_mask = torch.ones(N, device=device, dtype=torch.bool)
        atom_chain_id = torch.arange(N, device=device)
        num_chains = N
        connected_chain_matrix = torch.eye(num_chains, device=device, dtype=torch.bool)

    elif case_name == "all_connected":
        atom_pad_mask = torch.ones(N, device=device, dtype=torch.bool)
        connected_chain_matrix = torch.ones(num_chains, num_chains, device=device, dtype=torch.bool)

    elif case_name == "non_overlapping":
        N = 10
        coords = torch.zeros(N, 3, device=device, dtype=dtype)
        coords[:, 0] = torch.arange(N, device=device, dtype=dtype) * 100.0
        ref_element = torch.zeros(N, const.num_elements, dtype=dtype, device=device)
        ref_element[:, element_idx] = 1.0
        atom_pad_mask = torch.ones(N, device=device, dtype=torch.bool)
        atom_chain_id = torch.zeros(N, device=device, dtype=torch.long)
        atom_chain_id[N // 2 :] = 1
        num_chains = 2
        connected_chain_matrix = torch.eye(2, device=device, dtype=torch.bool)

    elif case_name == "overlapping":
        N = 10
        coords = torch.randn(N, 3, device=device, dtype=dtype) * 0.01
        ref_element = torch.zeros(N, const.num_elements, dtype=dtype, device=device)
        ref_element[:, element_idx] = 1.0
        atom_pad_mask = torch.ones(N, device=device, dtype=torch.bool)
        atom_chain_id = torch.zeros(N, device=device, dtype=torch.long)
        atom_chain_id[N // 2 :] = 1
        num_chains = 2
        connected_chain_matrix = torch.eye(2, device=device, dtype=torch.bool)

    else:
        raise ValueError(f"Unknown edge case: {case_name}")

    feats = _build_vdw_feats(coords, atom_pad_mask, atom_chain_id, connected_chain_matrix, ref_element)
    return feats, {"buffer": buffer}, coords.unsqueeze(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "case_name, expect_zero",
    [
        ("all_padded", True),
        ("single_atom", True),
        ("all_ions", True),
        ("all_connected", True),
        ("non_overlapping", True),
        ("overlapping", False),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_geometric_edge_cases(case_name, expect_zero):
    """Energy and gradient match reference for geometric edge cases."""
    device = torch.device("cuda")
    dtype = torch.float64
    feats, parameters, coords_batched = _build_edge_case_inputs(case_name, device, dtype)

    ref_energy, ref_gradient = _vdw_reference(feats, parameters, coords_batched)
    triton_args = _extract_triton_args(feats, parameters)
    triton_energy = vdw_overlap_energy(coords_batched[0], *triton_args)
    triton_gradient = vdw_overlap_gradient(coords_batched[0], *triton_args)

    if expect_zero:
        assert ref_energy.item() == 0.0, f"Reference energy should be 0 for {case_name}"
        assert torch.all(ref_gradient == 0.0), f"Reference gradient should be 0 for {case_name}"
        assert triton_energy.item() == 0.0, f"Triton energy should be 0 for {case_name}"
        assert torch.all(triton_gradient == 0.0), f"Triton gradient should be 0 for {case_name}"
    else:
        assert ref_energy.item() > 0.0, f"Reference energy should be positive for {case_name}"
        assert not torch.all(ref_gradient == 0.0), f"Reference gradient should be non-zero for {case_name}"
        assert triton_energy.item() > 0.0, f"Triton energy should be positive for {case_name}"
        assert not torch.all(triton_gradient == 0.0), f"Triton gradient should be non-zero for {case_name}"
        torch.testing.assert_close(triton_energy, ref_energy[0])
        torch.testing.assert_close(triton_gradient, ref_gradient[0])


# ---------------------------------------------------------------------------
# Batched potential dispatch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=lambda x: f"dtype:{x}")
def test_batched_potential_parity(dtype):
    """VDWOverlapPotential.compute/compute_gradient on CUDA match PyTorch fallback for size_batch > 1."""
    device = torch.device("cuda")
    size_batch = 3
    feats, parameters, coords_batched = make_vdw_test_inputs(128, 5, dtype, device, size_batch=size_batch)

    ref_energy, ref_gradient = _vdw_reference(feats, parameters, coords_batched)
    assert ref_energy.shape == (size_batch,)
    assert ref_gradient.shape == (size_batch, 128, 3)
    assert (ref_energy > 0).all(), "Reference energy must not be trivially zero"

    potential = VDWOverlapPotential()
    cuda_energy = potential.compute(coords_batched, feats, parameters)
    cuda_gradient = potential.compute_gradient(coords_batched, feats, parameters)

    assert cuda_energy.shape == (size_batch,)
    assert cuda_gradient.shape == (size_batch, 128, 3)
    torch.testing.assert_close(cuda_energy, ref_energy)
    torch.testing.assert_close(cuda_gradient, ref_gradient)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=lambda x: f"dtype:{x}")
def test_rejects_low_precision_inputs(dtype):
    """vdw_overlap_energy and vdw_overlap_gradient reject fp16/bf16 inputs."""
    device = torch.device("cuda")
    N = 32
    feats, parameters, coords_batched = make_vdw_test_inputs(N, 3, torch.float32, device)

    triton_args = _extract_triton_args(feats, parameters)
    # Replace coords with low-precision version
    coords_low = coords_batched[0].to(dtype)

    with pytest.raises(ValueError, match="below float32"):
        vdw_overlap_energy(coords_low, *triton_args)

    with pytest.raises(ValueError, match="below float32"):
        vdw_overlap_gradient(coords_low, *triton_args)


# ---------------------------------------------------------------------------
# Register spilling
# ---------------------------------------------------------------------------


def _assert_no_register_spilling(path_to_ptx_file: Path):
    """Check that a PTX file shows no register spilling."""
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
    """VDW overlap kernel must not spill registers (both specializations)."""
    monkeypatch.setenv("TRITON_KERNEL_DUMP", "1")
    monkeypatch.setenv("TRITON_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TRITON_PTXAS_PATH", shutil.which("ptxas"))

    # Clear in-process JIT cache so the kernel recompiles and dumps PTX even
    # if earlier tests in this session already compiled it.
    _vdw_overlap_kernel.device_caches.clear()

    device = torch.device("cuda")
    N = 64
    feats, parameters, coords_batched = make_vdw_test_inputs(N, 3, torch.float32, device)
    triton_args = _extract_triton_args(feats, parameters)

    # Compile energy-only specialization (COMPUTE_GRADIENT=False)
    vdw_overlap_energy(coords_batched, *triton_args)

    # Compile energy+gradient specialization (COMPUTE_GRADIENT=True)
    vdw_overlap_gradient(coords_batched, *triton_args)

    # Check all PTX files dumped for the fused kernel
    ptx_files = list(tmp_path.glob("**/_vdw_overlap_kernel.ptx"))
    if not ptx_files:
        raise RuntimeError(f"No kernel PTX files found in {tmp_path}")
    for ptx_file in ptx_files:
        _assert_no_register_spilling(ptx_file)
