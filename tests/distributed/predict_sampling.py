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

"""Record and replay serial sampling inputs for end-to-end geometry tests.

This is an offline test oracle, not a production distributed RNG. Recording leaves
the serial RNG stream unchanged; replay gives CP the same noise and augmentations
so geometry comparisons do not include differences between independent ensembles.
Production RNG entropy is covered separately by the inference RNG tests.
"""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


@contextmanager
def record_sampling_inputs(path: Path):
    """Record one serial prediction's raw coordinate noise and augmentations."""
    import boltz.model.modules.diffusionv2 as serial

    original_sample = serial.AtomDiffusion.sample
    samples_recorded = 0

    def sample(module, atom_mask, *args, **kwargs):
        nonlocal samples_recorded
        assert samples_recorded == 0, "Sampling recorder expects exactly one input"
        trace = {"noise": [], "rotations": [], "translations": [], "atom_mask": atom_mask.cpu()}
        original_randn = torch.randn
        original_augmentation = serial.compute_random_augmentation

        def randn(*args, **kwargs):
            value = original_randn(*args, **kwargs)
            if value.ndim == 3 and value.shape[1:] == (atom_mask.shape[1], 3):
                trace["noise"].append(value.cpu())
            return value

        def augmentation(*args, **kwargs):
            rotation, translation = original_augmentation(*args, **kwargs)
            trace["rotations"].append(rotation.cpu())
            trace["translations"].append(translation.cpu())
            return rotation, translation

        with patch.object(torch, "randn", randn), patch.object(serial, "compute_random_augmentation", augmentation):
            result = original_sample(module, atom_mask, *args, **kwargs)

        steps = len(trace["rotations"])
        assert steps > 0, "Geometry guard must exercise coordinate augmentation"
        assert len(trace["translations"]) == steps
        assert len(trace["noise"]) == steps + 1
        torch.save(trace, path)
        samples_recorded += 1
        print(f"Recorded sampling inputs: {steps + 1} noise tensors, {steps} rotations/translations", flush=True)
        return result

    with patch.object(serial.AtomDiffusion, "sample", sample):
        yield
    assert samples_recorded == 1, "Serial prediction did not produce sampling inputs"


@contextmanager
def replay_sampling_inputs(path: Path):
    """Replay one recorded prediction on a DP=1, 1D-CP mesh, restoring on exit."""
    import boltz.distributed.model.modules.diffusion_1d as parallel

    trace = torch.load(path, map_location="cpu", weights_only=True)
    original_sample = parallel.AtomDiffusion1D.sample
    samples_replayed = 0

    def sample(module, atom_mask, *args, **kwargs):
        nonlocal samples_replayed
        assert samples_replayed == 0, "Sampling replay expects exactly one input"
        mesh = atom_mask.device_mesh
        assert mesh.ndim == 2 and mesh.size(0) == 1, "Sampling replay requires a DP=1 1D mesh"
        torch.testing.assert_close(atom_mask.full_tensor().cpu(), trace["atom_mask"], rtol=0, atol=0)
        positions = {"noise": 0, "translations": 0, "rotations": 0}

        def take(key):
            index = positions[key]
            assert index < len(trace[key]), f"Unexpected additional sampling draw: {key}"
            positions[key] += 1
            return trace[key][index]

        def randn(shape, device_mesh, placements, dtype=torch.float32, scale=1.0):
            key = "translations" if tuple(placements) == (Shard(0), Replicate()) else "noise"
            if key == "noise":
                assert tuple(placements) == (Shard(0), Shard(1))
                assert shape[1] % device_mesh.size(1) == 0
            value = take(key)
            assert tuple(value.shape) == tuple(shape), f"{key}: recorded {value.shape}, requested {shape}"
            value = value.to(device=module.device, dtype=dtype) * scale
            result = distribute_tensor(value, device_mesh, placements)
            if key == "noise" and device_mesh.size(1) > 1:
                assert result.to_local().shape[1] < result.shape[1], "Noise must be sharded over CP"
            return result

        def rotations(n, dtype, device):
            value = take("rotations")
            assert value.shape == (n, 3, 3)
            return value.to(device=device, dtype=dtype)

        with (
            patch.object(parallel, "create_distributed_randn", randn),
            patch.object(parallel, "random_rotations", rotations),
        ):
            result = original_sample(module, atom_mask, *args, **kwargs)

        assert positions["noise"] == len(trace["noise"])
        assert positions["translations"] == len(trace["translations"])
        expected_rotations = len(trace["rotations"]) if mesh.get_local_rank(1) == 0 else 0
        assert positions["rotations"] == expected_rotations
        samples_replayed += 1
        if mesh.get_local_rank(1) == 0:
            print(f"Replayed sampling inputs on mesh {tuple(mesh.shape)}: {positions}", flush=True)
        return result

    with patch.object(parallel.AtomDiffusion1D, "sample", sample):
        yield
    assert samples_replayed == 1, "CP prediction did not replay sampling inputs"


def run_serial_predict_with_sampling_inputs(
    yaml_path: Path,
    checkpoint: Path,
    cache_dir: Path,
    diffusion_samples: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Generate a fresh serial reference and matching trace in an isolated process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "sampling_inputs.pt"
    log_path = out_dir / "serial.log"
    command = [
        sys.executable,
        "-m",
        "tests.distributed.predict_sampling",
        str(trace_path),
        "predict",
        str(yaml_path),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(cache_dir),
        "--checkpoint",
        str(checkpoint),
        "--diffusion_samples",
        str(diffusion_samples),
        "--seed",
        "42",
        "--input_format",
        "config_files",
        "--devices",
        "1",
        "--accelerator",
        "gpu",
        "--recycling_steps",
        "10",
        "--sampling_steps",
        "200",
        "--model",
        "boltz2",
        "--max_msa_seqs",
        "2048",
        "--override",
    ]
    print(f"Generating {diffusion_samples} serial samples and sampling trace: {out_dir}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=False)
    assert result.returncode == 0, f"Serial predict failed; see {log_path}:\n{log_path.read_text()[-4000:]}"
    assert trace_path.is_file(), f"Serial sampling trace missing: {trace_path}"
    predictions = out_dir / f"boltz_results_{yaml_path.stem}" / "predictions"
    assert len(list(predictions.rglob("*.cif"))) == diffusion_samples
    return predictions, trace_path


if __name__ == "__main__":
    from boltz.main import cli

    with record_sampling_inputs(Path(sys.argv[1])):
        cli(sys.argv[2:], standalone_mode=False)
