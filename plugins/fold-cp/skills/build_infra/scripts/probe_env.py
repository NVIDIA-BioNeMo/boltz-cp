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

"""Probe the local software/hardware stack for CP testability (no distributed init).

Prints a JSON report on stdout and a human summary on stderr. The
``recommended_cp`` block maps the live GPU count to whether 2D CP can be tested
here, following the build_infra rules:
  * 2D-CP needs a perfect-square cp size, smallest 4 (cp0=cp1=2).
  * Full end-to-end integration wants 8 GPUs.

It also reports the resource prerequisites the fold-cp skills care about (R4/R10):
free disk space, the NVIDIA driver version, and a hardware/software
*self-consistency* verdict. fold-cp is model-agnostic — it does NOT prescribe an
exact GPU SKU or count, because a custom model may ship kernels that only run on
certain GPUs (unknowable here). What it requires is that the user's hardware, their
PyTorch runtime, and their model be mutually compatible; the authoritative signal is
``torch.cuda.is_available()`` (CUDA usable on this driver + GPU arch). Model import
is checked in build_infra Step 3, and DTensor/device_mesh capability is proven by the
Step 4 smoke tests — the two checks probe_env.py cannot do on its own.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys

# fold-cp resource-prerequisite guidance (info-only, model-agnostic). Covers
# checkpoints + model containers + datasets + build artifacts for a typical
# co-folding model; a large model or many checkpoints can need more.
RECOMMEND_DISK_GIB = 150


def recommend(n: int) -> dict:
    rec: dict = {"gpu_count": n}
    # 2D: largest perfect square <= n, but at least 4
    sq = int(math.isqrt(n)) if n >= 1 else 0
    cp2d = sq * sq
    rec["can_2d"] = cp2d >= 4
    rec["cp2d"] = {
        "cp0": sq,
        "cp1": sq,
        "size_cp": cp2d,
        "dp": (n // cp2d) if cp2d >= 4 else 0,
        "world_size": cp2d * (n // cp2d) if cp2d >= 4 else 0,
    }
    rec["can_integration"] = n >= 8
    if rec["can_2d"]:
        rec["suggested_world_size"] = rec["cp2d"]["size_cp"]
        rec["suggested_topology"] = "2d"
    else:
        rec["suggested_world_size"] = 0
        rec["suggested_topology"] = "none-local (escalate to SLURM)"
    return rec


def disk_free_gib(path: str) -> dict:
    """Free/total disk (GiB) for the filesystem holding ``path`` (best-effort)."""
    try:
        du = shutil.disk_usage(path)
        return {
            "path": os.path.abspath(path),
            "free_GiB": round(du.free / 2**30, 1),
            "total_GiB": round(du.total / 2**30, 1),
        }
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "error": repr(exc)}


def nvidia_driver_version() -> str | None:
    """NVIDIA kernel-driver version via nvidia-smi (best-effort; may be absent)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def main() -> int:
    info: dict = {"python": platform.python_version()}
    n = 0
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda  # CUDA torch was BUILT against
        # CUDA driver API version torch sees (e.g. 12040 -> "12.4"); best-effort.
        try:
            dv = torch._C._cuda_getDriverVersion()
            info["cuda_driver_api"] = f"{dv // 1000}.{(dv % 1000) // 10}"
        except Exception:  # noqa: BLE001
            info["cuda_driver_api"] = None
        if torch.cuda.is_available():
            try:
                info["nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
            except Exception:  # noqa: BLE001
                info["nccl_version"] = None
            n = torch.cuda.device_count()
            gpus = []
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": p.name,
                        "capability": f"{p.major}.{p.minor}",
                        "mem_GiB": round(p.total_memory / 2**30, 1),
                    }
                )
            info["gpus"] = gpus
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)

    # --- Resource prerequisites (R4/R10): disk, driver, self-consistency ------
    # fold-cp is model-agnostic: it does not prescribe a fixed GPU SKU/count. The
    # requirement is a self-consistent stack (hardware <-> runtime <-> model).
    info["nvidia_driver"] = nvidia_driver_version()
    workdir = os.environ.get("FOLD_CP_WORKDIR", os.getcwd())
    info["disk"] = disk_free_gib(workdir)
    info["recommend_disk_GiB"] = RECOMMEND_DISK_GIB

    sc: dict = {
        "cuda_usable": bool(info.get("cuda_available")),
        "torch_cuda_build": info.get("cuda_version"),
        "cuda_driver_api": info.get("cuda_driver_api"),
        "nvidia_driver": info.get("nvidia_driver"),
        # These two require the target model / a multi-rank launch, so probe_env
        # cannot settle them; build_infra runs them in later steps.
        "model_import_check": "deferred to build_infra Step 3",
        "dtensor_mesh_check": "deferred to build_infra Step 4 smoke tests",
    }
    notes: list = []
    # torch.cuda.is_available() is the authoritative self-consistency signal: it is
    # True only when the installed driver, the CUDA build of torch, and the GPU arch
    # are mutually compatible.
    if not info.get("cuda_available"):
        notes.append(
            "torch.cuda.is_available() is False — the NVIDIA driver, the CUDA build "
            "of torch, and the GPU arch are NOT mutually consistent (or no GPU is "
            "visible). Resolve before any CP work."
        )
    disk = info.get("disk", {})
    if isinstance(disk, dict) and disk.get("free_GiB") is not None:
        if disk["free_GiB"] < RECOMMEND_DISK_GIB:
            notes.append(
                f"free disk {disk['free_GiB']} GiB < recommended {RECOMMEND_DISK_GIB} "
                f"GiB at {disk['path']} (checkpoints/containers/datasets may not fit)."
            )
    sc["consistent"] = bool(info.get("cuda_available"))
    sc["notes"] = notes
    info["self_consistency"] = sc

    info["recommended_cp"] = recommend(n)
    print(json.dumps(info, indent=2))

    # --- human summary on stderr ----------------------------------------------
    r = info["recommended_cp"]
    print("\n=== CP testability summary ===", file=sys.stderr)
    print(f"GPUs detected: {n}", file=sys.stderr)
    print(f"  2D-CP testable: {r['can_2d']}  -> {r['cp2d']}", file=sys.stderr)
    print(f"  integration (8 GPU): {r['can_integration']}", file=sys.stderr)
    print(
        f"  suggested: topology={r['suggested_topology']} " f"world_size={r['suggested_world_size']}",
        file=sys.stderr,
    )

    print(
        "\n=== Resource prerequisites (info-only, model-agnostic) ===",
        file=sys.stderr,
    )
    d = info.get("disk", {})
    if isinstance(d, dict) and d.get("free_GiB") is not None:
        print(
            f"  disk free: {d['free_GiB']} GiB / {d['total_GiB']} GiB at {d['path']} "
            f"(recommend >= {RECOMMEND_DISK_GIB} GiB)",
            file=sys.stderr,
        )
    print(
        f"  NVIDIA driver: {info.get('nvidia_driver')}   "
        f"torch CUDA build: {info.get('cuda_version')}   "
        f"CUDA driver API: {info.get('cuda_driver_api')}",
        file=sys.stderr,
    )
    print(f"  self-consistent (CUDA usable here): {sc['consistent']}", file=sys.stderr)
    print(
        "  NOTE: fold-cp prescribes no fixed GPU SKU/count — a custom model may need\n"
        "        specific GPUs (custom kernels). The requirement is that YOUR hardware,\n"
        "        PyTorch runtime, and model are mutually compatible; model import is\n"
        "        checked in build_infra Step 3, DTensor/device_mesh in Step 4.",
        file=sys.stderr,
    )
    for note in notes:
        print(f"  WARNING: {note}", file=sys.stderr)

    if not info.get("cuda_available"):
        print(
            "WARNING: torch.cuda.is_available() is False — CP requires a CUDA build "
            "of PyTorch. Resolve before running the smoke tests.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
