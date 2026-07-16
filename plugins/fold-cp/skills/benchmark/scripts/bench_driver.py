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

"""fold-cp benchmark driver — max-token probe + walltime, self-launched with mp.spawn.

Adapt to a real model by providing an ``--adapter`` module that defines:

    build_model(mesh, device) -> nn.Module
    build_batch(N, n_atoms, S, mesh, device) -> Any        # the forward input
    run_step(model, batch, train: bool) -> Any             # fwd (+bwd+step if train)

Run ``--adapter demo`` for a synthetic sharded-pair workload that validates the
harness mechanics without a real model.

Examples:
    timeout 600 python bench_driver.py --adapter demo --cp 4 \
        --workflow inference --sizes 256,512,1024 --max-token-probe
    timeout 600 python bench_driver.py --adapter myproj.bench_adapter --cp 4 \
        --workflow training --sizes 512,1024 --atoms 8192 --msa 128

Walltime per point is the median over repeats of the slowest rank (all_reduce MAX);
peak memory is the max over ranks. Writes a markdown table + CSV on rank 0.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib
import math
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor


# --------------------------------------------------------------------------- #
# Demo adapter — synthetic workload so the harness runs without a real model.
# --------------------------------------------------------------------------- #
class _DemoModel(torch.nn.Module):
    def __init__(self, c: int = 64):
        super().__init__()
        self.w1 = torch.nn.Linear(c, c)
        self.w2 = torch.nn.Linear(c, c)

    def forward(self, z):  # z: pair-like DTensor [N, N, C]
        h = torch.relu(self.w1(z))
        return self.w2(h)


def _demo_build_model(mesh, device):
    return _DemoModel().to(device)


def _demo_build_batch(N, n_atoms, S, mesh, device):
    c = 64
    gen = torch.Generator(device=device).manual_seed(0)
    z = torch.randn(N, N, c, generator=gen, device=device)
    return distribute_tensor(z, mesh, [Shard(0), Shard(1)]).requires_grad_(True)


def _demo_run_step(model, batch, train):
    out = model(batch)
    if train:
        go = torch.randn_like(out.to_local())
        out.to_local().backward(go)
    return out


_DEMO = {
    "build_model": _demo_build_model,
    "build_batch": _demo_build_batch,
    "run_step": _demo_run_step,
}


def _load_adapter(name: str) -> dict:
    if name == "demo":
        return _DEMO
    mod = importlib.import_module(name)
    return {k: getattr(mod, k) for k in ("build_model", "build_batch", "run_step")}


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def _reduce_max(value: float, device) -> float:
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _any_oom(flag: bool, device) -> bool:
    t = torch.tensor([1.0 if flag else 0.0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() > 0


def _time_point(adapter, model, N, n_atoms, S, mesh, device, train, warmup, repeats):
    """Return (median_ms, peak_gib) or None if this size OOMs on any rank."""
    torch.cuda.reset_peak_memory_stats(device)
    oom = False
    try:
        for _ in range(warmup):
            batch = adapter["build_batch"](N, n_atoms, S, mesh, device)
            adapter["run_step"](model, batch, train)
        torch.cuda.synchronize(device)

        times = []
        for _ in range(repeats):
            batch = adapter["build_batch"](N, n_atoms, S, mesh, device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            adapter["run_step"](model, batch, train)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1e3)
    except torch.cuda.OutOfMemoryError:
        oom = True
        torch.cuda.empty_cache()

    if _any_oom(oom, device):
        return None

    slowest = [_reduce_max(t, device) for t in times]
    peak = _reduce_max(torch.cuda.max_memory_allocated(device) / 2**30, device)
    return statistics.median(slowest), peak


def worker(rank, world_size, args, port):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    sq = math.isqrt(world_size)
    assert sq * sq == world_size, "2d benchmark needs a perfect-square world size"
    mesh = init_device_mesh("cuda", (sq, sq), mesh_dim_names=("cp0", "cp1"))

    adapter = _load_adapter(args.adapter)
    model = adapter["build_model"](mesh, device)
    train = args.workflow == "training"

    rows = []
    sizes = [int(s) for s in args.sizes.split(",")]
    for N in sizes:
        res = _time_point(
            adapter,
            model,
            N,
            args.atoms,
            args.msa,
            mesh,
            device,
            train,
            args.warmup,
            args.repeats,
        )
        if res is None:
            if rank == 0:
                print(f"N={N}: OOM")
            rows.append((N, args.atoms, args.msa, "OOM", "OOM"))
            break
        ms, gib = res
        if rank == 0:
            print(f"N={N}: {ms:.1f} ms, peak {gib:.2f} GiB")
        rows.append((N, args.atoms, args.msa, f"{gib:.2f}", f"{ms:.1f}"))

    max_fit = None
    if args.max_token_probe:
        N = sizes[-1] if sizes else 512
        while True:
            nxt = N * 2
            res = _time_point(adapter, model, nxt, args.atoms, args.msa, mesh, device, train, 1, 1)
            if res is None:
                break
            max_fit = nxt
            N = nxt
        if rank == 0:
            print(f"max-token probe: largest fitting N ~ {max_fit}")

    if rank == 0:
        _emit(args, rows, max_fit, world_size)
    dist.destroy_process_group()


def _emit(args, rows, max_fit, world_size):
    cp = f"2d cp={world_size}"
    md = [
        f"# CP benchmark — {args.workflow}",
        "",
        f"- date: (stamp externally)  cp: {cp}  adapter: {args.adapter}",
        f"- torch {torch.__version__}, cuda {torch.version.cuda}, " f"gpu {torch.cuda.get_device_name(0)}",
        f"- max-fit N at this CP size: {max_fit if max_fit else 'not probed'}",
        "",
        "| N_tokens | N_atoms | MSA_seq | CP config | peak_mem (GiB) | walltime (ms) |",
        "|---------:|--------:|--------:|-----------|---------------:|--------------:|",
    ]
    for N, a, s, mem, ms in rows:
        md.append(f"| {N} | {a} | {s} | {cp} | {mem} | {ms} |")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    csv_path = args.out.rsplit(".", 1)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["N_tokens", "N_atoms", "MSA_seq", "cp_config", "peak_GiB", "ms"])
        for N, a, s, mem, ms in rows:
            w.writerow([N, a, s, cp, mem, ms])
    print(f"wrote {args.out} and {csv_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="demo", help="module path or 'demo'")
    ap.add_argument("--workflow", choices=["inference", "training"], default="inference")
    ap.add_argument("--cp", type=int, required=True, help="world size (cp ranks)")
    ap.add_argument("--sizes", default="256,512,1024", help="comma N_tokens sweep")
    ap.add_argument("--atoms", type=int, default=0)
    ap.add_argument("--msa", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-token-probe", action="store_true")
    ap.add_argument("--out", default="docs/cp_benchmark.md")
    ap.add_argument("--port", type=int, default=29560)
    args = ap.parse_args()

    sq = math.isqrt(args.cp)
    if sq < 2 or sq * sq != args.cp:
        raise SystemExit("cp must be a perfect square >= 4 for the 2D benchmark")

    if not torch.cuda.is_available() or torch.cuda.device_count() < args.cp:
        raise SystemExit(
            f"need {args.cp} CUDA devices; have " f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    mp.spawn(worker, args=(args.cp, args, args.port), nprocs=args.cp, join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
