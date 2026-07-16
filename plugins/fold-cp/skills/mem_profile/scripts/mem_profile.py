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

"""Skeleton: per-rank CUDA memory snapshot for a torchrun workflow.

Mirrors Boltz2-CP's ``CUDAMemoryProfile`` Lightning callback, but for a plain
torchrun driver: wrap one end-to-end forward with
``torch.cuda.memory._record_memory_history()`` and dump a per-rank snapshot via
``_dump_snapshot()`` (opens at https://pytorch.org/memory_viz). Analyze the
pickle with ``mem_profile_analysis.py`` to attribute peaks to modules/lines.

Fill in the two TODO hooks for your model, then launch with torchrun:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    torchrun --standalone --nnodes=1 --nproc_per_node=<world> mem_profile.py --tag n1024
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist


def build_model(device: torch.device, args: argparse.Namespace):
    """TODO: construct/load your model on `device`, .eval(), and (for CP) wrap it
    with your distributed wrapper. Seed identically on every rank if the workflow
    has replicated stochastic parts (so ranks stay in sync)."""
    raise NotImplementedError("implement build_model for your model")


def run_forward(model, args: argparse.Namespace):
    """TODO: run ONE end-to-end forward (your real inference call) and return it.
    Keep it under torch.no_grad() for an inference memory profile."""
    raise NotImplementedError("implement run_forward for your model")


def _init_distributed(timeout_s: int) -> tuple[torch.device, int]:
    # torchrun sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=timeout_s))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    return torch.device("cuda", local), int(os.environ["RANK"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="snapshot filename tag (e.g. n1024)")
    ap.add_argument("--out-dir", default="profiling/results/mem")
    ap.add_argument("--max-entries", type=int, default=400_000, help="memory-history ring size")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument(
        "--record-from-start",
        action="store_true",
        help="record from before model build so params/load are in the trace "
        "(peak then matches max_memory_allocated exactly; larger pickle)",
    )
    args = ap.parse_args()
    if "RANK" not in os.environ:
        raise SystemExit("launch with torchrun --nproc_per_node=<world> mem_profile.py ...")

    torch.manual_seed(args.seed)
    device, rank = _init_distributed(args.timeout)

    if args.record_from_start:
        torch.cuda.memory._record_memory_history(max_entries=args.max_entries)

    model = build_model(device, args)

    for _ in range(args.warmup):  # settle cuBLAS/cuDNN workspaces + allocator pool
        run_forward(model, args)
    torch.cuda.synchronize(device)
    dist.barrier()

    if not args.record_from_start:
        # record only the profiled forward; peak excludes the resident baseline
        # (params/buffers allocated before this point) — note that in the report.
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.memory._record_memory_history(max_entries=args.max_entries)

    run_forward(model, args)
    torch.cuda.synchronize(device)

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"mem_{args.tag}_rank{rank}.pickle")
    try:
        torch.cuda.memory._dump_snapshot(path)
    except Exception as e:  # noqa: BLE001
        print(f"[rank{rank}] snapshot dump failed: {repr(e)[:160]}", flush=True)
    torch.cuda.memory._record_memory_history(enabled=None)

    stats = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device) / 2**30,
            torch.cuda.max_memory_reserved(device) / 2**30,
        ],
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"[rank0] peak_alloc={stats[0].item():.1f} GiB " f"peak_reserved={stats[1].item():.1f} GiB",
            flush=True,
        )
        print(
            f"[rank0] snapshots: {args.out_dir}/mem_{args.tag}_rank*.pickle "
            f"(view at https://pytorch.org/memory_viz)",
            flush=True,
        )
        print(
            f"[rank0] attribute peaks: python mem_profile_analysis.py {path}",
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
