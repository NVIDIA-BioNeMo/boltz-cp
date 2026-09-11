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

"""Smoke-test the collectives CP relies on, self-launched with mp.spawn.

Verifies, on every rank, against analytic expected values:
  * batch_isend_irecv  — ring P2P (send to next, recv from prev)
  * all_gather         — each rank receives every rank's shard
  * all_reduce         — sum across ranks
  * reduce_scatter     — scattered sum across ranks

Usage:
    timeout 120 python test_collectives.py --world-size 4
Exit 0 = all collectives correct on all ranks. A hang to the timeout means an
NCCL/transport/deadlock problem — re-run with NCCL_DEBUG=INFO.
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _setup(rank: int, world_size: int, backend: str, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )


def worker(rank: int, world_size: int, backend: str, port: int, use_cuda: bool) -> None:
    _setup(rank, world_size, backend, port)
    if use_cuda:
        dev = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")

    # --- all_reduce: sum of (r+1) over ranks ---------------------------------
    t = torch.full((4,), float(rank + 1), device=dev)
    dist.all_reduce(t)
    expected = float(sum(range(1, world_size + 1)))
    assert torch.allclose(
        t, torch.full_like(t, expected)
    ), f"[rank {rank}] all_reduce got {t[0].item()} expected {expected}"

    # --- all_gather: rank r contributes value r -----------------------------
    src = torch.full((2,), float(rank), device=dev)
    gathered = [torch.empty(2, device=dev) for _ in range(world_size)]
    dist.all_gather(gathered, src)
    for i, g in enumerate(gathered):
        assert torch.allclose(
            g, torch.full_like(g, float(i))
        ), f"[rank {rank}] all_gather slot {i} = {g[0].item()} expected {i}"

    # --- reduce_scatter: every rank's chunk is sum_r(r) ----------------------
    input_list = [torch.full((2,), float(rank), device=dev) for _ in range(world_size)]
    out = torch.empty(2, device=dev)
    dist.reduce_scatter(out, input_list)
    rs_expected = float(sum(range(world_size)))
    assert torch.allclose(
        out, torch.full_like(out, rs_expected)
    ), f"[rank {rank}] reduce_scatter got {out[0].item()} expected {rs_expected}"

    # --- batch_isend_irecv: ring (recv prev rank's id) -----------------------
    send_t = torch.full((4,), float(rank), device=dev)
    recv_t = torch.empty(4, device=dev)
    nxt = (rank + 1) % world_size
    prv = (rank - 1) % world_size
    ops = [
        dist.P2POp(dist.isend, send_t, nxt),
        dist.P2POp(dist.irecv, recv_t, prv),
    ]
    for req in dist.batch_isend_irecv(ops):
        req.wait()
    if use_cuda:
        torch.cuda.synchronize()
    assert torch.allclose(
        recv_t, torch.full_like(recv_t, float(prv))
    ), f"[rank {rank}] batch_isend_irecv got {recv_t[0].item()} expected {prv}"

    dist.barrier()
    if rank == 0:
        print(
            f"OK: batch_isend_irecv, all_gather, all_reduce, reduce_scatter "
            f"correct on all {world_size} ranks (backend={backend})."
        )
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--port", type=int, default=29555)
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"world_size={args.world_size} but only " f"{torch.cuda.device_count()} CUDA devices visible.")
    backend = "nccl" if use_cuda else "gloo"
    mp.spawn(
        worker,
        args=(args.world_size, backend, args.port, use_cuda),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
