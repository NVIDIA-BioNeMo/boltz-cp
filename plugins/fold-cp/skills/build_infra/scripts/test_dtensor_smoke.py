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

"""DTensor substrate smoke test, self-launched with mp.spawn.

Proves the DTensor machinery CP is built on works in this environment:
  * 2D mesh: square-tile a pair-like [N, N] tensor as
    (Shard(0), Shard(1)); check the local tile shape and a full_tensor()
    round-trip against the known global tensor.

Usage:
    timeout 120 python test_dtensor_smoke.py --world-size 4
Exit 0 = DTensor sharding + reassembly correct on all ranks.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor


def _setup(rank: int, world_size: int, backend: str, port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )


def worker(rank: int, world_size: int, backend: str, port: int, device_type: str) -> None:
    _setup(rank, world_size, backend, port)
    if device_type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # --- 2D mesh: square tile (Shard(0), Shard(1)) ---------------------------
    sq = int(math.isqrt(world_size))
    mesh2d = init_device_mesh(device_type, (sq, sq), mesh_dim_names=("cp0", "cp1"))
    m = sq * 4
    gen = torch.Generator().manual_seed(1)
    pair = torch.randn(m, m, generator=gen)
    if device_type == "cuda":
        pair = pair.cuda()
    dt2 = distribute_tensor(pair, mesh2d, [Shard(0), Shard(1)])
    tile = dt2.to_local()
    assert tile.shape == (
        m // sq,
        m // sq,
    ), f"[rank {rank}] 2D tile {tuple(tile.shape)} expected {(m // sq, m // sq)}"
    full2 = dt2.full_tensor()
    assert torch.allclose(full2, pair), f"[rank {rank}] 2D full_tensor round-trip mismatch"

    dist.barrier()
    if rank == 0:
        print(f"OK: DTensor 2D square-tile (Shard(0),Shard(1)) " f"verified on all {world_size} ranks.")
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--port", type=int, default=29556)
    args = ap.parse_args()

    sq = math.isqrt(args.world_size)
    if sq < 2 or sq * sq != args.world_size:
        raise SystemExit("world_size must be a perfect square >= 4 for the 2D DTensor smoke test")

    use_cuda = torch.cuda.is_available()
    if use_cuda and torch.cuda.device_count() < args.world_size:
        raise SystemExit(f"world_size={args.world_size} but only " f"{torch.cuda.device_count()} CUDA devices visible.")
    device_type = "cuda" if use_cuda else "cpu"
    backend = "nccl" if use_cuda else "gloo"
    mp.spawn(
        worker,
        args=(args.world_size, backend, args.port, device_type),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
