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

"""Cross-rank error propagation for CP serial-dataset fetches.

The distributed dataset wrappers under
``boltz.distributed.data.module.trainingv2`` (2D CP) and
``trainingv2_1d`` (1D CP) execute the serial featurizer only on CP rank
zero, then broadcast the resulting feature dict to peer CP ranks via
``_distribute_features``.  If the rank-zero serial fetch raises, the
exception unwinds rank zero before it enters the first broadcast — peer
ranks are already blocked in ``broadcast_object_list`` and deadlock
indefinitely.  NCCL's per-collective watchdog does not fire because
rank zero never *entered* the collective.

:func:`rank_zero_fetch_with_propagation` is the cross-rank-safe wrapper
that converts this silent hang into a uniform ``RuntimeError`` on every
CP rank.  It prepends a single ``broadcast_object_list`` over the CP
group carrying either an "ok" sentinel or a serialized rank-zero
exception (type, repr, traceback).  Peers wake from that broadcast
*before* entering ``_distribute_features``; on error they re-raise
with the rank-zero traceback embedded so the failure is debuggable
across ranks.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

import torch
import torch.distributed


def rank_zero_fetch_with_propagation(
    fetch_fn: Callable[[], Any],
    is_cp_rank_zero: bool,
    cp_group: torch.distributed.ProcessGroup,
    cp_group_src_rank_global: int,
) -> Optional[Any]:
    """Run ``fetch_fn()`` on CP rank zero and propagate exceptions to peers.

    All CP ranks must call this function in lock-step (rank-symmetric
    bytecode).  The function performs exactly one collective: a
    ``broadcast_object_list`` over ``cp_group`` carrying a small payload
    indicating either success or the serialized rank-zero exception.

    On rank zero:
        - calls ``fetch_fn()``
        - if it raises, captures ``(exc_type_name, exc_repr, traceback_text)``
        - broadcasts ``[None]`` on success, ``[error_payload]`` on failure
        - returns the result, or raises ``RuntimeError`` (so rank zero
          and its peers raise uniformly)

    On peer ranks:
        - awaits the broadcast
        - returns ``None`` on success (caller still has to participate in
          the subsequent collectives inside ``_distribute_features``)
        - raises ``RuntimeError`` on failure, embedding rank zero's
          traceback

    The catch is intentionally ``BaseException`` so that ``SystemExit`` and
    ``KeyboardInterrupt`` (or any other deliberate-shutdown signal)
    propagate uniformly across CP ranks instead of leaving peers blocked.

    Parameters
    ----------
    fetch_fn : callable
        Zero-argument callable executed only on rank zero.  Typical
        usage wraps ``serial_dataset[idx]`` (training) or a small loop
        that scans for the first valid sample (validation).
    is_cp_rank_zero : bool
        True for the CP rank that owns serial-fetch execution.  Other
        CP ranks pass False.
    cp_group : torch.distributed.ProcessGroup
        Process group spanning the CP ranks that share the broadcast
        from rank zero.  This must be the same group used by the
        subsequent ``_distribute_features`` collectives so the
        rank-zero source matches.
    cp_group_src_rank_global : int
        Global (world-level) rank of the CP-rank-zero process inside
        ``cp_group``.  Matches ``min(get_process_group_ranks(cp_group))``
        by convention in this codebase.

    Returns
    -------
    Any or None
        Rank zero returns whatever ``fetch_fn()`` returned.  Other CP
        ranks return ``None``; they are expected to enter
        ``_distribute_features`` next, which will rebuild the feature
        dict from rank zero's broadcasts.

    Raises
    ------
    RuntimeError
        Raised on every CP rank if rank zero's ``fetch_fn`` raised.
        The message embeds the original exception type, repr, and
        rank-zero traceback so the failure is fully debuggable from
        any rank's stderr.
    """
    if is_cp_rank_zero:
        try:
            result = fetch_fn()
            error_payload: list[Optional[tuple[str, str, str]]] = [None]
        except BaseException as exc:  # noqa: BLE001 — propagate uniformly to peers
            tb_text = traceback.format_exc()
            error_payload = [(type(exc).__name__, repr(exc), tb_text)]
            result = None
    else:
        error_payload = [None]
        result = None

    torch.distributed.broadcast_object_list(
        error_payload,
        src=cp_group_src_rank_global,
        group=cp_group,
    )

    payload = error_payload[0]
    if payload is not None:
        exc_name, exc_repr, exc_tb = payload
        message = "\n".join(
            [
                f"CP rank-zero serial-dataset fetch raised {exc_name}: {exc_repr}",
                "Rank-zero traceback:",
                exc_tb,
            ]
        )
        raise RuntimeError(message)

    return result
