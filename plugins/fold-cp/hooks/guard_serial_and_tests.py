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

"""fold-cp PreToolUse guard — opt-in, fail-open.

Mechanically enforces two of the fold-cp hard rules (see hooks/RULES.md):

  * Rule 2  — serial code is ground truth: block Write/Edit to files the user has
              designated as the serial (non-distributed) reference.
  * Rule 16 — distributed tests must carry a timeout: block a Bash command that
              runs the distributed test suite without a ``timeout`` prefix.

The guard is OFF until the project contains a ``.fold-cp/config.json`` (written by
``/fold-cp:learn_context``). With no config, every call is allowed. Any error or
unexpected input also allows the call — this hook never blocks legitimate work
because of its own bugs.

Reads the PreToolUse event as JSON on stdin and, to block, prints
``{"decision": "block", "reason": "..."}`` to stdout (per the plugin hooks
reference). Otherwise prints nothing and exits 0.

config.json schema (all keys optional)::

    {
      "enforce_serial_protection": true,
      "serial_paths": ["src/mymodel/model/**", "src/mymodel/data/**"],
      "allow_serial_dtype_promotion": false,
      "enforce_test_timeout": true,
      "distributed_test_markers": ["tests/distributed"]
    }

``allow_serial_dtype_promotion`` (default false) is the Rule-2 escape hatch: when true,
a *targeted* ``Edit`` to a serial file is allowed **iff** it introduces
``torch.promote_types(...)`` (the sanctioned, production-no-op dtype-parametrisation that
enables fp64 parity tests). Arbitrary serial rewrites and full-file ``Write``s are still
blocked. It is a backstop, not a security boundary — RULES.md Rule 2 is the real constraint.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys


def _allow() -> None:
    """Allow the tool call (emit nothing, exit 0)."""
    sys.exit(0)


def _block(reason: str) -> None:
    """Block the tool call with an actionable reason."""
    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)


def _load_config(cwd: str) -> dict | None:
    path = os.path.join(cwd, ".fold-cp", "config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return cfg if isinstance(cfg, dict) else None
    except (OSError, ValueError):
        return None


def _rel_variants(file_path: str, cwd: str) -> list[str]:
    """Return path spellings to match against globs: absolute and cwd-relative."""
    variants = {file_path}
    abspath = os.path.abspath(file_path)
    variants.add(abspath)
    try:
        rel = os.path.relpath(abspath, cwd)
        if not rel.startswith(".."):
            variants.add(rel)
    except ValueError:
        pass
    # Normalise to posix separators so config globs stay OS-agnostic.
    return [v.replace(os.sep, "/") for v in variants]


def _matches_any(file_path: str, cwd: str, patterns: list[str]) -> bool:
    candidates = _rel_variants(file_path, cwd)
    for pat in patterns:
        pat = str(pat).replace(os.sep, "/")
        if any(fnmatch.fnmatch(c, pat) for c in candidates):
            return True
    return False


def _is_sanctioned_dtype_edit(tool_name: str, tool_input: dict) -> bool:
    """Rule 2 exception: a *benign* dtype-parametrisation edit that swaps a hardcoded
    fp32 down-cast for ``torch.promote_types(...)`` — a production no-op that enables
    fp64 parity tests. Recognised only for a targeted ``Edit`` that INTRODUCES
    ``promote_types`` (absent from the replaced text); a full-file ``Write`` to a serial
    path is never auto-allowed. A backstop, not a security boundary."""
    if tool_name != "Edit":
        return False
    new = tool_input.get("new_string") or ""
    old = tool_input.get("old_string") or ""
    return "promote_types" in new and "promote_types" not in old


def _guard_write(tool_name: str, tool_input: dict, cwd: str, cfg: dict) -> None:
    if not cfg.get("enforce_serial_protection", True):
        _allow()
    serial_paths = cfg.get("serial_paths") or []
    if not serial_paths:
        _allow()
    file_path = tool_input.get("file_path") or tool_input.get("path") or ""
    if file_path and _matches_any(file_path, cwd, serial_paths):
        if cfg.get("allow_serial_dtype_promotion") and _is_sanctioned_dtype_edit(tool_name, tool_input):
            _allow()  # Rule 2 sanctioned dtype-parametrisation (.float() -> promote_types)
        _block(
            "fold-cp Rule 2 (serial is ground truth): "
            f"'{file_path}' is a designated serial-reference file and must not be "
            "edited for distributed purposes. Copy it into your distributed "
            "subtree and modify the copy instead. (The ONLY sanctioned serial edit is a "
            "benign `.float()`/`.to(torch.float32)` -> `torch.promote_types(...)` dtype "
            "fix; enable it with `allow_serial_dtype_promotion: true` in "
            ".fold-cp/config.json. A test-scoped monkeypatch needs no serial edit.) To "
            "override entirely, edit .fold-cp/config.json (serial_paths / "
            "enforce_serial_protection)."
        )
    _allow()


def _is_distributed_test_cmd(command: str, markers: list[str]) -> bool:
    runs_pytest = "pytest" in command or "spawn_multiprocessing" in command
    if not runs_pytest:
        return False
    return any(str(m) in command for m in markers)


def _has_timeout_prefix(command: str) -> bool:
    # Accept `timeout`, GNU `timeout 120`, or `timeout -k ... 120` anywhere a
    # leading clause of a sub-command. Conservative: any standalone `timeout`
    # token followed by something counts.
    import re

    return re.search(r"(^|[;&|]|\bthen\b|\bdo\b)\s*timeout\s+\S", command) is not None


def _guard_bash(tool_input: dict, cfg: dict) -> None:
    if not cfg.get("enforce_test_timeout", True):
        _allow()
    markers = cfg.get("distributed_test_markers") or ["tests/distributed"]
    command = tool_input.get("command") or ""
    if _is_distributed_test_cmd(command, markers) and not _has_timeout_prefix(command):
        _block(
            "fold-cp Rule 16 (distributed tests carry a timeout): this command "
            "runs the distributed test suite without a `timeout` prefix. NCCL "
            "deadlocks would hang the session indefinitely. Re-run as "
            "`timeout 120 <command>`. To override, edit .fold-cp/config.json "
            "(enforce_test_timeout / distributed_test_markers)."
        )
    _allow()


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except (ValueError, OSError):
        _allow()
        return

    cwd = event.get("cwd") or os.getcwd()
    cfg = _load_config(cwd)
    if cfg is None:  # guard disabled until the project opts in
        _allow()
        return

    tool_name = event.get("tool_name") or ""
    tool_input = event.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        _allow()
        return

    if tool_name in ("Write", "Edit", "NotebookEdit"):
        _guard_write(tool_name, tool_input, cwd, cfg)
    elif tool_name == "Bash":
        _guard_bash(tool_input, cfg)
    else:
        _allow()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # never let a guard bug block real work
        _allow()
