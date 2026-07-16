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


# Codex PreToolUse guard for fold-cp.
#
# Adapts the Claude guard (guard_serial_and_tests.py) to Codex's tool model WITHOUT
# modifying it:
#   * Codex edits files via the `apply_patch` tool (no Write/Edit). Target paths live
#     inside tool_input.command as `*** {Add,Update,Delete} File: <path>` /
#     `*** Move to: <path>` lines.
#   * Codex shell calls use tool_name "Bash" with tool_input.command (same as Claude).
#   * Codex PreToolUse denies via top-level {"permissionDecision":"deny",
#     "permissionDecisionReason":"..."} (Claude nests it under hookSpecificOutput).
# Config-load + serial/test logic is reused from the Claude guard via import.
# Fail-open: any error allows the call.

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard_serial_and_tests as g  # noqa: E402  (import after sys.path tweak)


def _allow() -> None:
    sys.exit(0)


def _deny(reason: str) -> None:
    print(json.dumps({"permissionDecision": "deny", "permissionDecisionReason": reason}))
    sys.exit(0)


_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+?)\s*$", re.MULTILINE)
_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+?)\s*$", re.MULTILINE)


def _command_str(tool_input: dict) -> str:
    cmd = tool_input.get("command")
    if isinstance(cmd, list):
        return "\n".join(str(c) for c in cmd)
    return str(cmd or "")


def _apply_patch_paths(patch: str) -> list[str]:
    return _FILE_RE.findall(patch) + _MOVE_RE.findall(patch)


def _patch_introduces_promote_types(patch: str) -> bool:
    return any(ln.startswith("+") and "promote_types" in ln for ln in patch.splitlines())


def _hunk_for_path(patch: str, target: str) -> str:
    """Return only the apply_patch hunk for file ``target`` — the lines from that
    file's ``*** ... File:`` / ``*** Move to:`` header up to the next file header
    (empty if absent). Lets the promote_types bypass inspect the matched serial
    file's own hunk, not a ``+promote_types`` anywhere in a multi-file patch."""
    out: list[str] = []
    capturing = False
    for line in patch.splitlines():
        m = _FILE_RE.match(line) or _MOVE_RE.match(line)
        if m:
            capturing = m.group(1).strip() == target
            continue
        if capturing:
            out.append(line)
    return "\n".join(out)


def main() -> None:
    """Read the Codex PreToolUse event from stdin and allow or deny the call: deny an
    un-timeout-ed distributed test (Rule 16) or an edit to a serial ground-truth file
    (Rule 2), else allow. Fail-open on any error."""
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except (ValueError, OSError):
        _allow()
        return

    cwd = event.get("cwd") or os.getcwd()
    cfg = g._load_config(cwd)
    if cfg is None:  # guard disabled until the project opts in
        _allow()
        return

    tool_name = event.get("tool_name") or ""
    tool_input = event.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        _allow()
        return

    # Rule 16 — distributed tests must carry a timeout (Bash; identical to Claude).
    if tool_name == "Bash":
        if not cfg.get("enforce_test_timeout", True):
            _allow()
            return
        markers = cfg.get("distributed_test_markers") or ["tests/distributed"]
        command = _command_str(tool_input)
        if g._is_distributed_test_cmd(command, markers) and not g._has_timeout_prefix(command):
            _deny(
                "fold-cp Rule 16 (distributed tests carry a timeout): this command runs the "
                "distributed test suite without a `timeout` prefix. NCCL deadlocks would hang the "
                "session. Re-run as `timeout 120 <command>`. Override via .fold-cp/config.json."
            )
        _allow()
        return

    # Rule 2 — serial code is ground truth. Codex edits via apply_patch; the target
    # paths are inside the patch text. (Edit/Write kept for forward-compat / aliases.)
    if tool_name in ("apply_patch", "Edit", "Write"):
        if not cfg.get("enforce_serial_protection", True):
            _allow()
            return
        serial_paths = cfg.get("serial_paths") or []
        if not serial_paths:
            _allow()
            return
        patch = _command_str(tool_input)
        targets = _apply_patch_paths(patch) or [p for p in (tool_input.get("file_path"), tool_input.get("path")) if p]
        hit = next((t for t in targets if g._matches_any(t, cwd, serial_paths)), None)
        if hit:
            if cfg.get("allow_serial_dtype_promotion") and _patch_introduces_promote_types(_hunk_for_path(patch, hit)):
                _allow()  # sanctioned .float() -> torch.promote_types(...) dtype fix
                return
            _deny(
                f"fold-cp Rule 2 (serial is ground truth): '{hit}' is a designated serial-reference "
                "file and must not be edited for distributed purposes. Copy it into your distributed "
                "subtree and modify the copy. (Only sanctioned serial edit: a `.float()` -> "
                "`torch.promote_types(...)` dtype fix, enabled with allow_serial_dtype_promotion: "
                "true.) Override via .fold-cp/config.json."
            )
        _allow()
        return

    _allow()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # never let a guard bug block real work
        _allow()
