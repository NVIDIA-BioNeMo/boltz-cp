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

# --- Editable-install health-check (defensive scaffolding) ----------------
#
# The boltz editable install can silently re-point to a different worktree
# whenever any agent runs `pip install -e .` from there; when that worktree
# is later cleaned up, Python falls back to `/workspace/boltz/src` (a stale
# April-6 Docker baseline). Tests in spawned worker processes then either
# raise `ModuleNotFoundError` for post-April-6 modules (port_utils,
# confidence_1d, ...) or, worse, silently exercise stale source. Observed
# twice on 2026-05-15.
#
# Belt-and-suspenders fix that fires on every pytest invocation:
#  * Belt:     prepend the REPO-LOCAL `src/` to sys.path BEFORE `import
#              boltz` so the test always sees the worktree we're testing
#              from, regardless of where the editable install points.
#  * Suspenders: assert that `boltz.__file__` resolves under that same
#              `src/` after the prepend. If not, the test environment is
#              ambiguous and we want a loud, actionable failure rather
#              than a silent stale-source pass.
#
# This is preferable to a `.devcontainer/postCreate` check (only fires on
# container start; misses mid-session un-do) and to a CLAUDE.md note
# (relies on agent discipline; forgotten twice already).
import pathlib as _pathlib
import sys as _sys

_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[2]
_LOCAL_SRC = _REPO_ROOT / "src"
if str(_LOCAL_SRC) not in _sys.path:
    _sys.path.insert(0, str(_LOCAL_SRC))

import boltz as _boltz  # noqa: E402

_RESOLVED = _pathlib.Path(_boltz.__file__).resolve()
if not _RESOLVED.is_relative_to(_LOCAL_SRC):
    raise RuntimeError(
        f"boltz resolved to {_RESOLVED}\n"
        f"  expected under: {_LOCAL_SRC}\n"
        f"  fix: pip install -e {_REPO_ROOT} --no-deps --quiet\n"
        f"  then re-verify: pip show boltz | grep 'Editable project location'"
    )
del _pathlib, _sys, _boltz, _REPO_ROOT, _LOCAL_SRC, _RESOLVED
# --- end editable-install health-check ------------------------------------
