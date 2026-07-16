#!/usr/bin/env bash
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

# Codex SessionStart hook for fold-cp — inject the CP hard rules as
# hookSpecificOutput.additionalContext (Codex's SessionStart context channel).
# The Claude hook injects the same RULES.md via raw `cat` stdout; that path is
# left untouched. Codex sets CLAUDE_PLUGIN_ROOT, so the same path resolves here.
set -uo pipefail

RULES="${CLAUDE_PLUGIN_ROOT:-}/hooks/RULES.md"
[ -f "$RULES" ] || RULES="$(dirname "$0")/RULES.md"
[ -f "$RULES" ] || exit 0

if command -v jq >/dev/null 2>&1; then
  jq -Rs '{hookSpecificOutput:{hookEventName:"SessionStart",additionalContext:.}}' "$RULES"
elif command -v python3 >/dev/null 2>&1; then
  python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":open(sys.argv[1],encoding="utf-8").read()}}))' "$RULES"
else
  cat "$RULES"
fi
