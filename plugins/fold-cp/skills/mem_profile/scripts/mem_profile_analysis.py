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

"""Attribute the top-N CUDA-memory peaks in a snapshot to modules + lines.

Model-agnostic, pure stdlib (no torch import needed). Consumes a snapshot pickle
from ``torch.cuda.memory._dump_snapshot`` (see mem_profile.py).

Method
------
The snapshot's ``device_traces`` is a time-ordered list of allocator events
(``alloc`` / ``free_requested`` / ...), each with ``size``, ``addr``, ``time_us``
and a captured Python+C++ stack (``frames``, innermost->outermost). We:
  1. replay events -> the *allocated-bytes* timeline (alloc += size; first free of
     an addr -= size), which matches ``torch.cuda.max_memory_allocated``;
  2. take the top-N *distinct* local maxima (deduped by level so peaks are
     different plateaus/phases, not adjacent samples of one peak);
  3. at each peak, attribute every *live* allocation to the deepest stack frame
     under ``--project-root`` (file:line + enclosing Class.method via ast),
     aggregating bytes per site;
  4. emit markdown — peaks sorted by size, contributors within a peak by bytes —
     with clickable links that jump to the line of code.

Usage
-----
    python mem_profile_analysis.py <snapshot.pickle> [--top-n 6] [--top-contributors 15]
        [--project-root .] [--link-style vscode|cursor|github|file|none]
        [--repo-url https://github.com/org/repo --commit <sha>] [--out OUT.md]
"""

from __future__ import annotations

import argparse
import ast
import functools
import os
import pickle
import subprocess

_GiB = 1024**3
_MiB = 1024**2


# --- source -> enclosing Class.method (readable, navigable site labels) ------
@functools.cache
def _qualname_index(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except Exception:  # noqa: BLE001
        return []
    spans: list[tuple[int, int, str]] = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                end = getattr(child, "end_lineno", child.lineno)
                spans.append((child.lineno, end, name))
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(tree, "")
    return spans


def _qualname_at(path: str, line: int) -> str | None:
    best = None
    for start, end, name in _qualname_index(path):
        if start <= line <= end and (best is None or (end - start) < (best[1] - best[0])):
            best = (start, end, name)
    return best[2] if best else None


# --- frame helpers ------------------------------------------------------------
def _is_py(fr) -> bool:
    fn = fr.get("filename")
    return isinstance(fn, str) and fn.endswith(".py")


def _project_frames(frames, project_root):
    return [f for f in frames if _is_py(f) and os.path.abspath(f["filename"]).startswith(project_root)]


def _site_of(frames, project_root):
    """(filename, line, is_project): deepest project .py frame, else deepest .py
    frame, else deepest named frame."""
    pf = _project_frames(frames, project_root)
    if pf:
        return pf[0]["filename"], pf[0]["line"], True
    for f in frames:
        if _is_py(f):
            return f["filename"], f["line"], False
    for f in frames:
        if f.get("name"):
            return f.get("filename") or "??", f.get("line", 0), False
    return "??", 0, False


# --- timeline replay ----------------------------------------------------------
_ALLOC, _FREE = "alloc", {"free_requested", "free_completed"}


def _busiest_trace(snapshot):
    traces = snapshot["device_traces"]
    dev = max(range(len(traces)), key=lambda i: len(traces[i]))
    return dev, traces[dev]


def reconstruct_timeline(trace):
    """Replay -> series = [(event_idx, time_us, allocated_bytes)] at each alloc/free."""
    allocated, live, series = 0, {}, []
    for idx, e in enumerate(trace):
        act = e.get("action")
        if act == _ALLOC:
            allocated += e["size"]
            live[e["addr"]] = e["size"]
            series.append((idx, e.get("time_us", 0), allocated))
        elif act in _FREE and e.get("addr") in live:  # first free of this addr
            allocated -= live.pop(e["addr"])
            series.append((idx, e.get("time_us", 0), allocated))
    return series


def find_peaks(series, top_n, min_sep_us, dedup_pct):
    """Local maxima -> top-N *distinct* peaks (deduped by level, optional time gap)."""
    cands = [
        series[k]
        for k in range(len(series))
        if series[k][2] >= (series[k - 1][2] if k else -1)
        and series[k][2] > (series[k + 1][2] if k + 1 < len(series) else -1)
    ]
    cands.sort(key=lambda s: -s[2])
    selected = []
    for c in cands:
        if min_sep_us and any(abs(c[1] - s[1]) < min_sep_us for s in selected):
            continue
        if dedup_pct and any(abs(c[2] - s[2]) <= dedup_pct / 100.0 * s[2] for s in selected):
            continue
        selected.append(c)
        if len(selected) >= top_n:
            break
    return selected


def live_set_at(trace, peak_idx):
    live = {}
    for idx in range(peak_idx + 1):
        e = trace[idx]
        act = e.get("action")
        if act == _ALLOC:
            live[e["addr"]] = e
        elif act in _FREE and e.get("addr") in live:
            del live[e["addr"]]
    return list(live.values())


# --- attribution + markdown ---------------------------------------------------
def aggregate_by_site(live_events, project_root):
    groups: dict[tuple, dict] = {}
    for e in live_events:
        frames = e.get("frames") or []
        fn, line, is_proj = _site_of(frames, project_root)
        g = groups.setdefault(
            (fn, line),
            {
                "bytes": 0,
                "count": 0,
                "is_proj": is_proj,
                "filename": fn,
                "line": line,
                "chain": _project_frames(frames, project_root)[:5],
            },
        )
        g["bytes"] += e["size"]
        g["count"] += 1
    return sorted(groups.values(), key=lambda g: -g["bytes"])


def _rel(path, project_root):
    ap = os.path.abspath(path)
    return ap[len(project_root) + 1 :] if ap.startswith(project_root) else ap


def _link(path, line, args):
    ap = os.path.abspath(path)
    text = f"{_rel(ap, args.project_root)}:{line}"
    if not _is_py({"filename": path}):
        return f"`{text}`"
    if args.link_style == "github" and args.repo_url and ap.startswith(args.project_root):
        return f"[{text}]({args.repo_url}/blob/{args.commit}/{_rel(ap, args.project_root)}#L{line})"
    if args.link_style == "cursor":
        return f"[{text}](cursor://file{ap}:{line})"
    if args.link_style == "file":
        return f"[{text}](file://{ap})"
    if args.link_style == "none":
        return f"`{text}`"
    return f"[{text}](vscode://file{ap}:{line})"


def _site_label(g):
    if not g["is_proj"]:
        return "_(external)_"
    qn = _qualname_at(g["filename"], g["line"])
    return f"`{qn}`" if qn else ""


def _chain_str(g):
    parts = [_qualname_at(f["filename"], f["line"]) or os.path.basename(f["filename"]) for f in g["chain"]]
    return " ← ".join(parts) if parts else "—"


def emit_markdown(snapshot, dev, series, peaks, args) -> str:
    gpeak = max((s[2] for s in series), default=0)
    t0 = series[0][1] if series else 0
    out = [
        f"# CUDA memory peak analysis — `{os.path.basename(args.snapshot)}`\n",
        f"- snapshot: `{args.snapshot}` (device {dev})\n"
        f"- peak allocated (timeline): **{gpeak / _GiB:.2f} GiB** over {len(series):,} events\n"
        f"- top {len(peaks)} distinct peaks (levels merged within {args.dedup_pct:.0f}%); "
        f"links: `{args.link_style}`\n",
    ]
    for rank, (idx, t, alloc) in enumerate(peaks, 1):
        groups = aggregate_by_site(live_set_at(snapshot["device_traces"][dev], idx), args.project_root)
        tag = "  ← global peak" if alloc == gpeak else ""
        out.append(f"\n## Peak {rank} — {alloc / _GiB:.2f} GiB (t≈{(t - t0) / 1e6:.2f}s){tag}\n")
        out.append("| # | bytes | % peak | tensors | site (Class.method) | call chain |")
        out.append("|--:|------:|-------:|--------:|---------------------|------------|")
        for i, g in enumerate(groups[: args.top_contributors], 1):
            sz = g["bytes"]
            u = f"{sz / _GiB:.2f} GiB" if sz >= _GiB else f"{sz / _MiB:.0f} MiB"
            out.append(
                f"| {i} | {u} | {100 * sz / alloc:.1f}% | {g['count']} | "
                f"{_link(g['filename'], g['line'], args)} {_site_label(g)} | {_chain_str(g)} |"
            )
        rest = groups[args.top_contributors :]
        if rest:
            rb = sum(g["bytes"] for g in rest)
            out.append(
                f"| … | {rb / _GiB:.2f} GiB | {100 * rb / alloc:.1f}% | "
                f"{sum(g['count'] for g in rest)} | _{len(rest)} more sites_ | |"
            )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--top-contributors", type=int, default=15)
    ap.add_argument("--dedup-pct", type=float, default=3.0)
    ap.add_argument("--min-sep-ms", type=float, default=0.0)
    ap.add_argument(
        "--project-root",
        default=os.getcwd(),
        help="frames under this dir are 'project'",
    )
    ap.add_argument(
        "--link-style",
        choices=["vscode", "cursor", "github", "file", "none"],
        default="vscode",
    )
    ap.add_argument("--repo-url", default=None, help="base repo URL for --link-style github")
    ap.add_argument("--commit", default=None, help="git sha for github links (default: HEAD)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.project_root = os.path.abspath(args.project_root)
    if args.commit is None:
        try:
            args.commit = subprocess.check_output(
                ["git", "-C", args.project_root, "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:  # noqa: BLE001
            args.commit = "main"

    with open(args.snapshot, "rb") as f:
        snapshot = pickle.load(f)
    dev, trace = _busiest_trace(snapshot)
    series = reconstruct_timeline(trace)
    peaks = find_peaks(series, args.top_n, args.min_sep_ms * 1e3, args.dedup_pct)
    md = emit_markdown(snapshot, dev, series, peaks, args)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.snapshot)) or ".",
        f"MEM_PEAK_ANALYSIS_{os.path.splitext(os.path.basename(args.snapshot))[0]}.md",
    )
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"peak={max((s[2] for s in series), default=0) / _GiB:.2f} GiB; {len(peaks)} peaks; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
