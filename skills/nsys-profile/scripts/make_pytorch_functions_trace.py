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

"""Build a custom nsys ``--python-functions-trace`` JSON that names your model's
high-level modules (TriMul / Pairformer / trunk / diffusion / ...).

nsys ``--pytorch=functions-trace`` uses a predefined ``pytorch.json`` covering
only ``torch.*`` (generic ``Module.__call__``, ``Linear.forward``, ...). It will
NOT name your model's own classes. This script merges those torch.* entries with
entries for *your* module forwards, so passing the result via
``--python-functions-trace=<out.json>`` annotates the full module hierarchy with
no in-model NVTX.

The JSON schema (a list of groups) matches nsys's predefined file:
    [ {"domain": "<name>", "module": "<import.path>",
       "functions": ["ClassName.forward", "ClassName.some_method", ...]} ]

Spec file (``--spec``) maps each import path -> list of "ClassName.method":
    { "pkg.model.layers.trimul": ["TriMulDistributed.forward"],
      "pkg.model.diffusion":     ["DiffusionModule.forward",
                                   "DiffusionStructureHead.sample"] }

Usage:
    python make_pytorch_functions_trace.py --spec spec.json --out merged.json \
        [--base /path/to/nsys/.../PythonFunctionsTrace/pytorch.json] \
        [--domain MyModel] [--shim projects.huggingface.transformers=transformers]

``--base`` is auto-discovered from the ``nsys`` on PATH if omitted.
``--shim ALIAS=REAL`` installs an import alias before importing your modules (for
vendored code that imports itself under a different root); repeatable.
Validation: each ``module`` is imported and each ``Class.method`` checked; missing
ones are skipped with a warning so the emitted JSON is always loadable by nsys.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import os
import shutil
import sys
import types


def _install_shim(alias: str, real: str) -> None:
    """Make ``alias[.x]`` resolve to the same module objects as ``real[.x]``."""

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            name = spec.name
            # synthesize pure-namespace parents that have no real counterpart
            for i in range(len(alias.split("."))):
                parent = ".".join(alias.split(".")[: i + 1])
                if name == parent and name != alias:
                    m = types.ModuleType(name)
                    m.__path__ = []
                    return m
            return importlib.import_module(real + name[len(alias) :])

        def exec_module(self, module):
            return None

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            parents = [".".join(alias.split(".")[: i + 1]) for i in range(len(alias.split(".")))]
            if fullname in parents or fullname == alias or fullname.startswith(alias + "."):
                return importlib.util.spec_from_loader(fullname, _Loader())
            return None

    sys.meta_path.insert(0, _Finder())


def _autodiscover_base() -> str | None:
    exe = shutil.which("nsys")
    if not exe:
        return None
    # nsys lives in .../target-linux-x64/nsys; the json sits beside it.
    cand = os.path.join(os.path.dirname(os.path.realpath(exe)), "PythonFunctionsTrace", "pytorch.json")
    return cand if os.path.isfile(cand) else None


def build(base_json: str, spec: dict, domain: str) -> list:
    entries: list = []
    if base_json and os.path.isfile(base_json):
        with open(base_json) as fh:
            entries = list(json.load(fh))
    if not entries:
        print(
            f"WARNING: no/invalid --base ({base_json}); emitting model entries only",
            file=sys.stderr,
        )
    for mod_path, fns in spec.items():
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:  # noqa: BLE001
            print(
                f"  skip module (import failed): {mod_path}: {repr(e)[:160]}",
                file=sys.stderr,
            )
            continue
        valid = []
        for fn in fns:
            cls_name, _, meth = fn.partition(".")
            cls = getattr(mod, cls_name, None)
            if cls is not None and meth and hasattr(cls, meth):
                valid.append(fn)
            else:
                print(f"  skip (not found): {mod_path}.{fn}", file=sys.stderr)
        if valid:
            entries.append({"domain": domain, "module": mod_path, "functions": valid})
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="JSON: {import_path: [Class.method, ...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="predefined pytorch.json (auto-found if omitted)")
    ap.add_argument("--domain", default="Model", help="NVTX domain for appended entries")
    ap.add_argument(
        "--shim",
        action="append",
        default=[],
        help="ALIAS=REAL import alias (repeatable)",
    )
    args = ap.parse_args()

    for s in args.shim:
        alias, _, real = s.partition("=")
        _install_shim(alias, real)

    base = args.base or _autodiscover_base()
    with open(args.spec) as fh:
        spec = json.load(fh)
    entries = build(base, spec, args.domain)
    with open(args.out, "w") as fh:
        json.dump(entries, fh, indent=2)
    n_model = sum(1 for e in entries if e.get("domain") == args.domain)
    print(f"wrote {args.out}: {len(entries)} groups ({n_model} model groups, domain={args.domain})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
