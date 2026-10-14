#!/usr/bin/env python3
"""List the third-party packages the ICRL training scripts need, and which are missing here.

It reads the code (it does not run it): starting from the entry scripts it follows imports
into the copied code under algos/icrl and collects every third-party import it can reach.
So one run shows all missing packages, instead of one ModuleNotFoundError per attempt.

Usage (inside the venv you want to check):
    python scripts/check_icrl_imports.py [algos/icrl] [entry.py ...]

Default entries: interface/train_policy.py, interface/train_icrl.py, interface/train_gail.py
Needs Python 3.10+.
"""
import ast
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "algos/icrl").resolve()
ENTRIES = sys.argv[2:] or [
    "interface/train_policy.py",
    "interface/train_icrl.py",
    "interface/train_gail.py",
]

# import name -> pip name, where they differ
PIP_NAME = {
    "yaml": "pyyaml",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "git": "gitpython",
}

STDLIB = set(sys.stdlib_module_names)
LOCAL = {p.name for p in ROOT.iterdir() if p.is_dir()} | {p.stem for p in ROOT.glob("*.py")}


def find_module_files(parts):
    """Files Python would execute for `import a.b.c` under ROOT (package inits + the module)."""
    files = []
    cur = ROOT
    for i, part in enumerate(parts):
        init = cur / part / "__init__.py"
        mod = cur / (part + ".py")
        last = i == len(parts) - 1
        if mod.is_file() and last:
            files.append(mod)
            break
        if init.is_file():
            files.append(init)
        elif not (cur / part).is_dir():
            break
        cur = cur / part
    return files


class Scanner(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.rel = path.relative_to(ROOT)
        self.optional_depth = 0
        self.local_files = []
        self.third = []          # (top, lineno, optional)
        self.unresolved = []     # (module, lineno)

    def visit_Try(self, node):
        catches = False
        for h in node.handlers:
            names = []
            if h.type is None:
                catches = True
            elif isinstance(h.type, ast.Tuple):
                names = [getattr(e, "id", "") for e in h.type.elts]
            else:
                names = [getattr(h.type, "id", "")]
            if {"ImportError", "ModuleNotFoundError", "Exception"} & set(names):
                catches = True
        if catches:
            self.optional_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        if catches:
            self.optional_depth -= 1
        for part in node.handlers + node.orelse + node.finalbody:
            self.visit(part)

    def visit_Import(self, node):
        for alias in node.names:
            self._absolute(alias.name, [], node.lineno)

    def visit_ImportFrom(self, node):
        names = [a.name for a in node.names if a.name != "*"]
        if node.level == 0:
            self._absolute(node.module or "", names, node.lineno)
        else:
            base = self.path.parent
            for _ in range(node.level - 1):
                base = base.parent
            rel_parts = (node.module or "").split(".") if node.module else []
            target = base.joinpath(*rel_parts) if rel_parts else base
            found = False
            for cand in (target.with_suffix(".py"), target / "__init__.py"):
                if cand.is_file():
                    self.local_files.append(cand)
                    found = True
            for n in names:
                for cand in (target / (n + ".py"), target / n / "__init__.py"):
                    if cand.is_file():
                        self.local_files.append(cand)
                        found = True
            if not found and not names:
                self.unresolved.append(("." * node.level + (node.module or ""), node.lineno))

    def _absolute(self, module, names, lineno):
        parts = module.split(".")
        top = parts[0]
        if top in STDLIB:
            return
        if top in LOCAL:
            files = find_module_files(parts)
            for n in names:
                files += [f for f in find_module_files(parts + [n]) if f not in files]
            if files:
                self.local_files += files
            else:
                self.unresolved.append((module, lineno))
        else:
            self.third.append((top, lineno, self.optional_depth > 0))


def main():
    queue = [ROOT / e for e in ENTRIES]
    for q in queue:
        if not q.is_file():
            print(f"entry not found: {q}", file=sys.stderr)
    seen, third, unresolved = set(), defaultdict(list), []
    while queue:
        f = queue.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            print(f"cannot parse {f.relative_to(ROOT)}: {e}", file=sys.stderr)
            continue
        sc = Scanner(f)
        sc.visit(tree)
        queue += [p for p in sc.local_files if p not in seen]
        for top, lineno, opt in sc.third:
            third[top].append((str(sc.rel), lineno, opt))
        for mod, lineno in sc.unresolved:
            unresolved.append((str(sc.rel), lineno, mod))

    missing, present = {}, []
    for top, locs in third.items():
        try:
            installed = importlib.util.find_spec(top) is not None
        except (ImportError, ValueError):
            installed = False
        if installed:
            present.append(top)
        else:
            missing[top] = locs

    print(f"python: {sys.version.split()[0]}  ({sys.prefix})")
    print(f"scanned {len(seen)} files reachable from {', '.join(ENTRIES)}\n")

    def show(top, locs):
        required = [l for l in locs if not l[2]]
        tag = "" if required else "  (only in try/except: probably optional)"
        shown = ", ".join(f"{p}:{n}" for p, n, _ in locs[:3])
        more = f" (+{len(locs) - 3} more)" if len(locs) > 3 else ""
        pip = PIP_NAME.get(top, top)
        print(f"  {top:<22} pip: {pip:<18}{tag}\n      used in {shown}{more}")

    if missing:
        print(f"MISSING in this environment ({len(missing)}):")
        for top in sorted(missing):
            show(top, missing[top])
    else:
        print("MISSING in this environment: none")
    print(f"\nINSTALLED ({len(present)}): {', '.join(sorted(set(present)))}")
    if unresolved:
        print(f"\nLOCAL imports that point at files not in {ROOT.name} ({len(unresolved)}):")
        for p, n, m in sorted(set(unresolved))[:15]:
            print(f"  {p}:{n}  {m}")
        print("  (e.g. the excluded commonroad_environment; check whether each is on the "
              "path you actually use)")


if __name__ == "__main__":
    main()