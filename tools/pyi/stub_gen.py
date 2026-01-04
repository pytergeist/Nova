#!/usr/bin/env python3
"""
Generate .pyi stubs for the fused C++ extension using pybind11-stubgen.

Usage (from repo root):
    poetry run python tools/pyi/stub_gen.py
    # or with explicit args:
    poetry run python tools/pyi/stub_gen.py \
        --module nova.src.backend.core.clib.fusion \
        --dest nova/src/backend/core/clib
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = (
        Path(__file__).resolve().parents[2]
    )  # tools/pyi/stub_gen.py -> repo root
    default_dest = repo_root / "nova" / "src" / "backend" / "core" / "clib"

    p = argparse.ArgumentParser()
    p.add_argument(
        "--module",
        default="nova.src.backend.core.clib.fusion",
        help="Fully-qualified module name to stubgen.",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=default_dest,
        help="Destination directory where .pyi files are placed.",
    )
    p.add_argument(
        "--tmpdir",
        type=Path,
        default=repo_root / "build" / "pyi_tmp",
        help="Temporary output dir used by stubgen.",
    )
    p.add_argument("--keep-tmp", action="store_true", help="Keep temporary directory.")
    p.add_argument(
        "--skip-import-check",
        action="store_true",
        help="Skip importing the module before running stubgen.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p.parse_args()


def check_module_import(modname: str) -> None:
    try:
        m = importlib.import_module(modname)
        print(
            f"[gen.py] Imported '{modname}' from: {getattr(m, '__file__', '<unknown>')}"
        )
    except Exception as e:
        print(
            f"[gen.py] ERROR: failed to import '{modname}'.\n"
            f"         Make sure the extension is built for this interpreter "
            f"         and that PYTHONPATH includes your package.\n"
            f"         ({e})",
            file=sys.stderr,
        )
        sys.exit(2)


def run_stubgen(modname: str, outdir: Path, verbose: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pybind11_stubgen",
        modname,
        "--output-dir",
        str(outdir),
    ]
    print(f"[gen.py] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if verbose:
        print_dir_tree(outdir, prefix="[gen.py] tmp ")


def print_dir_tree(root: Path, depth: int = 4, prefix: str = "") -> None:
    try:

        def _walk(p: Path, d: int):
            if d < 0:
                return
            for child in sorted(p.iterdir()):
                print(f"{prefix}{child.relative_to(root)}")
                if child.is_dir():
                    _walk(child, d - 1)

        print(f"{prefix}tree of {root}:")
        if root.exists():
            _walk(root, depth)
        else:
            print(f"{prefix}<missing>")
    except Exception:
        pass


def find_stub_outputs(tmpdir: Path, module: str) -> list[Path]:
    """
    Locate generated stubs for a dotted module. We check (in order):

    1) <tmpdir>/<module_as_path>/<leaf>.pyi
    2) <tmpdir>/<module_as_path>/<leaf>/  (package with __init__.pyi)
    3) Fallback: any '**/<leaf>.pyi' or '**/<leaf>/__init__.pyi' under tmpdir
    """
    parts = module.split(".")
    leaf = parts[-1]
    base_dir = tmpdir.joinpath(*parts[:-1])  # dotted path minus the leaf

    candidates: list[Path] = []
    # 1) Exact expected file location for dotted module
    p1 = base_dir / f"{leaf}.pyi"
    if p1.exists():
        candidates.append(p1)

    # 2) Exact expected package dir
    p2 = base_dir / leaf
    if p2.exists():
        # Prefer package dir if it has __init__.pyi
        init_pyi = p2 / "__init__.pyi"
        if init_pyi.exists():
            candidates.append(p2)
        else:
            # still keep the directory (some versions split files inside)
            candidates.append(p2)

    if candidates:
        return candidates

    # 3) Fallback glob search
    candidates.extend(tmpdir.rglob(f"{leaf}.pyi"))
    candidates.extend(p.parent for p in tmpdir.rglob(f"{leaf}/__init__.pyi"))

    # De-duplicate while preserving order
    uniq: list[Path] = []
    seen = set()
    for c in candidates:
        key = c.resolve()
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def copy_results(found: list[Path], dest: Path, leaf_name: str) -> None:
    if not found:
        print(
            f"[gen.py] ERROR: no stub outputs found in {dest.parent}", file=sys.stderr
        )
        sys.exit(3)

    dest.mkdir(parents=True, exist_ok=True)

    # Clean old stubs named 'leaf.pyi' or 'leaf/' under dest
    old_file = dest / f"{leaf_name}.pyi"
    old_pkg = dest / leaf_name
    for path in (old_file, old_pkg):
        if path.exists():
            print(f"[gen.py] Removing old {path}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    # Copy new ones
    for src in found:
        if src.is_dir():
            print(f"[gen.py] Copying dir {src} -> {dest / src.name}")
            shutil.copytree(src, dest / src.name)
        else:
            print(f"[gen.py] Copying file {src} -> {dest / src.name}")
            shutil.copy2(src, dest / src.name)


def ensure_py_typed(repo_root: Path) -> None:
    py_typed = repo_root / "nova" / "py.typed"
    if not py_typed.exists():
        print(f"[gen.py] Creating {py_typed}")
        py_typed.parent.mkdir(parents=True, exist_ok=True)
        py_typed.write_text("")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    leaf_name = args.module.split(".")[-1]

    print(f"[gen.py] Repo root: {repo_root}")
    print(f"[gen.py] Module   : {args.module}")
    print(f"[gen.py] Dest dir : {args.dest}")
    print(f"[gen.py] Temp dir : {args.tmpdir}")

    if not args.skip_import_check:
        check_module_import(args.module)

    if args.tmpdir.exists():
        print(f"[gen.py] Cleaning tmp dir {args.tmpdir}")
        shutil.rmtree(args.tmpdir, ignore_errors=True)

    run_stubgen(args.module, args.tmpdir, args.verbose)

    # find artifacts under nested module path
    found = find_stub_outputs(args.tmpdir, args.module)
    if not found:
        print_dir_tree(args.tmpdir, prefix="[gen.py] tmp ", depth=8)
        print(
            f"[gen.py] ERROR: no stub outputs found in {args.tmpdir} "
            f"for module '{args.module}'.",
            file=sys.stderr,
        )
        sys.exit(3)

    copy_results(found, args.dest, leaf_name)
    ensure_py_typed(repo_root)

    if not args.keep_tmp:
        print(f"[gen.py] Removing tmp dir {args.tmpdir}")
        shutil.rmtree(args.tmpdir, ignore_errors=True)

    print("[gen.py] Done.")


if __name__ == "__main__":
    main()
