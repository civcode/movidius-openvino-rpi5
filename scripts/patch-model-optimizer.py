#!/usr/bin/env python3
"""Apply deterministic compatibility edits to a staged OpenVINO 2020.3 MO tree."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_optional(path: Path, old: str, new: str) -> bool:
    """Replace one known legacy variant when present.

    OpenVINO 2020.3 existed in several packaging variants. Some copies of
    versions_checker.py contain dynamic packaging imports; the pinned source
    used by this repository does not. Absence of an optional variant is not an
    error.
    """

    text = path.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_versions_checker(path: Path) -> bool:
    if not path.exists():
        return False

    text = path.read_text(encoding="utf-8")
    # Sanity-check that this is still the expected OpenVINO MO module. The
    # exact compatibility snippets below are optional because not every 2020.3
    # source variant contains them.
    if "def check_requirements(" not in text:
        raise SystemExit(f"unexpected versions_checker.py layout: {path}")

    changed = False
    changed |= replace_optional(
        path,
        'exec("import platform,sys,packaging")',
        "import packaging, platform, sys",
    )
    changed |= replace_optional(
        path,
        'exec("del packaging,platform,sys")',
        "del packaging, platform, sys",
    )
    return changed


def patch_emitter(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if ".getchildren()" not in text:
        return False
    path.write_text(text.replace(".getchildren()", ""), encoding="utf-8")
    return True


def patch_tree(root: Path) -> list[str]:
    root = root.resolve()
    changed: list[str] = []

    versions = root / "mo" / "utils" / "versions_checker.py"
    if patch_versions_checker(versions):
        changed.append(str(versions.relative_to(root)))

    emitter = root / "mo" / "back" / "ie_ir_ver_2" / "emitter.py"
    if patch_emitter(emitter):
        changed.append(str(emitter.relative_to(root)))

    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mo_root")
    args = parser.parse_args()
    root = Path(args.mo_root).resolve()
    changed = patch_tree(root)
    print(
        "patched:" if changed else "already compatible:",
        ", ".join(changed) or root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
