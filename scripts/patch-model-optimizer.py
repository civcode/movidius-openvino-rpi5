#!/usr/bin/env python3
"""Apply deterministic compatibility edits to a staged OpenVINO 2020.3 MO tree."""
from pathlib import Path
import argparse

def replace_once(path, old, new):
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise SystemExit(f"expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1))
    return True

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mo_root"); args=ap.parse_args()
    root=Path(args.mo_root).resolve(); changed=[]
    v=root/"mo/utils/versions_checker.py"
    if v.exists():
        if replace_once(v, 'exec("import platform,sys,packaging")', 'import packaging, platform, sys'): changed.append(str(v.relative_to(root)))
        if replace_once(v, 'exec("del packaging,platform,sys")', 'del packaging, platform, sys'): changed.append(str(v.relative_to(root)))
    e=root/"mo/back/ie_ir_ver_2/emitter.py"
    if e.exists() and ".getchildren()" in e.read_text():
        e.write_text(e.read_text().replace(".getchildren()", "")); changed.append(str(e.relative_to(root)))
    print("patched:" if changed else "already compatible:", ", ".join(changed) or root)
if __name__=="__main__": main()
