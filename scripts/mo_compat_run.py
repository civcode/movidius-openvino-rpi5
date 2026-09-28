#!/usr/bin/env python3
"""Relocatable OpenVINO 2020.3 Model Optimizer launcher."""
from pathlib import Path
import os, runpy, sys
import numpy as np

for alias, value in (("float", float), ("int", int), ("bool", bool),
                     ("object", object), ("str", str), ("unicode", str)):
    if alias not in vars(np):
        setattr(np, alias, value)

here = Path(__file__).resolve()
repo = here.parents[1]
candidates = []
if os.environ.get("OV203_MO_ROOT"):
    candidates.append(Path(os.environ["OV203_MO_ROOT"]))
if os.environ.get("OV_STANDALONE_ROOT"):
    candidates.append(Path(os.environ["OV_STANDALONE_ROOT"]) / "tools/model-optimizer")
candidates += [repo / "work/model-optimizer"]
mo_root = next((p for p in candidates if (p / "mo.py").exists()), None)
if mo_root is None:
    raise SystemExit("staged Model Optimizer not found; run scripts/prepare-model-optimizer.sh")
sys.path.insert(0, str(mo_root))
sys.argv = [str(mo_root / "mo.py")] + sys.argv[1:]
runpy.run_path(str(mo_root / "mo.py"), run_name="__main__")
