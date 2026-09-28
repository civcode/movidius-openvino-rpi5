#!/usr/bin/env python3
from pathlib import Path
import os, runpy, sys
import numpy as np
for alias,value in (("float",float),("int",int),("bool",bool),("object",object),("str",str),("unicode",str)):
    if alias not in vars(np): setattr(np,alias,value)
root=Path(os.environ.get("OV_STANDALONE_ROOT",Path(__file__).resolve().parents[1])).resolve()
mo=root/"tools/model-optimizer"
if not (mo/"mo.py").exists(): raise SystemExit(f"Model Optimizer is missing from {mo}")
sys.path.insert(0,str(mo)); sys.argv=[str(mo/"mo.py")]+sys.argv[1:]
runpy.run_path(str(mo/"mo.py"),run_name="__main__")
