#!/usr/bin/env python3
"""Run the staged Model Optimizer 2020.3 with modern numpy (>=1.24).

MO 2020.3 predates numpy 1.24, so it references the removed aliases
np.float / np.int / np.bool / np.str / np.object / np.unicode. This
shim restores the old semantics (they were exact aliases of the
corresponding built-in types) before running mo_tf.py unchanged.

Usage: python3 scripts/mo_compat_run.py [mo_tf.py arguments ...]
"""
import io
import os
import runpy
import sys

import numpy as np

# numpy 1.24 removed these deprecated aliases; restore the old meanings.
for alias, value in (
    ("float", float),
    ("int", int),
    ("bool", bool),
    ("object", object),
    ("str", str),
    ("unicode", str),
):
    if alias not in vars(np):
        setattr(np, alias, value)

def patch_staged_tree(mo_root):
    """Idempotent Python-3.13 compatibility patches for the staged MO 2020.3
    tree (the vendored source is never touched).  MO 2020.3 predates modern
    interpreters in two places:

    - mo/utils/versions_checker.py wraps its imports in exec("import ..."),
      which no longer binds names in the caller's scope on Python >= 3.13.
    - mo/back/ie_ir_ver_2/emitter.py calls Element.getchildren(), removed in
      Python 3.9; iterating the element (or len() of it) is equivalent.
    """
    v = os.path.join(mo_root, "mo", "utils", "versions_checker.py")
    if os.path.exists(v):
        t = io.open(v).read()
        if "exec(\"import platform,sys,packaging\")" in t:
            t = t.replace('exec("import platform,sys,packaging")',
                          "import packaging, platform, sys", 1)
            t = t.replace('exec("del packaging,platform,sys")',
                          "del packaging, platform, sys", 1)
            io.open(v, "w").write(t)
            print("mo_compat_run: patched mo/utils/versions_checker.py (py3.13)")
    e = os.path.join(mo_root, "mo", "back", "ie_ir_ver_2", "emitter.py")
    if os.path.exists(e):
        t = io.open(e).read()
        if ".getchildren()" in t:
            io.open(e, "w").write(t.replace(".getchildren()", ""))
            print("mo_compat_run: patched mo/back/ie_ir_ver_2/emitter.py (py3.9+)")


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mo_tf = os.path.join(root, "work", "mo-2020.3", "mo_tf.py")
if not os.path.exists(mo_tf):
    sys.exit(f"staged Model Optimizer not found at {mo_tf} - run the prepare script staging step first")
patch_staged_tree(os.path.dirname(mo_tf))
sys.path.insert(0, os.path.dirname(mo_tf))
sys.argv = [mo_tf] + sys.argv[1:]
runpy.run_path(mo_tf, run_name="__main__")
