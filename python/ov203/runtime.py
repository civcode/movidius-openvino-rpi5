"""Path discovery shared by source-tree and standalone execution."""
from pathlib import Path
import os

def standalone_root():
    value=os.environ.get("OV_STANDALONE_ROOT")
    return Path(value).expanduser().resolve() if value else None

def source_repo_root(start=None):
    p=Path(start or __file__).resolve()
    if p.is_file(): p=p.parent
    for candidate in (p,*p.parents):
        if (candidate/"scripts"/"platform.sh").exists() and (candidate/"examples").is_dir():
            return candidate
    return None

def models_root(start=None):
    value=os.environ.get("OV_MODELS_DIR")
    if value: return Path(value).expanduser().resolve()
    root=standalone_root()
    if root: return root/"models"
    repo=source_repo_root(start)
    if repo: return repo/"vendor"/"models"
    raise RuntimeError("cannot locate model root; set OV_MODELS_DIR")
