#!/usr/bin/env python3
"""Download and verify the pinned QuartzNet15x5Base-En NeMo archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request

SOURCE_NAME = "QuartzNet15x5-En-Base.nemo"
SOURCE_SIZE = 71083664
SOURCE_SHA384 = "74e8284e77098906afb7a15a861ef60ec14db1a4acb206fa719492fa43050ad69a91c245652c05c5f0ded38b5903ed55"
URLS = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/quartznet-15x5-en/QuartzNet15x5-En-Base.nemo",
    "https://api.ngc.nvidia.com/v2/models/nvidia/multidataset_quartznet15x5/versions/2/files/QuartzNet15x5-En-Base.nemo",
)
REQUIRED_MEMBERS = (
    ".nemo_tmp/module.yaml",
    ".nemo_tmp/JasperEncoder.pt",
    ".nemo_tmp/JasperDecoderForCTC.pt",
)


def sha384(path: pathlib.Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise ValueError(f"pretrained archive missing: {path}")
    size = path.stat().st_size
    if size != SOURCE_SIZE:
        raise ValueError(f"pretrained archive size mismatch: {size} != {SOURCE_SIZE}")
    digest = sha384(path)
    if digest != SOURCE_SHA384:
        raise ValueError(f"pretrained archive SHA-384 mismatch: {digest}")
    with tarfile.open(path, mode="r:gz") as archive:
        names = set(archive.getnames())
    missing = [name for name in REQUIRED_MEMBERS if name not in names]
    if missing:
        raise ValueError(f"pretrained archive is missing members: {missing}")
    return {
        "schema": "speech-asr/pretrained-source",
        "version": 1,
        "status": "valid",
        "source_name": SOURCE_NAME,
        "path": str(path),
        "size_bytes": size,
        "sha384": digest,
        "required_members": list(REQUIRED_MEMBERS),
        "license": "Apache-2.0",
        "provenance": "NVIDIA NGC artifact pinned by Open Model Zoo quartznet-15x5-en",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.is_file() and not args.verify_only:
            last_error = None
            for url in URLS:
                temp = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=args.output.parent,
                        prefix=SOURCE_NAME + ".",
                        suffix=".part",
                        delete=False,
                    ) as tmp:
                        temp = pathlib.Path(tmp.name)
                        with urllib.request.urlopen(url, timeout=120) as response:
                            shutil.copyfileobj(response, tmp, length=1024 * 1024)
                    verify(temp)
                    temp.replace(args.output)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if temp is not None:
                        temp.unlink(missing_ok=True)
            if last_error is not None:
                raise last_error
        result = verify(args.output)
        manifest = args.output.parent / "source.json"
        manifest.write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
