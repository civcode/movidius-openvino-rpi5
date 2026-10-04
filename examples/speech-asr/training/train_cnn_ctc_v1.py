#!/usr/bin/env python3
"""Train or initialize the cnn_ctc_v1 PyTorch skeleton."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))
sys.path.insert(0, str(HERE))

from cnn_ctc_v1 import CnnCtcV1, ManifestCtcDataset, collate_ctc  # noqa: E402
from speech_asr.cnn_ctc import canonical_sha256, load_spec, load_vocab  # noqa: E402

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v1" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v1" / "vocab.json"
DEFAULT_MANIFEST = ROOT / "work" / "speech-asr" / "ami" / "ami-smoke-v1" / "manifest.jsonl"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.strip()


def choose_device(request: str) -> torch.device:
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unsupported device request: {request}")


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            logits = model(features)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = loss_fn(
                log_probs,
                batch["targets"].to(device),
                batch["input_lengths"],
                batch["target_lengths"],
            )
            losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(1, len(losses)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-manifest", type=pathlib.Path)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v1" / "training",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        seed = int(spec["training"]["seed"])
        set_deterministic(seed)
        device = choose_device(args.device)

        train_dataset = ManifestCtcDataset(
            args.manifest,
            spec,
            vocab,
            max_samples=args.max_samples,
        )
        validation_manifest = args.validation_manifest or args.manifest
        validation_dataset = ManifestCtcDataset(
            validation_manifest,
            spec,
            vocab,
            max_samples=args.max_samples,
        )

        batch_size = args.batch_size or int(spec["training"]["batch_size"])
        epochs = args.epochs if args.epochs is not None else int(spec["training"]["epochs"])
        if epochs < 0 or batch_size < 1:
            raise ValueError("epochs must be >= 0 and batch size must be >= 1")

        generator = torch.Generator()
        generator.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            collate_fn=collate_ctc,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_ctc,
        )

        model = CnnCtcV1(spec, len(vocab["tokens"])).to(device)
        loss_fn = nn.CTCLoss(
            blank=int(vocab["blank_index"]),
            zero_infinity=bool(spec["training"]["zero_infinity"]),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(spec["training"]["learning_rate"]),
        )

        history = []
        start = time.monotonic()
        if not args.init_only:
            for epoch in range(epochs):
                model.train()
                batch_losses = []
                for batch in train_loader:
                    features = batch["features"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
                    loss = loss_fn(
                        log_probs,
                        batch["targets"].to(device),
                        batch["input_lengths"],
                        batch["target_lengths"],
                    )
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(float(loss.detach().cpu()))
                validation_loss = evaluate_loss(
                    model,
                    validation_loader,
                    loss_fn,
                    device,
                )
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(sum(batch_losses) / max(1, len(batch_losses))),
                        "validation_loss": validation_loss,
                    }
                )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output_dir / "checkpoint.pt"
        torch.save(
            {
                "format": "speech-asr/cnn-ctc-checkpoint-v1",
                "model_id": spec["id"],
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "seed": seed,
                "spec_sha256": canonical_sha256(spec),
                "vocab_sha256": canonical_sha256(vocab),
            },
            checkpoint,
        )

        result = {
            "schema": "speech-asr/training-result",
            "version": 1,
            "status": "completed",
            "model_id": spec["id"],
            "init_only": args.init_only,
            "device_requested": args.device,
            "device_used": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "seed": seed,
            "epochs": 0 if args.init_only else epochs,
            "batch_size": batch_size,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "skipped": {
                "train_too_long": list(train_dataset.skipped_long),
                "train_target_too_long": list(train_dataset.skipped_target),
                "validation_too_long": list(validation_dataset.skipped_long),
                "validation_target_too_long": list(validation_dataset.skipped_target),
            },
            "history": history,
            "duration_seconds": time.monotonic() - start,
            "artifacts": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_path(checkpoint),
                "spec_sha256": canonical_sha256(spec),
                "vocab_sha256": canonical_sha256(vocab),
            },
            "provenance": {
                "repo_commit": git_head(),
                "manifest": str(args.manifest),
                "validation_manifest": str(validation_manifest),
            },
        }
        (args.output_dir / "training-result.json").write_text(
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
