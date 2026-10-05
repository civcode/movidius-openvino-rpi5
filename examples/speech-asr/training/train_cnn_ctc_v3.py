#!/usr/bin/env python3
"""Train or initialize Phase 11 generation-2 cnn_ctc_v3."""

from __future__ import annotations

import argparse
import copy
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

from cnn_ctc_v3 import (  # noqa: E402
    CnnCtcV3,
    ManifestCtcDataset,
    collate_ctc,
    parameter_count,
)
from speech_asr.cnn_ctc import canonical_sha256, load_spec, load_vocab  # noqa: E402

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v3" / "vocab.json"
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
            raise ValueError(
                "--device cuda requested but torch.cuda.is_available() is false"
            )
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


def deterministic_ctc_loss(loss_fn, logits, batch):
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    if log_probs.device.type == "cuda":
        log_probs = log_probs.cpu()
    return loss_fn(
        log_probs,
        batch["targets"],
        batch["input_lengths"],
        batch["target_lengths"],
    )


def evaluate_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            logits = model(features)
            loss = deterministic_ctc_loss(loss_fn, logits, batch)
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
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v3" / "training",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec)
        vocab = load_vocab(args.vocab)
        seed = args.seed if args.seed is not None else int(spec["training"]["seed"])
        if seed < 0:
            raise ValueError("seed must be >= 0")
        set_deterministic(seed)
        device = choose_device(args.device)

        batch_size = args.batch_size or int(spec["training"]["batch_size"])
        epochs = (
            args.epochs
            if args.epochs is not None
            else int(spec["training"]["epochs"])
        )
        learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else float(spec["training"]["learning_rate"])
        )
        gradient_clip_norm = float(spec["training"]["gradient_clip_norm"])
        min_learning_rate = float(spec["training"]["min_learning_rate"])
        if epochs < 0 or batch_size < 1 or learning_rate <= 0:
            raise ValueError(
                "epochs must be >= 0, batch size >= 1 and learning rate > 0"
            )
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0")
        if not 0 <= min_learning_rate <= learning_rate:
            raise ValueError(
                "min_learning_rate must be between 0 and learning_rate"
            )
        if spec["training"].get("lr_schedule") != "cosine":
            raise ValueError("cnn_ctc_v3 lr_schedule must be cosine")
        if spec["training"].get("checkpoint_selection") != "best_validation_loss":
            raise ValueError(
                "cnn_ctc_v3 checkpoint_selection must be best_validation_loss"
            )

        train_dataset = None
        validation_dataset = None
        train_loader = None
        validation_loader = None
        validation_manifest = None
        if not args.init_only:
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

        model = CnnCtcV3(spec, len(vocab["tokens"])).to(device)
        loss_fn = nn.CTCLoss(
            blank=int(vocab["blank_index"]),
            zero_infinity=bool(spec["training"]["zero_infinity"]),
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=min_learning_rate,
        )

        history = []
        optimizer_steps = 0
        best_epoch = None
        best_validation_loss = None
        best_state_dict = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        start = time.monotonic()

        if not args.init_only:
            assert train_loader is not None
            assert validation_loader is not None
            for epoch in range(epochs):
                model.train()
                batch_losses = []
                gradient_norms = []
                current_lr = float(optimizer.param_groups[0]["lr"])
                for batch in train_loader:
                    features = batch["features"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    loss = deterministic_ctc_loss(loss_fn, logits, batch)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                    optimizer.step()
                    optimizer_steps += 1
                    batch_losses.append(float(loss.detach().cpu()))
                    gradient_norms.append(float(gradient_norm.detach().cpu()))

                validation_loss = evaluate_loss(
                    model,
                    validation_loader,
                    loss_fn,
                    device,
                )
                mean_train_loss = float(
                    sum(batch_losses) / max(1, len(batch_losses))
                )
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": mean_train_loss,
                        "validation_loss": validation_loss,
                        "learning_rate": current_lr,
                        "optimizer_steps": optimizer_steps,
                        "max_gradient_norm_before_clip": (
                            max(gradient_norms) if gradient_norms else 0.0
                        ),
                    }
                )

                if (
                    best_validation_loss is None
                    or validation_loss < best_validation_loss
                ):
                    best_validation_loss = validation_loss
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                scheduler.step()

            model.load_state_dict(best_state_dict)
            optimizer.load_state_dict(best_optimizer_state)

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
                "optimizer_steps": optimizer_steps,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
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
            "ctc_loss_device": "cpu" if device.type == "cuda" else str(device),
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
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "lr_schedule": "cosine",
            "gradient_clip_norm": gradient_clip_norm,
            "checkpoint_selection": "best_validation_loss",
            "parameter_count": parameter_count(model),
            "optimizer_steps": optimizer_steps,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "train_samples": 0 if train_dataset is None else len(train_dataset),
            "validation_samples": (
                0 if validation_dataset is None else len(validation_dataset)
            ),
            "skipped": {
                "train_too_long": (
                    [] if train_dataset is None else list(train_dataset.skipped_long)
                ),
                "train_target_too_long": (
                    [] if train_dataset is None else list(train_dataset.skipped_target)
                ),
                "validation_too_long": (
                    [] if validation_dataset is None else list(validation_dataset.skipped_long)
                ),
                "validation_target_too_long": (
                    [] if validation_dataset is None else list(validation_dataset.skipped_target)
                ),
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
                "manifest": None if args.init_only else str(args.manifest),
                "validation_manifest": (
                    None if validation_manifest is None else str(validation_manifest)
                ),
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
