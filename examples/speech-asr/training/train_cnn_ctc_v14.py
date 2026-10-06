#!/usr/bin/env python3
"""Train or initialize the cnn_ctc_v14 architecture-screen candidate."""

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

from cnn_ctc_v14 import (  # noqa: E402
    CnnCtcV14,
    ManifestCtcDataset,
    collate_ctc,
    parameter_count,
)
from speech_asr.cnn_ctc import (  # noqa: E402
    aggregate_decoder_diagnostics,
    canonical_sha256,
    greedy_decode_logits_diagnostics,
    load_spec,
    load_vocab,
)
from speech_asr.evaluation import (  # noqa: E402
    character_error_counts,
    word_error_counts,
)

DEFAULT_SPEC = SPEECH_ROOT / "models" / "cnn_ctc_v14" / "model_spec.json"
DEFAULT_VOCAB = SPEECH_ROOT / "models" / "cnn_ctc_v14" / "vocab.json"
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


def deterministic_ctc_loss(
    loss_fn,
    logits,
    batch,
    *,
    blank_index: int = 0,
    blank_logit_penalty: float = 0.0,
):
    if blank_logit_penalty < 0:
        raise ValueError("blank_logit_penalty must be >= 0")
    adjusted_logits = logits
    if blank_logit_penalty:
        adjusted_logits = logits.clone()
        adjusted_logits[..., blank_index] = (
            adjusted_logits[..., blank_index] - blank_logit_penalty
        )
    log_probs = adjusted_logits.log_softmax(dim=-1).transpose(0, 1)
    if log_probs.device.type == "cuda":
        log_probs = log_probs.cpu()
    return loss_fn(
        log_probs,
        batch["targets"],
        batch["input_lengths"],
        batch["target_lengths"],
    )


def dataset_audio_samples(dataset: ManifestCtcDataset | None) -> int:
    if dataset is None:
        return 0
    return sum(
        int(record["audio"]["end_sample"]) - int(record["audio"]["start_sample"])
        for record in dataset.records
    )


def _sum_rate(counts) -> float:
    errors = sum(value.errors for value in counts)
    references = sum(value.reference_units for value in counts)
    return errors / max(1, references)


def apply_specaugment_v1(
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    *,
    generator: torch.Generator,
    policy: dict,
) -> tuple[torch.Tensor, dict]:
    if features.ndim != 3:
        raise ValueError(
            f"SpecAugment expects [N,F,T] features, got {tuple(features.shape)}"
        )
    if feature_lengths.ndim != 1 or feature_lengths.numel() != features.shape[0]:
        raise ValueError("SpecAugment feature-length batch does not match features")

    value = features.clone()
    frequency_masks_applied = 0
    time_masks_applied = 0
    frequency_bins_masked = 0
    time_frames_masked = 0
    mel_bins = int(value.shape[1])
    fixed_frames = int(value.shape[2])
    mask_value = float(policy["mask_value"])

    for sample_index in range(value.shape[0]):
        valid_frames = min(
            fixed_frames,
            max(1, int(feature_lengths[sample_index].item())),
        )
        for _ in range(int(policy["frequency_masks"])):
            max_width = min(
                mel_bins,
                int(policy["frequency_max_width"]),
            )
            width = int(
                torch.randint(
                    0,
                    max_width + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            if width == 0:
                continue
            start = int(
                torch.randint(
                    0,
                    mel_bins - width + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            value[
                sample_index,
                start : start + width,
                :valid_frames,
            ] = mask_value
            frequency_masks_applied += 1
            frequency_bins_masked += width

        time_cap = min(
            int(policy["time_max_width"]),
            max(
                1,
                int(valid_frames * float(policy["time_max_fraction"])),
            ),
        )
        for _ in range(int(policy["time_masks"])):
            width = int(
                torch.randint(
                    0,
                    time_cap + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            if width == 0:
                continue
            start = int(
                torch.randint(
                    0,
                    valid_frames - width + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            value[
                sample_index,
                :,
                start : start + width,
            ] = mask_value
            time_masks_applied += 1
            time_frames_masked += width

    return value, {
        "samples": int(value.shape[0]),
        "frequency_masks_applied": frequency_masks_applied,
        "time_masks_applied": time_masks_applied,
        "frequency_bins_masked": frequency_bins_masked,
        "time_frames_masked": time_frames_masked,
    }


def evaluate_validation(
    model,
    loader,
    loss_fn,
    device,
    vocab,
    *,
    progress_interval: int = 0,
    epoch: int | None = None,
    epochs: int | None = None,
) -> dict:
    model.eval()
    losses = []
    word_counts = []
    char_counts = []
    samples = []
    validation_started = time.monotonic()
    validation_batches = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            features = batch["features"].to(device)
            logits = model(features)
            loss = deterministic_ctc_loss(loss_fn, logits, batch)
            losses.append(float(loss.detach().cpu()))

            logits_cpu = logits.detach().cpu()
            input_lengths = batch["input_lengths"].tolist()
            for index, reference in enumerate(batch["texts"]):
                output_frames = int(input_lengths[index])
                decoder = greedy_decode_logits_diagnostics(
                    logits_cpu[index, :output_frames, :].tolist(),
                    vocab,
                )
                hypothesis = decoder["hypothesis"]
                words = word_error_counts(reference, hypothesis)
                chars = character_error_counts(reference, hypothesis)
                word_counts.append(words)
                char_counts.append(chars)
                samples.append(
                    {
                        "hypothesis": hypothesis,
                        "reference_words": len(reference.split()),
                        "reference_characters_no_spaces": len(
                            reference.replace(" ", "")
                        ),
                        "decoder": {
                            key: value
                            for key, value in decoder.items()
                            if key != "hypothesis"
                        },
                    }
                )

            if (
                progress_interval > 0
                and (
                    batch_index == 1
                    or batch_index % progress_interval == 0
                    or batch_index == validation_batches
                )
            ):
                elapsed = time.monotonic() - validation_started
                rate = batch_index / max(elapsed, 1e-9)
                remaining = validation_batches - batch_index
                print(
                    json.dumps(
                        {
                            "event": "validation_progress",
                            "epoch": epoch,
                            "epochs": epochs,
                            "batch": batch_index,
                            "batches": validation_batches,
                            "percent": (
                                100.0 * batch_index / max(1, validation_batches)
                            ),
                            "elapsed_seconds": elapsed,
                            "batches_per_second": rate,
                            "eta_seconds": remaining / max(rate, 1e-9),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    decoder_summary = aggregate_decoder_diagnostics(samples)
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "wer": _sum_rate(word_counts),
        "cer": _sum_rate(char_counts),
        "decoder": decoder_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--validation-manifest", type=pathlib.Path)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--vocab", type=pathlib.Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "cnn_ctc_v14" / "training",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation_loss", "validation_cer"),
    )
    parser.add_argument(
        "--ctc-objective-kind",
        choices=("blank-logit-penalty-v1",),
    )
    parser.add_argument("--blank-logit-penalty", type=float)
    parser.add_argument(
        "--augmentation-kind",
        choices=("specaugment-v1",),
    )
    parser.add_argument("--frequency-masks", type=int)
    parser.add_argument("--frequency-max-width", type=int)
    parser.add_argument("--time-masks", type=int)
    parser.add_argument("--time-max-width", type=int)
    parser.add_argument("--time-max-fraction", type=float)
    parser.add_argument("--augmentation-mask-value", type=float)
    parser.add_argument("--augmentation-seed-offset", type=int)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=250,
        help="emit training/validation progress every N batches; 0 disables",
    )
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            cuda_name = torch.cuda.get_device_name(device)
        else:
            cuda_name = None
        ctc_loss_device = "cpu" if device.type == "cuda" else str(device)
        print(
            json.dumps(
                {
                    "event": "training_device",
                    "requested": args.device,
                    "model_device": str(device),
                    "ctc_loss_device": ctc_loss_device,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_name": cuda_name,
                    "note": (
                        "CNN forward/backward runs on CUDA; deterministic "
                        "PyTorch 2.2 CTC loss runs on CPU."
                        if device.type == "cuda"
                        else "Training runs on CPU."
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

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
        if args.progress_interval < 0:
            raise ValueError("progress_interval must be >= 0")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0")
        if not 0 <= min_learning_rate <= learning_rate:
            raise ValueError(
                "min_learning_rate must be between 0 and learning_rate"
            )
        if spec["training"].get("lr_schedule") != "cosine":
            raise ValueError("cnn_ctc_v14 lr_schedule must be cosine")
        if spec["training"].get("checkpoint_selection") != "best_validation_loss":
            raise ValueError(
                "cnn_ctc_v14 package checkpoint_selection must remain best_validation_loss"
            )
        checkpoint_selection = args.checkpoint_selection or "validation_loss"

        ctc_objective = None
        blank_logit_penalty = 0.0
        if args.ctc_objective_kind is None:
            if args.blank_logit_penalty is not None:
                raise ValueError(
                    "--blank-logit-penalty requires --ctc-objective-kind"
                )
        else:
            if args.blank_logit_penalty is None:
                raise ValueError(
                    "blank-logit-penalty-v1 requires --blank-logit-penalty"
                )
            blank_logit_penalty = float(args.blank_logit_penalty)
            if not 0 < blank_logit_penalty <= 1.0:
                raise ValueError("blank_logit_penalty must be > 0 and <= 1")
            ctc_objective = {
                "kind": args.ctc_objective_kind,
                "blank_logit_penalty": blank_logit_penalty,
            }

        augmentation_policy = None
        augmentation_generator = None
        augmentation_stats = {
            "samples": 0,
            "frequency_masks_applied": 0,
            "time_masks_applied": 0,
            "frequency_bins_masked": 0,
            "time_frames_masked": 0,
        }
        augmentation_values = (
            args.frequency_masks,
            args.frequency_max_width,
            args.time_masks,
            args.time_max_width,
            args.time_max_fraction,
            args.augmentation_mask_value,
            args.augmentation_seed_offset,
        )
        if args.augmentation_kind is None:
            if any(value is not None for value in augmentation_values):
                raise ValueError(
                    "SpecAugment parameters require --augmentation-kind"
                )
        else:
            if args.augmentation_kind != "specaugment-v1":
                raise ValueError("unsupported augmentation policy")
            if any(value is None for value in augmentation_values):
                raise ValueError(
                    "specaugment-v1 requires all augmentation parameters"
                )
            augmentation_policy = {
                "kind": "specaugment-v1",
                "frequency_masks": int(args.frequency_masks),
                "frequency_max_width": int(args.frequency_max_width),
                "time_masks": int(args.time_masks),
                "time_max_width": int(args.time_max_width),
                "time_max_fraction": float(args.time_max_fraction),
                "mask_value": float(args.augmentation_mask_value),
                "seed_offset": int(args.augmentation_seed_offset),
            }
            if not 1 <= augmentation_policy["frequency_masks"] <= 4:
                raise ValueError("frequency_masks must be in [1,4]")
            if not 1 <= augmentation_policy["frequency_max_width"] <= 32:
                raise ValueError("frequency_max_width must be in [1,32]")
            if not 1 <= augmentation_policy["time_masks"] <= 4:
                raise ValueError("time_masks must be in [1,4]")
            if not 1 <= augmentation_policy["time_max_width"] <= 128:
                raise ValueError("time_max_width must be in [1,128]")
            if not 0 < augmentation_policy["time_max_fraction"] <= 0.5:
                raise ValueError("time_max_fraction must be in (0,0.5]")
            if augmentation_policy["mask_value"] != 0.0:
                raise ValueError("specaugment-v1 mask_value must be 0")
            if augmentation_policy["seed_offset"] < 1:
                raise ValueError("augmentation seed_offset must be >= 1")
            augmentation_generator = torch.Generator()
            augmentation_generator.manual_seed(
                seed + augmentation_policy["seed_offset"]
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

        train_audio_samples_per_epoch = dataset_audio_samples(train_dataset)
        validation_audio_samples = dataset_audio_samples(validation_dataset)

        model = CnnCtcV14(spec, len(vocab["tokens"])).to(device)
        model_device = str(next(model.parameters()).device)
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
        observed_logits_device = None
        best_epoch = None
        best_validation_loss = None
        best_validation_loss_epoch = None
        best_validation_cer = None
        best_validation_cer_epoch = None
        selected_metric_value = None
        best_state_dict = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        start = time.monotonic()

        if not args.init_only:
            total_train_steps = epochs * len(train_loader)
            print(
                json.dumps(
                    {
                        "event": "training_start",
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "train_samples": len(train_dataset),
                        "validation_samples": len(validation_dataset),
                        "train_batches_per_epoch": len(train_loader),
                        "validation_batches_per_epoch": len(validation_loader),
                        "total_train_steps": total_train_steps,
                        "progress_interval": args.progress_interval,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            assert train_loader is not None
            assert validation_loader is not None
            for epoch in range(epochs):
                epoch_started = time.monotonic()
                model.train()
                batch_losses = []
                gradient_norms = []
                current_lr = float(optimizer.param_groups[0]["lr"])
                print(
                    json.dumps(
                        {
                            "event": "epoch_start",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "learning_rate": current_lr,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                for batch_index, batch in enumerate(train_loader, start=1):
                    features = batch["features"]
                    if augmentation_policy is not None:
                        assert augmentation_generator is not None
                        features, batch_augmentation = apply_specaugment_v1(
                            features,
                            batch["feature_lengths"],
                            generator=augmentation_generator,
                            policy=augmentation_policy,
                        )
                        for key in augmentation_stats:
                            augmentation_stats[key] += int(
                                batch_augmentation[key]
                            )
                    features = features.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    if observed_logits_device is None:
                        observed_logits_device = str(logits.device)
                    loss = deterministic_ctc_loss(
                        loss_fn,
                        logits,
                        batch,
                        blank_index=int(vocab["blank_index"]),
                        blank_logit_penalty=blank_logit_penalty,
                    )
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                    optimizer.step()
                    optimizer_steps += 1
                    batch_losses.append(float(loss.detach().cpu()))
                    gradient_norms.append(float(gradient_norm.detach().cpu()))

                    if (
                        args.progress_interval > 0
                        and (
                            batch_index == 1
                            or batch_index % args.progress_interval == 0
                            or batch_index == len(train_loader)
                        )
                    ):
                        elapsed = time.monotonic() - start
                        completed_steps = epoch * len(train_loader) + batch_index
                        rate = completed_steps / max(elapsed, 1e-9)
                        remaining_steps = total_train_steps - completed_steps
                        print(
                            json.dumps(
                                {
                                    "event": "training_progress",
                                    "epoch": epoch + 1,
                                    "epochs": epochs,
                                    "batch": batch_index,
                                    "batches_per_epoch": len(train_loader),
                                    "completed_steps": completed_steps,
                                    "total_steps": total_train_steps,
                                    "percent": (
                                        100.0
                                        * completed_steps
                                        / max(1, total_train_steps)
                                    ),
                                    "elapsed_seconds": elapsed,
                                    "steps_per_second": rate,
                                    "eta_seconds": (
                                        remaining_steps / max(rate, 1e-9)
                                    ),
                                    "running_train_loss": float(
                                        sum(batch_losses) / len(batch_losses)
                                    ),
                                    "learning_rate": current_lr,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                print(
                    json.dumps(
                        {
                            "event": "validation_start",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "validation_batches": len(validation_loader),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                validation = evaluate_validation(
                    model,
                    validation_loader,
                    loss_fn,
                    device,
                    vocab,
                    progress_interval=args.progress_interval,
                    epoch=epoch + 1,
                    epochs=epochs,
                )
                validation_loss = float(validation["loss"])
                validation_cer = float(validation["cer"])
                validation_wer = float(validation["wer"])
                decoder = validation["decoder"]
                mean_train_loss = float(
                    sum(batch_losses) / max(1, len(batch_losses))
                )
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": mean_train_loss,
                        "validation_loss": validation_loss,
                        "validation_cer": validation_cer,
                        "validation_wer": validation_wer,
                        "validation_blank_frame_fraction": decoder[
                            "blank_frame_fraction"
                        ],
                        "validation_empty_hypothesis_fraction": decoder[
                            "empty_hypothesis_fraction"
                        ],
                        "validation_emitted_to_reference_character_ratio": decoder[
                            "emitted_to_reference_character_ratio"
                        ],
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
                    best_validation_loss_epoch = epoch + 1
                if (
                    best_validation_cer is None
                    or validation_cer < best_validation_cer
                ):
                    best_validation_cer = validation_cer
                    best_validation_cer_epoch = epoch + 1

                candidate_value = (
                    validation_loss
                    if checkpoint_selection == "validation_loss"
                    else validation_cer
                )
                if (
                    selected_metric_value is None
                    or candidate_value < selected_metric_value
                ):
                    selected_metric_value = candidate_value
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                scheduler.step()
                epoch_elapsed = time.monotonic() - epoch_started
                total_elapsed = time.monotonic() - start
                completed_epochs = epoch + 1
                mean_epoch_seconds = total_elapsed / completed_epochs
                print(
                    json.dumps(
                        {
                            "event": "epoch_complete",
                            "epoch": completed_epochs,
                            "epochs": epochs,
                            "epoch_seconds": epoch_elapsed,
                            "elapsed_seconds": total_elapsed,
                            "eta_seconds": (
                                mean_epoch_seconds * (epochs - completed_epochs)
                            ),
                            "train_loss": mean_train_loss,
                            "validation_loss": validation_loss,
                            "validation_cer": validation_cer,
                            "validation_wer": validation_wer,
                            "validation_blank_frame_fraction": decoder[
                                "blank_frame_fraction"
                            ],
                            "validation_empty_hypothesis_fraction": decoder[
                                "empty_hypothesis_fraction"
                            ],
                            "validation_emitted_to_reference_character_ratio": decoder[
                                "emitted_to_reference_character_ratio"
                            ],
                            "best_epoch": best_epoch,
                            "selected_metric_value": selected_metric_value,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

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
                "checkpoint_selection": checkpoint_selection,
                "selected_metric_value": selected_metric_value,
                "ctc_objective": ctc_objective,
                "augmentation": augmentation_policy,
                "augmentation_stats": augmentation_stats,
                "best_validation_loss": best_validation_loss,
                "best_validation_loss_epoch": best_validation_loss_epoch,
                "best_validation_cer": best_validation_cer,
                "best_validation_cer_epoch": best_validation_cer_epoch,
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
            "model_device": model_device,
            "observed_logits_device": observed_logits_device,
            "ctc_loss_device": ctc_loss_device,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": cuda_name,
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
            "cuda_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
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
            "checkpoint_selection": checkpoint_selection,
            "selected_metric_value": selected_metric_value,
            "ctc_objective": ctc_objective,
            "augmentation": augmentation_policy,
            "augmentation_stats": augmentation_stats,
            "parameter_count": parameter_count(model),
            "optimizer_steps": optimizer_steps,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_loss_epoch": best_validation_loss_epoch,
            "best_validation_cer": best_validation_cer,
            "best_validation_cer_epoch": best_validation_cer_epoch,
            "train_samples": 0 if train_dataset is None else len(train_dataset),
            "validation_samples": (
                0 if validation_dataset is None else len(validation_dataset)
            ),
            "train_audio_seconds_per_epoch": (
                train_audio_samples_per_epoch / 16000.0
            ),
            "processed_train_audio_seconds": (
                (train_audio_samples_per_epoch * epochs) / 16000.0
                if not args.init_only
                else 0.0
            ),
            "validation_audio_seconds": (
                validation_audio_samples / 16000.0
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
