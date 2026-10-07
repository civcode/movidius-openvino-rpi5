#!/usr/bin/env python3
"""Stage and run the frozen cnn_ctc_v19 + decoder measurement on edge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SPEECH_ROOT = HERE.parent
ROOT = SPEECH_ROOT.parents[1]
sys.path.insert(0, str(SPEECH_ROOT / "python"))

from speech_asr.cnn_ctc import load_vocab  # noqa: E402
from speech_asr.contracts import (  # noqa: E402
    validate_experiment_result,
    validate_speech_sample,
)
from speech_asr.ctc_beam import load_decoder_artifact  # noqa: E402
from speech_asr.orchestration import (  # noqa: E402
    rsync_pull_command,
    rsync_push_command,
    ssh_command,
)

SOURCE_EXPERIMENT = "exp-f4adb44ab833e896"
SOURCE_ATTEMPT = "attempt-0002"
EXPECTED_MANIFEST_SHA256 = (
    "fbd72a648826b2e200f9244b4dc73be425cada2e304be92d513f7033a8de4088"
)
EXPECTED_WER = 0.5497732671129346
EXPECTED_CER = 0.4634567759027818
EXPECTED_SAMPLES = 1273
BENCHMARK_ID = "ami-model-quality-v4-architecture-screen-v1-es2011-validation"


def sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stdout}"
        )
    return proc


def run_stream(argv: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = proc.wait()
    return subprocess.CompletedProcess(argv, returncode, "".join(lines), None)


def remote_text(worker: str, argv: list[str]) -> str:
    proc = run_capture(ssh_command(worker, argv))
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"remote command returned no output: {argv!r}")
    return lines[-1]


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def stage_validation_dataset(
    manifest: pathlib.Path,
) -> tuple[pathlib.Path, list[pathlib.Path], pathlib.Path]:
    """Mirror the frozen manifest plus referenced audio into a temporary tree."""

    ami_root = (ROOT / "work" / "speech-asr" / "ami").resolve()
    manifest = manifest.resolve()
    try:
        manifest_relative = manifest.relative_to(ami_root)
    except ValueError as exc:
        raise ValueError(
            f"validation manifest must live below frozen AMI root: {manifest}"
        ) from exc

    staging_parent = ROOT / "work" / "speech-asr"
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage_root = pathlib.Path(
        tempfile.mkdtemp(prefix="v19-deployed-dataset-", dir=staging_parent)
    )
    top_levels = {manifest_relative.parts[0]}
    copied_audio: set[pathlib.Path] = set()
    records = 0
    try:
        staged_manifest = stage_root / manifest_relative
        staged_manifest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, staged_manifest)

        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = validate_speech_sample(json.loads(line))
                records += 1
                raw = pathlib.Path(record["audio"]["path"])
                audio = raw if raw.is_absolute() else manifest.parent / raw
                audio = audio.resolve()
                try:
                    relative = audio.relative_to(ami_root)
                except ValueError as exc:
                    raise ValueError(
                        f"{record['id']}: validation audio escapes frozen AMI root: "
                        f"{audio}"
                    ) from exc
                if not audio.is_file():
                    raise ValueError(
                        f"{record['id']}: validation audio missing: {audio}"
                    )
                if audio in copied_audio:
                    continue
                copied_audio.add(audio)
                top_levels.add(relative.parts[0])
                destination = stage_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(audio, destination)
                except OSError:
                    shutil.copy2(audio, destination)

        if records != EXPECTED_SAMPLES:
            raise ValueError(
                f"validation manifest record count changed: "
                f"{records} != {EXPECTED_SAMPLES}"
            )
        sources = [stage_root / name for name in sorted(top_levels)]
        return stage_root, sources, manifest_relative
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=pathlib.Path,
        default=ROOT / "work" / "speech-asr" / "experiments" / SOURCE_EXPERIMENT,
    )
    parser.add_argument("--attempt", default=SOURCE_ATTEMPT)
    parser.add_argument(
        "--decoder-artifact",
        type=pathlib.Path,
        help="override the frozen attempt results/decoder-artifact-v1.json",
    )
    parser.add_argument("--worker", default="edge")
    parser.add_argument("--edge-repo")
    parser.add_argument("--cpuset", default="0-3")
    parser.add_argument("--output-dir", type=pathlib.Path)
    args = parser.parse_args()

    try:
        experiment = args.experiment.resolve()
        request = load_json(experiment / "request" / "experiment.json")
        if request.get("experiment_id") != SOURCE_EXPERIMENT:
            raise ValueError(
                f"expected source experiment {SOURCE_EXPERIMENT}, "
                f"got {request.get('experiment_id')!r}"
            )
        attempt_dir = experiment / "attempts" / args.attempt
        if args.attempt != SOURCE_ATTEMPT or not attempt_dir.is_dir():
            raise ValueError(
                f"deployed decoder measurement is frozen to {SOURCE_ATTEMPT}"
            )

        ir_dir = attempt_dir / "build" / "cnn_ctc_v19" / "openvino" / "fp16"
        xml = ir_dir / "cnn_ctc_v19.xml"
        binary = ir_dir / "cnn_ctc_v19.bin"
        decoder_artifact = (
            args.decoder_artifact.resolve()
            if args.decoder_artifact is not None
            else attempt_dir / "results" / "decoder-artifact-v1.json"
        )
        manifest = (
            ROOT
            / "work"
            / "speech-asr"
            / "ami"
            / "model-quality-v4"
            / "validation.manifest.jsonl"
        )
        for path in (xml, binary, decoder_artifact, manifest):
            if not path.is_file():
                raise ValueError(f"required local file missing: {path}")
        if sha256_path(manifest) != EXPECTED_MANIFEST_SHA256:
            raise ValueError("local validation manifest hash changed")

        vocab = load_vocab(
            SPEECH_ROOT / "models" / "cnn_ctc_v19" / "vocab.json"
        )
        artifact = load_decoder_artifact(decoder_artifact, vocab=vocab)
        provenance = artifact.get("provenance", {})
        if provenance.get("validation_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            raise ValueError("decoder artifact validation provenance changed")
        if abs(float(provenance.get("selected_wer", -1.0)) - EXPECTED_WER) > 1e-12:
            raise ValueError("decoder artifact selected WER changed")
        if abs(float(provenance.get("selected_cer", -1.0)) - EXPECTED_CER) > 1e-12:
            raise ValueError("decoder artifact selected CER changed")

        branch = run_capture(["git", "branch", "--show-current"]).stdout.strip()
        if branch != "main":
            raise ValueError(f"controller checkout must remain on main, got {branch!r}")
        local_head = run_capture(["git", "rev-parse", "HEAD"]).stdout.strip()
        decoder_sha = sha256_path(decoder_artifact)
        xml_sha = sha256_path(xml)
        bin_sha = sha256_path(binary)

        home = remote_text(
            args.worker,
            ["sh", "-c", 'printf "%s\\n" "$HOME"'],
        )
        remote_repo = args.edge_repo or f"{home}/workspace/movidius-openvino-rpi5"
        remote_branch = remote_text(
            args.worker,
            ["git", "-C", remote_repo, "branch", "--show-current"],
        )
        if remote_branch != "main":
            raise ValueError(
                f"edge human checkout must remain on main, got {remote_branch!r}"
            )
        pull = run_capture(
            ssh_command(
                args.worker,
                ["git", "-C", remote_repo, "pull", "--ff-only"],
            )
        )
        print(pull.stdout, end="" if pull.stdout.endswith("\n") else "\n")
        remote_head = remote_text(
            args.worker,
            ["git", "-C", remote_repo, "rev-parse", "HEAD"],
        )
        if remote_head != local_head:
            raise ValueError(
                f"controller/edge revision mismatch: {local_head} != {remote_head}"
            )

        host_runtime_preflight = run_capture(
            ssh_command(
                args.worker,
                [
                    f"{remote_repo}/scripts/run-myriad-tensor.sh",
                    "--platform",
                    "arm64",
                    "--backend",
                    "host",
                    "check",
                ],
            )
        )
        print(
            host_runtime_preflight.stdout,
            end="" if host_runtime_preflight.stdout.endswith("\n") else "\n",
        )

        run_id = (
            f"{SOURCE_EXPERIMENT}-{SOURCE_ATTEMPT}-decoder-v1-"
            f"{local_head[:8]}-{decoder_sha[:8]}"
        )
        remote_base = f"{remote_repo}/work/speech-asr/v19-deployed-eval/{run_id}"
        remote_input = f"{remote_base}/input"
        remote_dataset = f"{remote_base}/dataset/ami"
        remote_cache = f"{remote_base}/cache"
        remote_evidence = f"{remote_base}/evidence"
        run_capture(
            ssh_command(
                args.worker,
                [
                    "mkdir",
                    "-p",
                    remote_input,
                    remote_dataset,
                    remote_cache,
                    remote_evidence,
                ],
            )
        )

        stage_root, dataset_sources, manifest_relative = stage_validation_dataset(
            manifest
        )
        try:
            dataset_push = run_capture(
                rsync_push_command(
                    worker=args.worker,
                    sources=dataset_sources,
                    remote_dir=remote_dataset,
                )
            )
            if dataset_push.stdout:
                print(
                    dataset_push.stdout,
                    end="" if dataset_push.stdout.endswith("\n") else "\n",
                )
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)

        remote_manifest = (
            f"{remote_dataset}/{manifest_relative.as_posix()}"
        )
        remote_manifest_sha = remote_text(
            args.worker,
            ["sha256sum", remote_manifest],
        ).split()[0]
        if remote_manifest_sha != EXPECTED_MANIFEST_SHA256:
            raise ValueError(
                f"edge staged validation manifest hash changed: "
                f"{remote_manifest_sha}"
            )
        run_capture(
            rsync_push_command(
                worker=args.worker,
                sources=[xml, binary, decoder_artifact],
                remote_dir=remote_input,
            )
        )

        remote_decoder = f"{remote_input}/{decoder_artifact.name}"
        experiment_id = f"{SOURCE_EXPERIMENT}-deployed-decoder-v1"

        remote_repo_prefix = remote_repo.rstrip("/") + "/"
        if not remote_manifest.startswith(remote_repo_prefix):
            raise ValueError("staged manifest is not below edge repository")
        remote_manifest_relative = remote_manifest[len(remote_repo_prefix):]
        edge_preflight = run_capture(
            ssh_command(
                args.worker,
                [
                    "bash",
                    f"{remote_repo}/scripts/edge-speech-preflight.sh",
                    "--manifest",
                    remote_manifest_relative,
                    "--manifest-sha256",
                    EXPECTED_MANIFEST_SHA256,
                    "--runtime-backend",
                    "host",
                ],
            )
        )
        print(
            edge_preflight.stdout,
            end="" if edge_preflight.stdout.endswith("\n") else "\n",
        )

        deployed_preflight = run_capture(
            ssh_command(
                args.worker,
                [
                    "bash",
                    f"{remote_repo}/scripts/evaluate-cnn-ctc-v19-deployed.sh",
                    "--manifest",
                    remote_manifest,
                    "--ir-dir",
                    remote_input,
                    "--decoder-artifact",
                    remote_decoder,
                    "--preflight-only",
                ],
            )
        )
        print(
            deployed_preflight.stdout,
            end="" if deployed_preflight.stdout.endswith("\n") else "\n",
        )

        command = [
            "bash",
            f"{remote_repo}/scripts/evaluate-cnn-ctc-v19-deployed.sh",
            "--manifest",
            remote_manifest,
            "--ir-dir",
            remote_input,
            "--decoder-artifact",
            remote_decoder,
            "--work-dir",
            f"{remote_cache}/evaluation",
            "--output",
            f"{remote_evidence}/result.json",
            "--log",
            f"{remote_evidence}/evaluator.log",
            "--benchmark-id",
            BENCHMARK_ID,
            "--experiment-id",
            experiment_id,
            "--cpuset",
            args.cpuset,
            "--fresh",
        ]
        proc = run_stream(ssh_command(args.worker, command))

        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else (
                ROOT
                / "work"
                / "speech-asr"
                / "deployed-evaluations"
                / f"{SOURCE_EXPERIMENT}-{SOURCE_ATTEMPT}-decoder-v1"
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in (output_dir / "result.json", output_dir / "evaluator.log"):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass

        pull_evidence = run_capture(
            rsync_pull_command(
                worker=args.worker,
                remote_dir=remote_evidence,
                local_dir=output_dir,
            ),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"edge deployed evaluation failed ({proc.returncode}); "
                f"evidence pull rc={pull_evidence.returncode}\n"
                + "\n".join(proc.stdout.splitlines()[-40:])
            )
        if pull_evidence.returncode != 0:
            raise RuntimeError(
                "edge evaluation completed but evidence pull failed:\n"
                + pull_evidence.stdout
            )

        result_path = output_dir / "result.json"
        result = validate_experiment_result(load_json(result_path))
        if result["runtime"]["repo_commit"] != local_head:
            raise ValueError("result repository commit mismatch")
        if result["benchmark"]["manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
            raise ValueError("result validation manifest mismatch")
        artifacts = result["model"]["artifact_sha256"]
        if artifacts.get("cnn_ctc_v19.xml") != xml_sha:
            raise ValueError("result XML hash mismatch")
        if artifacts.get("cnn_ctc_v19.bin") != bin_sha:
            raise ValueError("result BIN hash mismatch")
        if result.get("provenance", {}).get("decoder_artifact_sha256") != decoder_sha:
            raise ValueError("result decoder artifact hash mismatch")

        deployed = result.get("metrics", {}).get("deployed_decoder")
        if not isinstance(deployed, dict):
            raise ValueError("result deployed decoder summary missing")
        if deployed.get("artifact_sha256") != decoder_sha:
            raise ValueError("deployed decoder summary artifact hash mismatch")
        expected_decoder = {
            "kind": "prefix-beam",
            "beam_width": 8,
            "token_top_k": 12,
            "lm_weight": 0.3,
            "word_bonus": -0.2,
        }
        for key, value in expected_decoder.items():
            if deployed.get(key) != value:
                raise ValueError(f"deployed decoder field changed: {key}")

        summary = {
            "status": "completed",
            "source_experiment": SOURCE_EXPERIMENT,
            "source_attempt": SOURCE_ATTEMPT,
            "repo_commit": local_head,
            "result": str(result_path),
            "evaluator_log": str(output_dir / "evaluator.log"),
            "wer": result["metrics"]["wer"],
            "cer": result["metrics"]["cer"],
            "frozen_quality_match": (
                abs(float(result["metrics"]["wer"]) - EXPECTED_WER) <= 1e-12
                and abs(float(result["metrics"]["cer"]) - EXPECTED_CER) <= 1e-12
            ),
            "inference_latency_p50_ms": result["metrics"]["inference_latency_p50_ms"],
            "inference_latency_p95_ms": result["metrics"]["inference_latency_p95_ms"],
            "decoder_latency_p50_ms": deployed["decode_latency_p50_ms"],
            "decoder_latency_p95_ms": deployed["decode_latency_p95_ms"],
            "decoder_latency_mean_ms": deployed["decode_latency_mean_ms"],
            "acoustic_plus_decoder_latency_p50_ms": (
                deployed["acoustic_plus_decoder_latency_p50_ms"]
            ),
            "acoustic_plus_decoder_latency_p95_ms": (
                deployed["acoustic_plus_decoder_latency_p95_ms"]
            ),
            "acoustic_plus_decoder_latency_mean_ms": (
                deployed["acoustic_plus_decoder_latency_mean_ms"]
            ),
            "acoustic_plus_decoder_realtime_factor": (
                deployed["acoustic_plus_decoder_realtime_factor"]
            ),
        }
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
