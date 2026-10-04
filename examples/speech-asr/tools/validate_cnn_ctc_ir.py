#!/usr/bin/env python3
"""Validate converted cnn_ctc_v1 IR against its declared external tensor contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def validate(spec: dict, contract: dict) -> dict:
    if spec.get("id") != "cnn_ctc_v1":
        raise ValueError("model spec id must be cnn_ctc_v1")
    if contract.get("schema") != "speech-asr/openvino-ir-contract":
        raise ValueError("unexpected IR contract schema")
    if contract.get("ir_version") not in {"7", "10"}:
        raise ValueError(
            f"unsupported IR version for cnn_ctc_v1: {contract.get('ir_version')!r}"
        )

    inputs = contract.get("inputs")
    outputs = contract.get("outputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("cnn_ctc_v1 IR must expose exactly one input")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("cnn_ctc_v1 IR must expose exactly one output")

    declared_input = spec["input_contract"]
    declared_output = spec["output_contract"]
    actual_input = inputs[0]
    actual_output = outputs[0]

    if actual_input.get("name") != declared_input["name"]:
        raise ValueError(
            "IR input name mismatch: "
            f"{actual_input.get('name')!r} != {declared_input['name']!r}"
        )
    if actual_output.get("name") != declared_output["name"]:
        raise ValueError(
            "IR output name mismatch: "
            f"{actual_output.get('name')!r} != {declared_output['name']!r}"
        )

    input_ports = actual_input.get("ports")
    output_ports = actual_output.get("ports")
    if not isinstance(input_ports, list) or len(input_ports) != 1:
        raise ValueError("cnn_ctc_v1 IR input must expose exactly one port")
    if not isinstance(output_ports, list) or len(output_ports) != 1:
        raise ValueError("cnn_ctc_v1 IR output must expose exactly one port")

    input_shape = input_ports[0].get("shape")
    output_shape = output_ports[0].get("shape")
    if input_shape != declared_input["shape"]:
        raise ValueError(
            f"IR input shape mismatch: {input_shape!r} != {declared_input['shape']!r}"
        )
    if output_shape != declared_output["shape"]:
        raise ValueError(
            f"IR output shape mismatch: {output_shape!r} != {declared_output['shape']!r}"
        )

    return {
        "schema": "speech-asr/cnn-ctc-ir-validation",
        "version": 1,
        "model_id": spec["id"],
        "status": "valid",
        "ir_version": contract["ir_version"],
        "input": {
            "name": actual_input["name"],
            "shape": input_shape,
        },
        "output": {
            "name": actual_output["name"],
            "shape": output_shape,
            **(
                {"result_name": actual_output["result_name"]}
                if actual_output.get("result_name") is not None
                else {}
            ),
        },
        "canonical_graph_sha256": contract["canonical_graph_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=pathlib.Path)
    parser.add_argument("ir_contract", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        contract = json.loads(args.ir_contract.read_text(encoding="utf-8"))
        result = validate(spec, contract)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
