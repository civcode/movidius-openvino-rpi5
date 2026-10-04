"""Small OpenVINO IR inspector used to freeze generated tensor contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def _port_shape(port: ET.Element) -> list[int]:
    return [int(dim.text) for dim in port.findall("dim")]


def inspect_ir(xml_path: Path, bin_path: Path | None = None) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    layers_node = root.find("layers")
    edges_node = root.find("edges")
    if layers_node is None or edges_node is None:
        raise ValueError("IR must contain layers and edges")

    layers = {}
    for layer in layers_node.findall("layer"):
        layer_id = layer.attrib["id"]
        output_ports = []
        output_node = layer.find("output")
        if output_node is not None:
            for port in output_node.findall("port"):
                output_ports.append({
                    "id": int(port.attrib["id"]),
                    "precision": port.attrib.get("precision", layer.attrib.get("precision")),
                    "shape": _port_shape(port),
                })
        layers[layer_id] = {
            "id": int(layer_id),
            "name": layer.attrib.get("name", ""),
            "type": layer.attrib.get("type", ""),
            "precision": layer.attrib.get("precision"),
            "output_ports": output_ports,
        }

    outgoing = {layer_id: 0 for layer_id in layers}
    incoming = {layer_id: 0 for layer_id in layers}
    for edge in edges_node.findall("edge"):
        source = edge.attrib["from-layer"]
        target = edge.attrib["to-layer"]
        if source in outgoing:
            outgoing[source] += 1
        if target in incoming:
            incoming[target] += 1

    inputs = []
    for layer_id, layer in layers.items():
        if layer["type"] in {"Input", "Parameter"} or incoming[layer_id] == 0:
            if layer["output_ports"]:
                inputs.append(layer)

    outputs = [
        layer for layer_id, layer in layers.items()
        if outgoing[layer_id] == 0 and layer["output_ports"]
    ]

    def compact(layer: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": layer["name"],
            "type": layer["type"],
            "precision": layer["precision"],
            "ports": layer["output_ports"],
        }

    result = {
        "schema": "speech-asr/openvino-ir-contract",
        "version": 1,
        "network_name": root.attrib.get("name", ""),
        "ir_version": root.attrib.get("version"),
        "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "inputs": [compact(layer) for layer in sorted(inputs, key=lambda item: item["id"])],
        "outputs": [compact(layer) for layer in sorted(outputs, key=lambda item: item["id"])],
    }
    if bin_path is not None:
        result["bin_sha256"] = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    if not result["inputs"] or not result["outputs"]:
        raise ValueError("could not identify IR input/output tensors")
    return result
