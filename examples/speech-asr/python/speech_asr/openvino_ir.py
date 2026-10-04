"""Small OpenVINO IR inspector used to freeze generated tensor contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def _port_shape(port: ET.Element) -> list[int]:
    return [int(dim.text) for dim in port.findall("dim")]


def _semantic_element(element: ET.Element) -> dict[str, Any]:
    """Return deterministic executable-graph XML content.

    Model Optimizer appends a top-level <meta_data> section containing
    conversion provenance/CLI details. Those values do not participate in
    inference and may differ across preparation hosts, so only that section is
    excluded. Layer/edge/blob structure and all of their attributes remain part
    of the fingerprint.
    """
    children = [
        _semantic_element(child)
        for child in list(element)
        if child.tag != "meta_data"
    ]
    text = (element.text or "").strip()
    return {
        "tag": element.tag,
        "attributes": dict(sorted(element.attrib.items())),
        "text": text or None,
        "children": children,
    }


def _json_sha256(document: Any) -> str:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def graph_sha256(root: ET.Element) -> str:
    """Hash executable XML while preserving serializer IDs and ordering."""
    return _json_sha256(_semantic_element(root))


def canonical_graph_document(root: ET.Element) -> dict[str, Any]:
    """Return an ID/order-insensitive but structure-sensitive IR document.

    Layer numeric IDs and XML serialization order are Model Optimizer
    implementation details. Canonical identity maps edges through unique layer
    names and sorts layers/edges. Everything else in executable layer XML is
    preserved, including names, types, precisions, data attributes, blobs,
    ports, dimensions and connectivity.
    """
    layers_node = root.find("layers")
    edges_node = root.find("edges")
    if layers_node is None or edges_node is None:
        raise ValueError("IR must contain layers and edges")

    id_to_name: dict[str, str] = {}
    names: set[str] = set()
    canonical_layers = []
    for layer in layers_node.findall("layer"):
        layer_id = layer.attrib["id"]
        name = layer.attrib.get("name", "")
        if not name:
            raise ValueError(f"IR layer {layer_id} has no stable name")
        if name in names:
            raise ValueError(f"IR layer name is not unique: {name}")
        names.add(name)
        id_to_name[layer_id] = name

        item = _semantic_element(layer)
        item["attributes"] = dict(item["attributes"])
        item["attributes"].pop("id", None)
        canonical_layers.append(item)

    canonical_edges = []
    for edge in edges_node.findall("edge"):
        source_id = edge.attrib["from-layer"]
        target_id = edge.attrib["to-layer"]
        if source_id not in id_to_name or target_id not in id_to_name:
            raise ValueError("IR edge references unknown layer id")
        canonical_edges.append({
            "from_layer": id_to_name[source_id],
            "from_port": edge.attrib.get("from-port"),
            "to_layer": id_to_name[target_id],
            "to_port": edge.attrib.get("to-port"),
        })

    extras = [
        _semantic_element(child)
        for child in list(root)
        if child.tag not in {"layers", "edges", "meta_data"}
    ]
    extras.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))

    return {
        "network": {
            "attributes": dict(sorted(root.attrib.items())),
        },
        "layers": sorted(
            canonical_layers,
            key=lambda item: (
                item["attributes"].get("name", ""),
                item["attributes"].get("type", ""),
            ),
        ),
        "edges": sorted(
            canonical_edges,
            key=lambda item: (
                item["from_layer"],
                item["from_port"] or "",
                item["to_layer"],
                item["to_port"] or "",
            ),
        ),
        "extras": extras,
    }


def canonical_graph_sha256(root: ET.Element) -> str:
    return _json_sha256(canonical_graph_document(root))


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

    # Only explicit graph data-input layers are model inputs. Const layers
    # also have zero incoming edges in IR v7, but they are embedded weights /
    # reshape metadata and must never appear in the external tensor contract.
    inputs = [
        layer
        for layer in layers.values()
        if layer["type"] in {"Input", "Parameter"} and layer["output_ports"]
    ]

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
        "graph_sha256": graph_sha256(root),
        "canonical_graph_sha256": canonical_graph_sha256(root),
        "inputs": [compact(layer) for layer in sorted(inputs, key=lambda item: item["id"])],
        "outputs": [compact(layer) for layer in sorted(outputs, key=lambda item: item["id"])],
    }
    if bin_path is not None:
        result["bin_sha256"] = hashlib.sha256(bin_path.read_bytes()).hexdigest()
    if not result["inputs"] or not result["outputs"]:
        raise ValueError("could not identify IR input/output tensors")
    return result
