"""Compact local-context cnn_ctc_v5 with training-only intermediate CTC head."""

from __future__ import annotations

import torch
from torch import nn

from cnn_ctc_v1 import ManifestCtcDataset, collate_ctc


class ResidualTemporalBlockV5(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: int,
        padding: int,
        projection_kernel: int,
        dropout: float,
    ):
        super().__init__()
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Conv1d(
            channels,
            channels,
            kernel_size=projection_kernel,
            stride=1,
            padding=0,
            bias=True,
        )
        nn.init.kaiming_normal_(
            self.projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        with torch.no_grad():
            self.projection.weight.mul_(0.01)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features):
        residual = features
        value = self.temporal(features)
        value = self.activation(value)
        value = self.dropout(value)
        value = self.projection(value)
        return self.activation(value + residual)


class CnnCtcV5(nn.Module):
    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        if spec.get("id") != "cnn_ctc_v5":
            raise ValueError("CnnCtcV5 requires cnn_ctc_v5 model spec")
        network = spec["network"]
        if network.get("kind") != "residual-temporal-v3":
            raise ValueError("cnn_ctc_v5 requires residual-temporal-v3 network")
        if network.get("normalization") != "none":
            raise ValueError("cnn_ctc_v5 must remain normalization-free")

        in_channels = int(spec["frontend"]["mel_bins"])
        stem_layers = []
        for layer in network["stem"]:
            out_channels = int(layer["channels"])
            stem_layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=int(layer["kernel"]),
                        stride=int(layer["stride"]),
                        padding=int(layer["padding"]),
                        bias=True,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels
        self.stem = nn.Sequential(*stem_layers)

        dropout = float(network["dropout"])
        blocks = []
        for block in network["residual_blocks"]:
            channels = int(block["channels"])
            if channels != in_channels:
                raise ValueError("cnn_ctc_v5 residual blocks must preserve channels")
            blocks.append(
                ResidualTemporalBlockV5(
                    channels=channels,
                    kernel=int(block["kernel"]),
                    padding=int(block["padding"]),
                    projection_kernel=int(block["projection_kernel"]),
                    dropout=dropout,
                )
            )
        self.residual = nn.ModuleList(blocks)
        self.intermediate_after_block = int(network["intermediate_ctc_after_block"])
        if not 1 <= self.intermediate_after_block < len(self.residual):
            raise ValueError("intermediate CTC head must precede the final block")
        projection_kernel = int(network["projection_kernel"])
        self.intermediate_projection = nn.Conv1d(
            in_channels,
            vocab_size,
            kernel_size=projection_kernel,
            bias=True,
        )
        self.projection = nn.Conv1d(
            in_channels,
            vocab_size,
            kernel_size=projection_kernel,
            bias=True,
        )

    def _encode(self, features, *, return_intermediate: bool):
        encoded = self.stem(features)
        intermediate = None
        for index, block in enumerate(self.residual, start=1):
            encoded = block(encoded)
            if return_intermediate and index == self.intermediate_after_block:
                intermediate = encoded
        return encoded, intermediate

    def forward(self, features):
        encoded, _ = self._encode(features, return_intermediate=False)
        return self.projection(encoded).transpose(1, 2)

    def forward_with_intermediate(self, features):
        encoded, intermediate = self._encode(features, return_intermediate=True)
        if intermediate is None:
            raise RuntimeError("intermediate CTC representation was not captured")
        logits = self.projection(encoded).transpose(1, 2)
        intermediate_logits = self.intermediate_projection(intermediate).transpose(1, 2)
        return logits, intermediate_logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def deployed_parameter_count(model: CnnCtcV5) -> int:
    auxiliary = sum(
        parameter.numel() for parameter in model.intermediate_projection.parameters()
    )
    return parameter_count(model) - auxiliary
