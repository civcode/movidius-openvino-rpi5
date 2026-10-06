"""PyTorch implementation for cnn_ctc_v12 448-channel 16-block large-capacity CTC."""

from __future__ import annotations

import torch
from torch import nn

from cnn_ctc_v1 import ManifestCtcDataset, collate_ctc


class ResidualTemporalBlockV12(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: int,
        dilation: int,
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
            dilation=dilation,
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
        value = value + residual
        return self.activation(value)


class CnnCtcV12(nn.Module):
    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        if spec.get("id") != "cnn_ctc_v12":
            raise ValueError("CnnCtcV12 requires cnn_ctc_v12 model spec")
        network = spec["network"]
        if network.get("kind") != "residual-temporal-v9":
            raise ValueError("cnn_ctc_v12 requires residual-temporal-v9 network")
        if network.get("normalization") != "none":
            raise ValueError("cnn_ctc_v12 must remain normalization-free")

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
                raise ValueError("cnn_ctc_v12 residual blocks must preserve channels")
            blocks.append(
                ResidualTemporalBlockV12(
                    channels=channels,
                    kernel=int(block["kernel"]),
                    dilation=int(block["dilation"]),
                    padding=int(block["padding"]),
                    projection_kernel=int(block["projection_kernel"]),
                    dropout=dropout,
                )
            )
        self.residual = nn.Sequential(*blocks)
        self.projection = nn.Conv1d(
            in_channels,
            vocab_size,
            kernel_size=int(network["projection_kernel"]),
            bias=True,
        )

    def forward(self, features):
        encoded = self.stem(features)
        encoded = self.residual(encoded)
        logits = self.projection(encoded)
        return logits.transpose(1, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
