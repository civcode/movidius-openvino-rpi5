"""PyTorch implementation for cnn_ctc_v15 BatchNorm-conditioned large-capacity CTC."""

from __future__ import annotations

import torch
from torch import nn

from cnn_ctc_v1 import ManifestCtcDataset, collate_ctc


class ResidualTemporalBlockV15(nn.Module):
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
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Conv1d(
            channels,
            channels,
            kernel_size=projection_kernel,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)
        nn.init.kaiming_normal_(
            self.projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        with torch.no_grad():
            self.projection.weight.mul_(0.01)

    def forward(self, features):
        residual = features
        value = self.temporal(features)
        value = self.bn1(value)
        value = self.activation(value)
        value = self.dropout(value)
        value = self.projection(value)
        value = self.bn2(value)
        value = value + residual
        return self.activation(value)


class CnnCtcV15(nn.Module):
    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        if spec.get("id") != "cnn_ctc_v15":
            raise ValueError("CnnCtcV15 requires cnn_ctc_v15 model spec")
        network = spec["network"]
        if network.get("kind") != "residual-temporal-v10":
            raise ValueError("cnn_ctc_v15 requires residual-temporal-v10 network")
        if network.get("normalization") != "batchnorm":
            raise ValueError("cnn_ctc_v15 requires BatchNorm conditioning")

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
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
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
                raise ValueError("cnn_ctc_v15 residual blocks must preserve channels")
            blocks.append(
                ResidualTemporalBlockV15(
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
