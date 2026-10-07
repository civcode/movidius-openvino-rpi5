"""PyTorch QuartzNet-15x5 acoustic model for cnn_ctc_v18 pretrained transfer."""

from __future__ import annotations

import torch
from torch import nn

from cnn_ctc_v1 import ManifestCtcDataset, collate_ctc


class TimeChannelSeparableConv1d(nn.Module):
    """Depthwise temporal convolution followed by pointwise channel mixing and BN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        batchnorm_eps: float = 1e-3,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(out_channels, eps=batchnorm_eps)

    def forward(self, features):
        return self.batchnorm(self.pointwise(self.depthwise(features)))


class QuartzNetBlock(nn.Module):
    """One QuartzNet Bi block: five TCS modules plus projected residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel: int,
        module_repeats: int,
        dilation: int,
        padding: int,
        dropout: float,
        batchnorm_eps: float,
    ):
        super().__init__()
        if module_repeats < 1:
            raise ValueError("QuartzNet module_repeats must be >= 1")
        modules = []
        current = in_channels
        for index in range(module_repeats):
            modules.append(
                TimeChannelSeparableConv1d(
                    current,
                    out_channels,
                    kernel=kernel,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    batchnorm_eps=batchnorm_eps,
                )
            )
            if index != module_repeats - 1:
                modules.extend([nn.ReLU(), nn.Dropout(p=dropout)])
            current = out_channels
        self.main = nn.Sequential(*modules)
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels, eps=batchnorm_eps),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        value = self.main(features) + self.residual(features)
        return self.dropout(self.activation(value))


class CnnCtcV18(nn.Module):
    """QuartzNet-15x5 adapted to the project's fixed 64-bin / 39-token contract."""

    def __init__(self, spec: dict, vocab_size: int):
        super().__init__()
        if spec.get("id") != "cnn_ctc_v18":
            raise ValueError("CnnCtcV18 requires cnn_ctc_v18 model spec")
        network = spec["network"]
        if network.get("kind") != "quartznet-15x5-v1":
            raise ValueError("cnn_ctc_v18 requires quartznet-15x5-v1")
        if network.get("normalization") != "batchnorm":
            raise ValueError("cnn_ctc_v18 requires BatchNorm")
        if network.get("separable_convolution") != "depthwise-pointwise":
            raise ValueError("cnn_ctc_v18 requires depthwise-pointwise TCS convolutions")

        dropout = float(network["dropout"])
        batchnorm_eps = float(network["batchnorm_eps"])
        prologue = network["prologue"]
        in_channels = int(spec["frontend"]["mel_bins"])
        prologue_channels = int(prologue["channels"])
        self.prologue = nn.Sequential(
            TimeChannelSeparableConv1d(
                in_channels,
                prologue_channels,
                kernel=int(prologue["kernel"]),
                stride=int(prologue["stride"]),
                dilation=int(prologue["dilation"]),
                padding=int(prologue["padding"]),
                batchnorm_eps=batchnorm_eps,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        in_channels = prologue_channels

        blocks = []
        for group in network["block_groups"]:
            for _ in range(int(group["block_repeats"])):
                out_channels = int(group["channels"])
                blocks.append(
                    QuartzNetBlock(
                        in_channels,
                        out_channels,
                        kernel=int(group["kernel"]),
                        module_repeats=int(group["module_repeats"]),
                        dilation=int(group["dilation"]),
                        padding=int(group["padding"]),
                        dropout=dropout,
                        batchnorm_eps=batchnorm_eps,
                    )
                )
                in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        c2, c3 = network["epilogue"]
        c2_channels = int(c2["channels"])
        self.c2 = nn.Sequential(
            TimeChannelSeparableConv1d(
                in_channels,
                c2_channels,
                kernel=int(c2["kernel"]),
                stride=int(c2["stride"]),
                dilation=int(c2["dilation"]),
                padding=int(c2["padding"]),
                batchnorm_eps=batchnorm_eps,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        c3_channels = int(c3["channels"])
        self.c3 = nn.Sequential(
            nn.Conv1d(c2_channels, c3_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(c3_channels, eps=batchnorm_eps),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.projection = nn.Conv1d(
            c3_channels,
            vocab_size,
            kernel_size=int(network["projection_kernel"]),
            bias=True,
        )

    def forward(self, features):
        encoded = self.prologue(features)
        encoded = self.blocks(encoded)
        encoded = self.c2(encoded)
        encoded = self.c3(encoded)
        return self.projection(encoded).transpose(1, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
