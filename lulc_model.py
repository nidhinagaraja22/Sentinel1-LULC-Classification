"""A compact U-Net-style model for Sentinel-1 semantic segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two convolutions that learn spatial patterns."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class SmallUNet(nn.Module):
    """Convert two radar channels into per-pixel class scores."""

    def __init__(self, num_classes, in_channels=2):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder: learn patterns at progressively smaller resolutions.
        self.encoder1 = ConvBlock(in_channels, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)

        self.bottleneck = ConvBlock(128, 256)

        # Decoder: combine enlarged features with encoder features.
        self.decoder3 = ConvBlock(256 + 128, 128)
        self.decoder2 = ConvBlock(128 + 64, 64)
        self.decoder1 = ConvBlock(64 + 32, 32)

        # One output score per class at every pixel.
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def combine(features, skip):
        features = F.interpolate(
            features,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat([features, skip], dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        center = self.bottleneck(self.pool(e3))

        d3 = self.decoder3(self.combine(center, e3))
        d2 = self.decoder2(self.combine(d3, e2))
        d1 = self.decoder1(self.combine(d2, e1))

        # Return raw scores. CrossEntropyLoss handles their conversion.
        return self.output(d1)
