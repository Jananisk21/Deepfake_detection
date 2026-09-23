"""
Frequency-domain branch.

Paper basis: Section 2.2 discusses MesoNet (Afchar et al., 2018) as a
lightweight "mesoscopic-level" CNN that the reviewed literature frequently
repurposes as a compact frequency-branch encoder in hybrid systems
(explicitly called out in the review: "Lightweight convolutional designs
such as MesoNet ... are frequently repurposed as compact frequency-branch
encoders in later hybrid systems"). D4-Net specifically pairs an
FFT-magnitude input with a MesoNet-style frequency branch.

Proposed implementation decision: we implement a MesoNet-style 4-conv-block
CNN (our own from-scratch implementation, following the architecture
described in the literature: alternating conv+BN+ReLU+pool blocks with a
small number of channels) trained from scratch on the frequency
representation. It is intentionally NOT the same network as the spatial
branch (the brief explicitly requires the branches to stay architecturally
separate before fusion, and the paper's Section 5/6 flags "frequency-
routing sensitivity" -- feeding FFT data into the wrong branch, or routing
it through a branch built for RGB statistics, degrades F1 substantially in
the reviewed ablations).
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, pool=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


class FrequencyBranch(nn.Module):
    """MesoNet-style lightweight CNN operating on the FFT/DCT magnitude map.

    Input: (B, 1, H, W) single-channel frequency representation.
    Output: (B, feature_dim) frequency feature vector per frame.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 8, kernel_size=3, pool=2)
        self.block2 = ConvBlock(8, 8, kernel_size=5, pool=2)
        self.block3 = ConvBlock(8, 16, kernel_size=5, pool=2)
        self.block4 = ConvBlock(16, 16, kernel_size=5, pool=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(16, feature_dim)
        self.feature_dim = feature_dim
        self._last_conv_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W). Returns (B, feature_dim)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        self._last_conv_features = x  # for frequency-branch Grad-CAM / saliency
        pooled = self.pool(x).flatten(1)
        return self.proj(pooled)

    def get_last_conv_features(self) -> torch.Tensor:
        return self._last_conv_features

    def get_target_layer(self) -> nn.Module:
        """The conv module Grad-CAM should hook (see explainability/gradcam.py)."""
        return self.block4.conv
