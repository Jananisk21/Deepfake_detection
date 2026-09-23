"""
Feature fusion mechanisms combining the spatial and frequency branches.

Paper basis: Section 2.3 reviews D4-Net's design, which "aligns an
Xception-derived semantic branch with an FFT-magnitude-based MesoNet
frequency branch through cross-attention, followed by an adaptive fusion
block that learns to up- or down-weight frequency evidence on a
per-location, per-channel basis", and reports that naive concatenation
consistently underperforms learned/adaptive fusion across the reviewed
literature (Section 2.3, and the "frequency-routing sensitivity" gap in
Section 5).

Proposed implementation decision -- three interchangeable fusion modules,
so the project can reproduce that exact "adaptive fusion beats
concatenation" comparison as its own experiment (see evaluation/, baseline
comparison table):

1. ConcatFusion       -- baseline: naive concatenation + FC projection.
                         No interpretable per-branch weighting.
2. GatedFusion (PROPOSED / default) -- a lightweight adaptive gating
   mechanism: a small MLP looks at both projected branch features and
   predicts a per-channel gate g in [0,1] that decides, channel by
   channel, how much to trust the spatial branch versus the frequency
   branch. Averaging g over channels gives a single interpretable
   "spatial contribution %" vs "frequency contribution %" per sample --
   this is what directly answers the paper's identified gap of
   branch-attributable explainability (Section 5/9 of the brief), without
   needing a heavier attention mechanism.
3. CrossAttentionFusion -- an optional, heavier alternative loosely
   inspired by D4-Net's cross-attention block, implemented with a single
   2-token nn.MultiheadAttention (token 0 = spatial, token 1 =
   frequency). Its attention weights double as a second, independent way
   to read out branch contribution. Included because the paper explicitly
   names cross-attention as the mechanism used in the architecture we
   take closest inspiration from, but it is not the default because on a
   CPU-only student machine the simpler gated fusion trains faster and is
   easier to explain in a viva while still being "adaptive" fusion rather
   than static concatenation.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Baseline fusion: concatenate projected features, then one FC layer.
    No learned per-branch weighting -> used as the ablation baseline in
    the comparison table (Section 12 of the brief)."""

    def __init__(self, spatial_dim: int, freq_dim: int, fused_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.fused_dim = fused_dim

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict]]:
        fused = self.proj(torch.cat([spatial_feat, freq_feat], dim=-1))
        return fused, None  # no interpretable branch-contribution signal


class GatedFusion(nn.Module):
    """Proposed adaptive fusion: per-channel learned gate between the two
    branches, computed from both branches jointly (so the gate is
    input-dependent / "adaptive", not a fixed learned scalar)."""

    def __init__(self, spatial_dim: int, freq_dim: int, fused_dim: int = 256):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, fused_dim)
        self.freq_proj = nn.Linear(freq_dim, fused_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.fused_dim = fused_dim

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        s = self.spatial_proj(spatial_feat)   # (B, fused_dim)
        f = self.freq_proj(freq_feat)         # (B, fused_dim)
        gate = self.gate_net(torch.cat([s, f], dim=-1))   # (B, fused_dim), in [0,1]
        fused = gate * s + (1.0 - gate) * f
        # spatial_weight (per sample) = mean gate value across channels
        spatial_weight = gate.mean(dim=-1)          # (B,) in [0,1]
        freq_weight = 1.0 - spatial_weight
        return fused, {"gate": gate, "spatial_weight": spatial_weight, "freq_weight": freq_weight}


class CrossAttentionFusion(nn.Module):
    """Optional heavier fusion: 2-token cross-attention between the two
    branches (loosely inspired by D4-Net's cross-attention alignment
    step). Each branch attends to the other; the two attended
    representations are concatenated and projected."""

    def __init__(self, spatial_dim: int, freq_dim: int, fused_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, fused_dim)
        self.freq_proj = nn.Linear(freq_dim, fused_dim)
        self.attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Sequential(nn.Linear(fused_dim * 2, fused_dim), nn.ReLU(inplace=True))
        self.fused_dim = fused_dim

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        s = self.spatial_proj(spatial_feat).unsqueeze(1)  # (B,1,D)
        f = self.freq_proj(freq_feat).unsqueeze(1)        # (B,1,D)
        tokens = torch.cat([s, f], dim=1)                  # (B,2,D) : token0=spatial, token1=freq
        attended, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        # attn_weights: (B, 2, 2) -- how much each token attends to each other token
        fused = self.out_proj(attended.flatten(1))
        # branch contribution read-out: how much attention mass each token RECEIVES on average
        received = attn_weights.mean(dim=1)  # (B,2) -> [spatial_received, freq_received]
        spatial_weight = received[:, 0]
        freq_weight = received[:, 1]
        return fused, {"attn_weights": attn_weights, "spatial_weight": spatial_weight, "freq_weight": freq_weight}


def build_fusion(fusion_type: str, spatial_dim: int, freq_dim: int, fused_dim: int = 256) -> nn.Module:
    if fusion_type == "concat":
        return ConcatFusion(spatial_dim, freq_dim, fused_dim)
    elif fusion_type == "gated":
        return GatedFusion(spatial_dim, freq_dim, fused_dim)
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(spatial_dim, freq_dim, fused_dim)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}")
