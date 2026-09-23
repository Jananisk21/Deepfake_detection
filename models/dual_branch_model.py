"""
The proposed model: Dual-Branch Frequency-Aware Deepfake Video Detector.

Assembles, per the pipeline in the project brief:

    RGB frame  -> SpatialBranch   -\
                                     >-- fusion (per frame) -> TemporalAggregator -> classifier -> REAL/FAKE
    FFT/DCT map -> FrequencyBranch -/

This module also implements the two single-branch BASELINE models
(`branch_mode="spatial_only"` / `"frequency_only"`) used for the
Section-12 comparison table, so the proposed dual-branch model and both
baselines share exactly the same temporal aggregator + classifier head --
the only thing that differs between them is what feeds into that head.
That is deliberate: it isolates the effect of "does dual-branch fusion
help" from confounds like "a different classifier head" (i.e. valid,
paper-style ablation practice).
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.spatial_branch import SpatialBranch
from models.frequency_branch import FrequencyBranch
from models.fusion import build_fusion
from models.temporal_model import TemporalAggregator


class DualBranchDeepfakeDetector(nn.Module):
    def __init__(
        self,
        branch_mode: str = "dual",           # "dual" | "spatial_only" | "frequency_only"
        fusion_type: str = "gated",           # "gated" | "concat" | "cross_attention"
        spatial_backbone: str = "mobilenet_v3_small",
        pretrained_spatial: bool = True,
        freeze_backbone_layers: bool = True,
        freq_feature_dim: int = 128,
        fused_dim: int = 256,
        temporal_hidden_dim: int = 128,
        temporal_bidirectional: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()
        assert branch_mode in ("dual", "spatial_only", "frequency_only")
        self.branch_mode = branch_mode
        self.fusion_type = fusion_type

        self.spatial_branch = None
        self.frequency_branch = None
        self.fusion = None

        if branch_mode in ("dual", "spatial_only"):
            self.spatial_branch = SpatialBranch(
                backbone=spatial_backbone, pretrained=pretrained_spatial, freeze_early=freeze_backbone_layers
            )
        if branch_mode in ("dual", "frequency_only"):
            self.frequency_branch = FrequencyBranch(in_channels=1, feature_dim=freq_feature_dim)

        if branch_mode == "dual":
            self.fusion = build_fusion(
                fusion_type, self.spatial_branch.feature_dim, self.frequency_branch.feature_dim, fused_dim
            )
            per_frame_dim = fused_dim
        elif branch_mode == "spatial_only":
            self.branch_proj = nn.Linear(self.spatial_branch.feature_dim, fused_dim)
            per_frame_dim = fused_dim
        else:  # frequency_only
            self.branch_proj = nn.Linear(self.frequency_branch.feature_dim, fused_dim)
            per_frame_dim = fused_dim

        self.temporal = TemporalAggregator(per_frame_dim, hidden_dim=temporal_hidden_dim, bidirectional=temporal_bidirectional)

        self.classifier = nn.Sequential(
            nn.Linear(self.temporal.output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # single logit; sigmoid applied outside (BCEWithLogitsLoss) or in inference helper
        )

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        rgb:  (B, T, 3, H, W) normalized RGB frames
        freq: (B, T, 1, H, W) frequency-domain maps
        Returns a dict with:
            logit            (B,)      raw classification logit (pre-sigmoid)
            frame_weights     (B, T)    temporal attention weight per frame
            spatial_weight    (B, T)    per-frame spatial-branch contribution in [0,1] (dual mode + gated/cross_attention only)
            freq_weight       (B, T)    per-frame frequency-branch contribution in [0,1] (same condition)
        """
        B, T = rgb.shape[0], rgb.shape[1]
        out: Dict[str, torch.Tensor] = {}

        if self.branch_mode in ("dual", "spatial_only"):
            rgb_flat = rgb.reshape(B * T, *rgb.shape[2:])
            spatial_feat = self.spatial_branch(rgb_flat)  # (B*T, Ds)
        if self.branch_mode in ("dual", "frequency_only"):
            freq_flat = freq.reshape(B * T, *freq.shape[2:])
            freq_feat = self.frequency_branch(freq_flat)  # (B*T, Df)

        if self.branch_mode == "dual":
            fused_flat, aux = self.fusion(spatial_feat, freq_feat)  # (B*T, fused_dim)
            if aux is not None and "spatial_weight" in aux:
                out["spatial_weight"] = aux["spatial_weight"].reshape(B, T)
                out["freq_weight"] = aux["freq_weight"].reshape(B, T)
        elif self.branch_mode == "spatial_only":
            fused_flat = self.branch_proj(spatial_feat)
        else:
            fused_flat = self.branch_proj(freq_feat)

        fused_seq = fused_flat.reshape(B, T, -1)  # (B, T, fused_dim)
        video_feat, frame_weights = self.temporal(fused_seq)
        logit = self.classifier(video_feat).squeeze(-1)  # (B,)

        out["logit"] = logit
        out["frame_weights"] = frame_weights
        return out

    @torch.no_grad()
    def predict_proba(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        out = self.forward(rgb, freq)
        return torch.sigmoid(out["logit"])


def build_model(cfg) -> DualBranchDeepfakeDetector:
    """Convenience factory reading straight from a TrainConfig instance."""
    return DualBranchDeepfakeDetector(
        branch_mode=cfg.branch_mode,
        fusion_type=cfg.fusion_type,
        spatial_backbone=cfg.spatial_backbone,
        freeze_backbone_layers=cfg.freeze_backbone_layers,
    )
