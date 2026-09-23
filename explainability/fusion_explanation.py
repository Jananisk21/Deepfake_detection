"""
Fusion-level explanation: turns the model's own learned gate (GatedFusion)
or cross-attention weights (CrossAttentionFusion) into the
"Spatial contribution: 62% / Frequency contribution: 38%" style summary
requested in the project brief.

Paper basis: this directly targets the review's identified gap (Section 5):
"few published systems provide branch-attributable explanations (i.e., how
much of a decision is attributable to the frequency branch versus the
semantic branch) for cross-attention dual-branch architectures." The
numbers here are NOT invented -- they are read directly from the model's
`spatial_weight` / `freq_weight` outputs (models/fusion.py), which are
themselves computed by a learned gate/attention module during the forward
pass on the actual input.

IMPORTANT: this explanation is only available for `branch_mode="dual"`
with `fusion_type in {"gated", "cross_attention"}`. The `concat` fusion
baseline has no interpretable per-branch weighting by construction (that
is precisely the point of comparing it against gated fusion in the
Section-12 ablation) -- callers must handle `available=False` explicitly
rather than assuming a number always exists.
"""
from typing import Dict, Optional

import numpy as np
import torch


def compute_branch_contribution(model_output: Dict[str, torch.Tensor]) -> Dict:
    """
    model_output: the dict returned by DualBranchDeepfakeDetector.forward(),
    for a single video (batch size 1).

    Returns a dict with:
        available          bool
        video_spatial_pct  float or None  (attention-weighted average over frames)
        video_freq_pct     float or None
        per_frame_spatial_pct  list[float] or None
        per_frame_freq_pct     list[float] or None
    """
    if "spatial_weight" not in model_output or "freq_weight" not in model_output:
        return {
            "available": False,
            "reason": "This fusion type (e.g. concat baseline) has no learned per-branch gate/attention weight.",
        }

    spatial_w = model_output["spatial_weight"].detach().cpu().numpy()[0]   # (T,)
    freq_w = model_output["freq_weight"].detach().cpu().numpy()[0]          # (T,)
    frame_w = model_output["frame_weights"].detach().cpu().numpy()[0]       # (T,) temporal attention, sums to 1

    # video-level contribution = temporal-attention-weighted average of the
    # per-frame branch weights, so frames the model trusted more for the
    # final video-level decision also count more towards the reported
    # branch contribution split.
    video_spatial = float(np.sum(spatial_w * frame_w) / (np.sum(frame_w) + 1e-8))
    video_freq = 1.0 - video_spatial

    return {
        "available": True,
        "video_spatial_pct": round(video_spatial * 100, 1),
        "video_freq_pct": round(video_freq * 100, 1),
        "per_frame_spatial_pct": [round(float(x) * 100, 1) for x in spatial_w],
        "per_frame_freq_pct": [round(float(x) * 100, 1) for x in freq_w],
        "temporal_attention": [round(float(x), 4) for x in frame_w],
    }


def describe_contribution_for_ui(result: Dict) -> str:
    if not result.get("available"):
        return f"Branch-level contribution is not available for this model configuration: {result.get('reason', '')}"
    return (
        f"Spatial contribution: {result['video_spatial_pct']}% | "
        f"Frequency contribution: {result['video_freq_pct']}% "
        f"(learned by the model's adaptive fusion gate, not a fixed value)."
    )
