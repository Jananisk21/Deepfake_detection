"""
Metric computation shared by evaluate.py, robustness.py and the baseline
comparison script. Every number here comes from sklearn on actual model
outputs -- nothing in this file invents a value (Section 19 of the brief:
academic integrity / no fabricated results).
"""
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_all_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    """labels: (N,) in {0,1}. probs: (N,) predicted P(fake) in [0,1]."""
    if len(labels) == 0:
        return {"note": "No samples available -- metrics not computed."}

    preds = (probs >= threshold).astype(int)
    result = {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    # AUC needs both classes present
    if len(np.unique(labels)) == 2:
        result["auc"] = float(roc_auc_score(labels, probs))
        fpr, tpr, _ = roc_curve(labels, probs)
        result["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        result["auc"] = None
        result["roc_curve"] = None
        result["note"] = "AUC/ROC not computed: only one class present in this split."

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    result["confusion_matrix"] = cm.tolist()  # [[TN, FP], [FN, TP]]
    return result


def aggregate_frame_to_video(video_ids: list, frame_probs: np.ndarray, agg: str = "mean") -> Dict[str, float]:
    """Aggregates frame-level probabilities into a single video-level score.
    Used only if a caller has raw per-frame predictions rather than the
    model's native video-level output (the dual-branch model already
    predicts at video level via its temporal aggregator, so this is mainly
    for the single-frame baselines / sanity checks)."""
    out: Dict[str, list] = {}
    for vid, p in zip(video_ids, frame_probs):
        out.setdefault(vid, []).append(p)
    if agg == "mean":
        return {vid: float(np.mean(ps)) for vid, ps in out.items()}
    elif agg == "max":
        return {vid: float(np.max(ps)) for vid, ps in out.items()}
    raise ValueError(f"Unknown aggregation: {agg}")
