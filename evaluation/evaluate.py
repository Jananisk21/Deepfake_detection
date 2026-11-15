#!/usr/bin/env python3
"""
Final evaluation of a trained checkpoint on a held-out split (normally the
test split, sometimes a second dataset for the cross-dataset generalization
experiment -- see evaluation/generalization.md in the README section).

Produces:
    - VIDEO-LEVEL metrics (the model's native prediction: the whole T-frame
      clip goes through spatial+frequency branches, fusion, and the GRU
      temporal aggregator to produce ONE prediction per video). This is
      the headline number for the project, since the brief specifically
      asks for video-level, not just frame-level, evaluation.
    - FRAME-LEVEL metrics, computed by running the SAME trained model on
      each individual cached frame as a length-1 sequence (T=1). This is
      NOT a separate frame-level classifier -- it reuses the trained
      video-level model with its temporal module degenerating to a no-op
      on a single frame -- but it gives a per-frame signal that is
      explicitly kept separate from, and not comparable one-to-one with,
      the video-level number (Section 13 of the brief: report both, but
      clearly distinguish them).

Usage:
    python evaluation/evaluate.py --checkpoint outputs/checkpoints/dual_gated_best.pt \\
        --test_dir data/processed/test --run_name dual_gated
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from evaluation.metrics import compute_all_metrics
from models.dual_branch_model import DualBranchDeepfakeDetector
from training.config import FIGURES_DIR, LOGS_DIR
from training.dataset import DeepfakeVideoDataset
from training.validate import run_validation
from utils.seed import get_device


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = dict(ckpt.get("args", {}))
    args["optimal_threshold"] = ckpt.get("optimal_threshold", 0.5)
    args["val_roc_auc"] = ckpt.get("val_roc_auc", None)
    args["val_loss"] = ckpt.get("val_loss", None)
    args["val_acc"] = ckpt.get("val_acc", None)
    model = DualBranchDeepfakeDetector(
        branch_mode=args.get("branch_mode", "dual"),
        fusion_type=args.get("fusion_type", "gated"),
        spatial_backbone=args.get("spatial_backbone", "mobilenet_v3_small"),
        pretrained_spatial=False,  # weights are overwritten by the checkpoint anyway
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, args


@torch.no_grad()
def frame_level_eval(model, dataset: DeepfakeVideoDataset, device) -> dict:
    all_probs, all_labels = [], []
    for i in range(len(dataset)):
        rgb, freq, label = dataset[i]         # rgb: (T,3,H,W), freq: (T,1,H,W)
        T = rgb.shape[0]
        rgb_frames = rgb.unsqueeze(1).to(device)   # (T,1,3,H,W) -> batch dim = T, seq_len = 1
        freq_frames = freq.unsqueeze(1).to(device)  # (T,1,1,H,W)
        out = model(rgb_frames, freq_frames)
        probs = torch.sigmoid(out["logit"]).cpu().numpy()  # (T,)
        all_probs.extend(probs.tolist())
        all_labels.extend([label.item()] * T)
    return {"probs": np.array(all_probs), "labels": np.array(all_labels)}


def plot_confusion_matrix(cm, out_path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(roc_curve_dict, auc_value, out_path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(roc_curve_dict["fpr"], roc_curve_dict["tpr"], label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_path, out_path, title):
    if not os.path.exists(history_path):
        return
    with open(history_path) as f:
        h = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(h["train_loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(h["train_acc"], label="train")
    axes[1].plot(h["val_acc"], label="val")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--run_name", default="eval")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=None, help="Decision threshold. Defaults to optimal_threshold stored in checkpoint (or 0.5).")
    ap.add_argument("--skip_frame_level", action="store_true", help="Frame-level eval re-runs the model once per frame; skip for speed on large test sets.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = get_device(force_cpu=args.cpu)
    model, train_args = load_model(args.checkpoint, device)
    effective_thresh = args.threshold if args.threshold is not None else train_args.get("optimal_threshold", 0.5)
    print(f"Loaded checkpoint {args.checkpoint} (trained with args: {train_args})")
    print(f"Using classification threshold: {effective_thresh:.3f}")

    test_ds = DeepfakeVideoDataset(args.test_dir, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- video-level (native) evaluation ----
    video_result = run_validation(model, test_loader, device, criterion=None)
    video_metrics = compute_all_metrics(video_result["labels"], video_result["probs"], threshold=effective_thresh)
    print("\n=== VIDEO-LEVEL metrics (native model output) ===")
    for k, v in video_metrics.items():
        if k not in ("roc_curve", "confusion_matrix"):
            print(f"  {k}: {v}")

    if video_metrics.get("confusion_matrix"):
        plot_confusion_matrix(
            np.array(video_metrics["confusion_matrix"]),
            os.path.join(FIGURES_DIR, f"{args.run_name}_video_confusion_matrix.png"),
            f"{args.run_name} - Video-level Confusion Matrix",
        )
    if video_metrics.get("roc_curve"):
        plot_roc_curve(
            video_metrics["roc_curve"], video_metrics["auc"],
            os.path.join(FIGURES_DIR, f"{args.run_name}_video_roc_curve.png"),
            f"{args.run_name} - Video-level ROC Curve",
        )

    results = {"video_level": {k: v for k, v in video_metrics.items()}}

    # ---- frame-level evaluation (see module docstring for the caveat) ----
    if not args.skip_frame_level:
        frame_result = frame_level_eval(model, test_ds, device)
        frame_metrics = compute_all_metrics(frame_result["labels"], frame_result["probs"], threshold=effective_thresh)
        print("\n=== FRAME-LEVEL metrics (same model, single-frame sequences; NOT directly comparable to video-level) ===")
        for k, v in frame_metrics.items():
            if k not in ("roc_curve", "confusion_matrix"):
                print(f"  {k}: {v}")
        results["frame_level"] = frame_metrics
        if frame_metrics.get("confusion_matrix"):
            plot_confusion_matrix(
                np.array(frame_metrics["confusion_matrix"]),
                os.path.join(FIGURES_DIR, f"{args.run_name}_frame_confusion_matrix.png"),
                f"{args.run_name} - Frame-level Confusion Matrix",
            )

    history_path = os.path.join(LOGS_DIR, f"{train_args.get('run_name', args.run_name)}_history.json")
    plot_training_curves(history_path, os.path.join(FIGURES_DIR, f"{args.run_name}_training_curves.png"), args.run_name)

    out_json = os.path.join(LOGS_DIR, f"{args.run_name}_eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {out_json}")
    print(f"Saved figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
