#!/usr/bin/env python3
"""
Builds the Section-12 baseline comparison table:

    Model                  | Accuracy | Precision | Recall | F1 | AUC

Expects that you have already trained each variant separately with
training/train.py (so this script never invents numbers -- it only loads
real checkpoints and evaluates them on the same held-out test set):

    python training/train.py --branch_mode spatial_only  --run_name baseline_rgb   ...
    python training/train.py --branch_mode frequency_only --run_name baseline_freq  ...
    python training/train.py --branch_mode dual --fusion_type concat --run_name dual_concat ...
    python training/train.py --branch_mode dual --fusion_type gated  --run_name dual_gated  ...

Then:
    python evaluation/compare_baselines.py --test_dir data/processed/test \\
        --checkpoints \\
            "RGB-only (baseline)=outputs/checkpoints/baseline_rgb_best.pt" \\
            "Frequency-only (baseline)=outputs/checkpoints/baseline_freq_best.pt" \\
            "Dual-branch + concat fusion (ablation)=outputs/checkpoints/dual_concat_best.pt" \\
            "Dual-branch + gated fusion (proposed)=outputs/checkpoints/dual_gated_best.pt"

If a listed checkpoint file does not exist, that row is written as
"Not evaluated" rather than skipped silently or fabricated, per the
brief's academic-integrity requirement.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from evaluation.evaluate import load_model
from evaluation.metrics import compute_all_metrics
from training.config import LOGS_DIR
from training.dataset import DeepfakeVideoDataset
from training.validate import run_validation
from utils.seed import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help='List of "Display Name=path/to/checkpoint.pt" entries.',
    )
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = get_device(force_cpu=args.cpu)
    test_ds = DeepfakeVideoDataset(args.test_dir, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    for entry in args.checkpoints:
        name, path = entry.split("=", 1)
        if not os.path.exists(path):
            rows.append({"model": name, "status": "Not evaluated (checkpoint not found)"})
            continue
        model, _ = load_model(path, device)
        result = run_validation(model, test_loader, device, criterion=None)
        metrics = compute_all_metrics(result["labels"], result["probs"])
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auc": metrics.get("auc"),
                "n_samples": metrics["n_samples"],
            }
        )

    header = f"| {'Model':<45} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'AUC':>6} |"
    sep = "|" + "-" * 47 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 8 + "|"
    lines = [header, sep]
    for r in rows:
        if "status" in r:
            lines.append(f"| {r['model']:<45} | {r['status']:>44} |")
        else:
            auc_str = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
            lines.append(
                f"| {r['model']:<45} | {r['accuracy']:>8.4f} | {r['precision']:>9.4f} | "
                f"{r['recall']:>6.4f} | {r['f1']:>6.4f} | {auc_str:>6} |"
            )

    table_str = "\n".join(lines)
    print(table_str)

    out_path = os.path.join(LOGS_DIR, "baseline_comparison_table.md")
    with open(out_path, "w") as f:
        f.write("# Baseline Comparison (video-level test-set metrics)\n\n")
        f.write(table_str + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
