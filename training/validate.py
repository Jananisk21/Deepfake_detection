"""Validation loop, shared by train.py (per-epoch validation) and
evaluation/evaluate.py (final test-set evaluation)."""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device: torch.device, criterion=None) -> Dict:
    model.eval()
    all_probs, all_labels, all_losses = [], [], []

    for rgb, freq, labels in loader:
        rgb, freq, labels = rgb.to(device), freq.to(device), labels.to(device)
        out = model(rgb, freq)
        logits = out["logit"]
        if criterion is not None:
            loss = criterion(logits, labels)
            all_losses.append(loss.item() * labels.size(0))
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    preds = (probs >= 0.5).astype(int)

    n = len(labels)
    acc = float((preds == labels).mean()) if n else 0.0
    avg_loss = float(np.sum(all_losses) / n) if (criterion is not None and n) else None

    roc_auc = 0.5
    optimal_thresh = 0.5
    best_bal_acc = acc
    if n > 0 and len(np.unique(labels)) == 2:
        try:
            roc_auc = float(roc_auc_score(labels, probs))
        except Exception:
            roc_auc = 0.5

        # Search for optimal threshold maximizing balanced accuracy (Youden's J)
        candidate_thresholds = np.linspace(0.10, 0.90, 81)
        best_score = -1.0
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        for th in candidate_thresholds:
            p_bin = (probs >= th).astype(int)
            tpr = (p_bin[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0.0
            tnr = (p_bin[neg_mask] == 0).mean() if neg_mask.sum() > 0 else 0.0
            score = 0.5 * (tpr + tnr)
            if score > best_score:
                best_score = score
                optimal_thresh = float(th)
                best_bal_acc = float(score)

    preds_opt = (probs >= optimal_thresh).astype(int)
    acc_opt = float((preds_opt == labels).mean()) if n else 0.0

    return {
        "probs": probs,
        "labels": labels,
        "preds": preds,
        "preds_opt": preds_opt,
        "accuracy": acc,
        "accuracy_opt": acc_opt,
        "loss": avg_loss,
        "roc_auc": roc_auc,
        "optimal_threshold": optimal_thresh,
        "balanced_accuracy": best_bal_acc,
    }

