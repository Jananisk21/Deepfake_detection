"""Validation loop, shared by train.py (per-epoch validation) and
evaluation/evaluate.py (final test-set evaluation)."""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
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

    return {"probs": probs, "labels": labels, "preds": preds, "accuracy": acc, "loss": avg_loss}
