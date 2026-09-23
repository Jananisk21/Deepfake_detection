#!/usr/bin/env python3
"""
Full training pipeline for the dual-branch deepfake detector (and its
single-branch baselines -- just pass --branch_mode spatial_only /
frequency_only to train a baseline with the exact same script).

Usage examples:
    # proposed dual-branch model, gated fusion
    python training/train.py --train_dir data/processed/train --val_dir data/processed/val \\
        --epochs 15 --batch_size 8 --branch_mode dual --fusion_type gated --run_name dual_gated

    # baseline: RGB-only
    python training/train.py --train_dir data/processed/train --val_dir data/processed/val \\
        --epochs 15 --branch_mode spatial_only --run_name baseline_rgb

    # baseline: frequency-only
    python training/train.py --train_dir data/processed/train --val_dir data/processed/val \\
        --epochs 15 --branch_mode frequency_only --run_name baseline_freq

    # ablation: naive concatenation fusion instead of the proposed gated fusion
    python training/train.py --train_dir data/processed/train --val_dir data/processed/val \\
        --epochs 15 --branch_mode dual --fusion_type concat --run_name dual_concat

Every run writes:
    outputs/checkpoints/<run_name>_best.pt   (best validation-loss checkpoint)
    outputs/checkpoints/<run_name>_last.pt   (last-epoch checkpoint, for resuming)
    outputs/logs/<run_name>_history.json     (per-epoch train/val loss & accuracy, used for the loss/accuracy curves)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dual_branch_model import DualBranchDeepfakeDetector
from training.config import CFG_DATA, CFG_TRAIN, CHECKPOINT_DIR, LOGS_DIR
from training.dataset import DeepfakeVideoDataset, compute_class_weights
from training.validate import run_validation
from utils.logger import get_logger
from utils.seed import get_device, set_seed


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def build_scheduler(optimizer, cfg, num_epochs: int):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--run_name", default="dual_gated")
    ap.add_argument("--epochs", type=int, default=CFG_TRAIN.epochs)
    ap.add_argument("--batch_size", type=int, default=CFG_TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=CFG_TRAIN.lr)
    ap.add_argument("--weight_decay", type=float, default=CFG_TRAIN.weight_decay)
    ap.add_argument("--optimizer", default=CFG_TRAIN.optimizer, choices=["adam", "adamw", "sgd"])
    ap.add_argument("--scheduler", default=CFG_TRAIN.scheduler, choices=["cosine", "step", "none"])
    ap.add_argument("--early_stopping_patience", type=int, default=CFG_TRAIN.early_stopping_patience)
    ap.add_argument("--num_workers", type=int, default=CFG_TRAIN.num_workers)
    ap.add_argument("--seed", type=int, default=CFG_TRAIN.seed)
    ap.add_argument("--branch_mode", default=CFG_TRAIN.branch_mode, choices=["dual", "spatial_only", "frequency_only"])
    ap.add_argument("--fusion_type", default=CFG_TRAIN.fusion_type, choices=["gated", "concat", "cross_attention"])
    ap.add_argument("--spatial_backbone", default=CFG_TRAIN.spatial_backbone)
    ap.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet-pretrained spatial backbone weights.")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=CFG_TRAIN.grad_clip)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(force_cpu=args.cpu)
    logger = get_logger("train", log_file=os.path.join(LOGS_DIR, f"{args.run_name}.log"))
    logger.info(f"Device: {device}")
    logger.info(f"Run config: {vars(args)}")

    train_ds = DeepfakeVideoDataset(args.train_dir, augment=True, seed=args.seed)
    val_ds = DeepfakeVideoDataset(args.val_dir, augment=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger.info(f"Train videos: {len(train_ds)} | Val videos: {len(val_ds)}")

    model = DualBranchDeepfakeDetector(
        branch_mode=args.branch_mode,
        fusion_type=args.fusion_type,
        spatial_backbone=args.spatial_backbone,
        pretrained_spatial=not args.no_pretrained,
        freeze_backbone_layers=CFG_TRAIN.freeze_backbone_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,} total, {n_trainable:,} trainable")

    if not args.no_class_weights and CFG_TRAIN.use_class_weights:
        class_weights = compute_class_weights(train_ds).to(device)
        # BCEWithLogitsLoss pos_weight balances the positive (fake) class only;
        # class_weights[1]/class_weights[0] converts the two per-class weights
        # into the single pos_weight scalar it expects.
        pos_weight = class_weights[1] / class_weights[0]
        logger.info(f"Class weights: real={class_weights[0]:.3f} fake={class_weights[1]:.3f} -> pos_weight={pos_weight:.3f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    cfg_train_overridden = argparse.Namespace(**{**vars(CFG_TRAIN), **vars(args)})
    optimizer = build_optimizer(model, cfg_train_overridden)
    scheduler = build_scheduler(optimizer, cfg_train_overridden, args.epochs)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    ckpt_best = os.path.join(CHECKPOINT_DIR, f"{args.run_name}_best.pt")
    ckpt_last = os.path.join(CHECKPOINT_DIR, f"{args.run_name}_last.pt")

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        for rgb, freq, labels in train_loader:
            rgb, freq, labels = rgb.to(device), freq.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(rgb, freq)
            loss = criterion(out["logit"], labels)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            preds = (torch.sigmoid(out["logit"]) >= 0.5).long()
            running_correct += (preds == labels.long()).sum().item()
            running_n += bs

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)

        val_result = run_validation(model, val_loader, device, criterion)
        val_loss = val_result["loss"] if val_result["loss"] is not None else float("inf")
        val_acc = val_result["accuracy"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t_start
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s elapsed"
        )

        torch.save(
            {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch},
            ckpt_last,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                ckpt_best,
            )
            logger.info(f"  -> new best model saved (val_loss={val_loss:.4f}) to {ckpt_best}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (no val improvement for {args.early_stopping_patience} epochs).")
                break

    with open(os.path.join(LOGS_DIR, f"{args.run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training finished. Best val_loss={best_val_loss:.4f}. History saved.")


if __name__ == "__main__":
    main()
