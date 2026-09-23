#!/usr/bin/env python3
"""
Robustness / post-processing degradation testing.

Paper basis: Section 5 of the review flags "Robustness under compression
and post-processing" as an open gap -- "frequency-domain cues are, by
construction, sensitive to compression, resizing, and re-encoding ... and
several reviewed frequency-robustness studies report that Gaussian blur
and rescaling degrade frequency-aware detectors substantially more than
JPEG compression alone." This module lets us test whether that holds for
OUR trained model, rather than assuming it.

For each degradation, every cached test-video's RGB frames are degraded
IN PIXEL SPACE, the frequency map is RECOMPUTED from the degraded frame
(this matters -- a blurred/recompressed image has a genuinely different
spectrum, so we must not reuse the clean frequency map), and the trained
model is re-evaluated. Metrics are compared against the clean (no
degradation) baseline in a single table.

Degradations implemented (brief Section 15):
    - jpeg        : JPEG re-encode at a given quality factor
    - resize      : downscale then upscale back (simulates low-res redistribution)
    - blur        : Gaussian blur
    - brightness  : global brightness scaling
    - noise       : additive Gaussian sensor-like noise
    - re_encoding : two successive JPEG compressions at different quality
                    factors, a simple proxy for "re-uploaded/re-shared on
                    social media" without requiring a full video-codec
                    re-encode pipeline (ffmpeg) inside this script. If
                    ffmpeg is available on your machine, scripts/ffmpeg_reencode.py
                    (documented in the README) can be used for a true codec
                    re-encode test as an optional extension.

Usage:
    python evaluation/robustness.py --checkpoint outputs/checkpoints/dual_gated_best.pt \\
        --test_dir data/processed/test --run_name dual_gated
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from evaluation.evaluate import load_model
from evaluation.metrics import compute_all_metrics
from preprocessing.frequency_transform import rgb_to_frequency_map
from training.config import CFG_DATA, LOGS_DIR
from training.dataset import IMAGENET_MEAN, IMAGENET_STD
from utils.seed import get_device


def degrade_jpeg(frame_uint8: np.ndarray, quality: int = 30) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def degrade_resize(frame_uint8: np.ndarray, scale: float = 0.5) -> np.ndarray:
    h, w = frame_uint8.shape[:2]
    small = cv2.resize(frame_uint8, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_blur(frame_uint8: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(frame_uint8, (ksize, ksize), 0)


def degrade_brightness(frame_uint8: np.ndarray, factor: float = 1.3) -> np.ndarray:
    return np.clip(frame_uint8.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def degrade_noise(frame_uint8: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, frame_uint8.shape)
    return np.clip(frame_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def degrade_reencoding(frame_uint8: np.ndarray) -> np.ndarray:
    once = degrade_jpeg(frame_uint8, quality=50)
    return degrade_jpeg(once, quality=35)


DEGRADATIONS = {
    "jpeg_q30": lambda f: degrade_jpeg(f, quality=30),
    "resize_0.5x": lambda f: degrade_resize(f, scale=0.5),
    "gaussian_blur_k5": lambda f: degrade_blur(f, ksize=5),
    "brightness_+30%": lambda f: degrade_brightness(f, factor=1.3),
    "gaussian_noise_s12": lambda f: degrade_noise(f, sigma=12.0),
    "re_encoding_x2": degrade_reencoding,
}


@torch.no_grad()
def evaluate_on_degradation(model, npz_files, degrade_fn, device, freq_type, image_size) -> dict:
    all_probs, all_labels = [], []
    for path in npz_files:
        data = np.load(path)
        rgb = data["rgb"]        # (T,H,W,3) float32 in [0,1]
        label = int(data["label"])

        rgb_deg_frames, freq_deg_frames = [], []
        for t in range(rgb.shape[0]):
            frame_uint8 = (rgb[t] * 255).astype(np.uint8)
            if degrade_fn is not None:
                frame_uint8 = degrade_fn(frame_uint8)
            freq_map = rgb_to_frequency_map(frame_uint8, freq_type=freq_type, size=image_size, log_scale=CFG_DATA.log_scale_fft)
            rgb_deg_frames.append(frame_uint8.astype(np.float32) / 255.0)
            freq_deg_frames.append(freq_map)

        rgb_arr = (np.stack(rgb_deg_frames) - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb_arr).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)   # (1,T,3,H,W)
        freq_t = torch.from_numpy(np.stack(freq_deg_frames)).unsqueeze(0).unsqueeze(2).float().to(device)  # (1,T,1,H,W)

        out = model(rgb_t, freq_t)
        prob = torch.sigmoid(out["logit"]).item()
        all_probs.append(prob)
        all_labels.append(label)

    return compute_all_metrics(np.array(all_labels), np.array(all_probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--run_name", default="robustness")
    ap.add_argument("--limit", type=int, default=None, help="Only test the first N videos (for a quick check).")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = get_device(force_cpu=args.cpu)
    model, train_args = load_model(args.checkpoint, device)
    freq_type = CFG_DATA.freq_type
    image_size = CFG_DATA.image_size

    npz_files = sorted(glob.glob(os.path.join(args.test_dir, "*.npz")))
    if args.limit:
        npz_files = npz_files[: args.limit]
    if not npz_files:
        raise FileNotFoundError(f"No cached videos found in {args.test_dir}")

    table = {}
    print("=== Clean (no degradation) baseline ===")
    clean = evaluate_on_degradation(model, npz_files, None, device, freq_type, image_size)
    table["clean"] = clean
    print(f"  accuracy={clean['accuracy']:.4f} f1={clean['f1']:.4f} auc={clean.get('auc')}")

    for name, fn in DEGRADATIONS.items():
        print(f"=== Degradation: {name} ===")
        result = evaluate_on_degradation(model, npz_files, fn, device, freq_type, image_size)
        table[name] = result
        print(f"  accuracy={result['accuracy']:.4f} f1={result['f1']:.4f} auc={result.get('auc')}")

    print("\n=== Robustness summary (accuracy drop relative to clean) ===")
    print(f"{'Degradation':<20}{'Accuracy':<12}{'F1':<12}{'AUC':<12}{'Acc drop':<10}")
    for name, result in table.items():
        drop = "-" if name == "clean" else f"{clean['accuracy'] - result['accuracy']:+.4f}"
        auc_str = f"{result.get('auc'):.4f}" if result.get("auc") is not None else "N/A"
        print(f"{name:<20}{result['accuracy']:<12.4f}{result['f1']:<12.4f}{auc_str:<12}{drop:<10}")

    out_path = os.path.join(LOGS_DIR, f"{args.run_name}_robustness.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nSaved full robustness results -> {out_path}")


if __name__ == "__main__":
    main()
