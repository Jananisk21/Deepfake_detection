#!/usr/bin/env python3
"""
Generates a tiny SYNTHETIC video dataset, laid out exactly like a
FaceForensics++ release, purely so the rest of the pipeline (manifest
building -> leakage-safe split -> preprocessing -> caching -> dataset
loader -> training -> evaluation -> Grad-CAM -> Streamlit app) can be
exercised end-to-end without needing to download the real (100+ GB)
FaceForensics++ / Celeb-DF datasets first.

*** THIS IS NOT REAL DATA. *** Metrics computed on this data say nothing
about real-world deepfake-detection performance; they only prove the code
runs correctly. See the README's "Academic Integrity" section.

"Real" clips: a face-like cartoon (ellipse + eyes + mouth) on a plain
background, smooth shading, no high-frequency texture added.
"Fake" clips: the same cartoon face but with a faint additive
checkerboard/grid pattern (mimicking the kind of periodic
upsampling/decoder artifact the review paper describes) plus a slightly
different color cast -- giving both the spatial and the frequency branch
a (synthetic, toy) signal to learn from, so a smoke-test training run
actually shows the loss decrease rather than training on pure noise.
"""
import argparse
import os

import cv2
import numpy as np


def _draw_face(canvas: np.ndarray, cx: int, cy: int, size: int, rng: np.random.Generator, fake: bool) -> np.ndarray:
    color_face = (200, 170, 150) if not fake else (150, 190, 200)  # different color cast per class
    cv2.ellipse(canvas, (cx, cy), (size // 2, int(size * 0.62)), 0, 0, 360, color_face, -1)
    eye_dx = size // 5
    eye_dy = -size // 8
    r_eye = max(2, size // 18)
    cv2.circle(canvas, (cx - eye_dx, cy + eye_dy), r_eye, (30, 30, 30), -1)
    cv2.circle(canvas, (cx + eye_dx, cy + eye_dy), r_eye, (30, 30, 30), -1)
    cv2.ellipse(canvas, (cx, cy + size // 5), (size // 6, size // 12), 0, 0, 180, (90, 40, 40), 2)

    if fake:
        # faint periodic checkerboard pattern -> injects a high-frequency
        # spectral cue, loosely inspired by the "checkerboard artifacts of
        # transposed convolutions" the paper attributes to generative decoders
        step = 6
        overlay = canvas.copy()
        for yy in range(0, canvas.shape[0], step):
            for xx in range(0, canvas.shape[1], step):
                if ((yy // step) + (xx // step)) % 2 == 0:
                    overlay[yy : yy + step, xx : xx + step] = np.clip(
                        overlay[yy : yy + step, xx : xx + step].astype(int) + 12, 0, 255
                    ).astype(np.uint8)
        canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

    noise = rng.normal(0, 3, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def make_video(path: str, fake: bool, seed: int, num_frames: int = 12, size: int = 240, fps: int = 8) -> None:
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (size, size))
    cx, cy = size // 2, size // 2
    for t in range(num_frames):
        canvas = np.full((size, size, 3), (30, 60, 30), dtype=np.uint8)  # background
        jitter_x = int(4 * np.sin(t / 2.0))
        jitter_y = int(3 * np.cos(t / 2.0))
        canvas = _draw_face(canvas, cx + jitter_x, cy + jitter_y, int(size * 0.55), rng, fake)
        writer.write(canvas)
    writer.release()


def build_dataset(root: str, num_identities: int = 8, num_fake_pairs_per_method: int = 4, num_frames: int = 12):
    methods = ["DeepFakes", "Face2Face"]  # keep the synthetic set small; real code path supports all 4

    real_dir = os.path.join(root, "original_sequences", "youtube", "c23", "videos")
    os.makedirs(real_dir, exist_ok=True)
    ids = [f"{i:03d}" for i in range(num_identities)]
    for i, vid in enumerate(ids):
        make_video(os.path.join(real_dir, f"{vid}.mp4"), fake=False, seed=1000 + i, num_frames=num_frames)

    for method in methods:
        fake_dir = os.path.join(root, "manipulated_sequences", method, "c23", "videos")
        os.makedirs(fake_dir, exist_ok=True)
        # Spread pairs across MANY different hub ids (rather than always
        # pairing everything with ids[0]) so the union-find grouping used by
        # the leakage-safe splitter produces several small components
        # instead of one giant one -- this keeps the resulting train/val/test
        # split reasonably balanced even on this tiny synthetic demo set.
        for k in range(num_fake_pairs_per_method):
            a = ids[(2 * k) % len(ids)]
            b = ids[(2 * k + 1) % len(ids)]
            make_video(
                os.path.join(fake_dir, f"{a}_{b}.mp4"),
                fake=True,
                seed=2000 + k + hash(method) % 100,
                num_frames=num_frames,
            )

    n_real = len(ids)
    n_fake = len(methods) * num_fake_pairs_per_method
    print(f"[generate_synthetic_dataset] wrote {n_real} real + {n_fake} fake videos under {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Output root. Defaults to data/raw/synthetic_ffpp")
    ap.add_argument("--num_identities", type=int, default=8)
    ap.add_argument("--num_fake_pairs_per_method", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=12)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.config import RAW_DIR

    root = args.root or os.path.join(RAW_DIR, "synthetic_ffpp")
    build_dataset(root, args.num_identities, args.num_fake_pairs_per_method, args.num_frames)
