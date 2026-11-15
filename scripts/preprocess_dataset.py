#!/usr/bin/env python3
"""
Batch-preprocesses every video listed in a split CSV (train.csv / val.csv /
test.csv, produced by preprocessing/dataset_split.py) into cached .npz
files under data/processed/<split>/.

Usage:
    python scripts/preprocess_dataset.py --split train --manifest data/splits/train.csv
    python scripts/preprocess_dataset.py --split val   --manifest data/splits/val.csv
    python scripts/preprocess_dataset.py --split test  --manifest data/splits/test.csv

Add --limit N to only process the first N videos (useful for a quick
smoke test / demo run before committing to processing an entire dataset).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.face_detector import build_face_detector
from preprocessing.video_processor import process_video, save_processed_video
from training.config import CFG_DATA, PROCESSED_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--manifest", required=True, help="CSV with columns video_path,label,dataset,method,source_ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num_frames", type=int, default=CFG_DATA.frames_per_video)
    ap.add_argument("--image_size", type=int, default=CFG_DATA.image_size)
    ap.add_argument("--freq_type", default=CFG_DATA.freq_type, choices=["fft", "dct"])
    ap.add_argument("--face_detector", default=CFG_DATA.face_detector, choices=["mtcnn", "haar"])
    ap.add_argument("--out_dir", default=None, help="Custom output directory (e.g. data/processed/ffpp/train). Defaults to data/processed/<split>")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(PROCESSED_DIR, args.split)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    detector = build_face_detector(args.face_detector)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    n_ok, n_fail = 0, 0
    for i, row in enumerate(rows):
        video_path = row["video_path"]
        if not os.path.isabs(video_path) and not os.path.exists(video_path):
            alt_path = os.path.join(project_root, video_path)
            if os.path.exists(alt_path):
                video_path = alt_path
        label = int(row["label"])
        video_id = f"{row['dataset']}_{row['method']}_{os.path.splitext(os.path.basename(video_path))[0]}_{i}"
        try:
            pv = process_video(
                video_path=video_path,
                video_id=video_id,
                label=label,
                detector=detector,
                num_frames=args.num_frames,
                image_size=args.image_size,
                freq_type=args.freq_type,
                freq_size=args.image_size,
                face_margin=CFG_DATA.face_margin,
                log_scale_fft=CFG_DATA.log_scale_fft,
            )
            if pv is None:
                n_fail += 1
                print(f"[{i+1}/{len(rows)}] SKIP (unreadable video): {video_path}")
                continue
            save_processed_video(pv, out_dir)
            n_ok += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(rows):
                elapsed = time.time() - t0
                print(f"[{i+1}/{len(rows)}] processed ({n_ok} ok, {n_fail} failed) | {elapsed:.1f}s elapsed")
        except Exception as e:
            n_fail += 1
            print(f"[{i+1}/{len(rows)}] ERROR on {video_path}: {e}")

    print(f"\nDone. {n_ok} videos cached to {out_dir}, {n_fail} failed/skipped.")


if __name__ == "__main__":
    main()
