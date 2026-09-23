"""
Scans a dataset directory (FaceForensics++ or Celeb-DF (V2) layout) and
builds a flat manifest CSV listing every video, its label, its
manipulation method, and its "source identity/identities" — the piece of
information the leakage-safe splitter (dataset_split.py) needs.

Expected FaceForensics++ layout (official release layout):
    <root>/original_sequences/youtube/c23/videos/*.mp4
    <root>/manipulated_sequences/DeepFakes/c23/videos/*.mp4
    <root>/manipulated_sequences/Face2Face/c23/videos/*.mp4
    <root>/manipulated_sequences/FaceSwap/c23/videos/*.mp4
    <root>/manipulated_sequences/NeuralTextures/c23/videos/*.mp4
Manipulated filenames follow FF++'s own "<sourceId>_<targetId>.mp4" convention.

Expected Celeb-DF (V2) layout (official release layout):
    <root>/Celeb-real/*.mp4
    <root>/Celeb-synthesis/*.mp4
    <root>/YouTube-real/*.mp4
Celeb-DF filenames follow "id<k>_..." (real / YouTube-real) or
"id<a>_id<b>_..." (synthesis, i.e. identity <a>'s face swapped onto
identity <b>'s video) conventions.

This module only reads file/folder names -- it never opens the videos --
so it is fast even on the full-size datasets.
"""
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ManifestRow:
    video_path: str
    label: int          # 0 real, 1 fake
    dataset: str         # "ffpp" | "celebdf"
    method: str          # "real" | "DeepFakes" | "Face2Face" | "FaceSwap" | "NeuralTextures" | "synthesis"
    source_ids: str       # semicolon-separated identity ids involved, used for leakage-safe splitting


def _ffpp_source_ids(stem: str, is_real: bool) -> List[str]:
    if is_real:
        return [stem]
    parts = stem.split("_")
    return parts[:2] if len(parts) >= 2 else [stem]


def scan_faceforensics(root: str, methods: List[str], compression: str = "c23") -> List[ManifestRow]:
    rows: List[ManifestRow] = []

    real_dir = os.path.join(root, "original_sequences", "youtube", compression, "videos")
    for path in sorted(glob.glob(os.path.join(real_dir, "*.mp4"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        rows.append(ManifestRow(path, 0, "ffpp", "real", ";".join(_ffpp_source_ids(stem, True))))

    for method in methods:
        fake_dir = os.path.join(root, "manipulated_sequences", method, compression, "videos")
        for path in sorted(glob.glob(os.path.join(fake_dir, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            rows.append(ManifestRow(path, 1, "ffpp", method, ";".join(_ffpp_source_ids(stem, False))))

    return rows


_CELEBDF_ID_RE = re.compile(r"id(\d+)")


def _celebdf_source_ids(stem: str) -> List[str]:
    ids = _CELEBDF_ID_RE.findall(stem)
    return [f"id{i}" for i in ids] if ids else [stem]


def scan_celebdf(root: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []

    for sub, label, method in [("Celeb-real", 0, "real"), ("YouTube-real", 0, "real")]:
        d = os.path.join(root, sub)
        for path in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            rows.append(ManifestRow(path, label, "celebdf", method, ";".join(_celebdf_source_ids(stem))))

    d = os.path.join(root, "Celeb-synthesis")
    for path in sorted(glob.glob(os.path.join(d, "*.mp4"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        rows.append(ManifestRow(path, 1, "celebdf", "synthesis", ";".join(_celebdf_source_ids(stem))))

    return rows


def write_manifest(rows: List[ManifestRow], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "dataset", "method", "source_ids"])
        for r in rows:
            writer.writerow([r.video_path, r.label, r.dataset, r.method, r.source_ids])


def build_manifest(root: str, dataset: str, out_csv: str, methods: List[str] = None) -> str:
    if dataset == "ffpp":
        methods = methods or ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        rows = scan_faceforensics(root, methods)
    elif dataset == "celebdf":
        rows = scan_celebdf(root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'ffpp' or 'celebdf'.")
    if not rows:
        raise RuntimeError(
            f"No videos found under {root} for dataset={dataset}. "
            "Check the folder layout matches the official release structure "
            "(see the docstring at the top of this file)."
        )
    write_manifest(rows, out_csv)
    print(f"[build_manifest] wrote {len(rows)} rows -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    p = argparse.ArgumentParser(description="Build a flat video manifest from a raw dataset folder.")
    p.add_argument("--root", required=True, help="Path to the dataset root (FF++ or Celeb-DF layout).")
    p.add_argument("--dataset", required=True, choices=["ffpp", "celebdf"])
    p.add_argument("--out", default=None, help="Output CSV path. Defaults to data/splits/<dataset>_manifest.csv")
    args = p.parse_args()

    from training.config import SPLITS_DIR

    out_csv = args.out or os.path.join(SPLITS_DIR, f"{args.dataset}_manifest.csv")
    build_manifest(args.root, args.dataset, out_csv)
