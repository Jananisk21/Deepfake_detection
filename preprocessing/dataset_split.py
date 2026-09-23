"""
Leakage-safe train/val/test splitting.

The project brief explicitly warns: "Videos from the same source/person
must not be distributed incorrectly across train and test sets. Prefer a
video-level split rather than randomly splitting individual frames."

We go one step further than a plain video-level split, because
FaceForensics++/Celeb-DF manipulated videos are built FROM PAIRS of source
identities (e.g. "identity A's face swapped onto identity B's video").
If we split naively by a single id, a manipulated video can still leak
identity A's face into the test set while identity A's real video sits in
the training set (same face, same person, different split -> leakage).

Algorithm:
  1. Treat every source identity referenced in the manifest as a graph
     node.
  2. For every video that references 2+ ids (i.e. every manipulated
     video), add an edge between those ids -- they must never be
     separated across splits.
  3. Compute connected components (union-find) over this graph. Each
     component is treated as one indivisible "group".
  4. Assign whole groups (not individual ids or videos) to train/val/test
     using a group-level shuffle split, so every video whose ids fall in
     a group ends up entirely in one split.

This guarantees zero identity leakage between splits by construction,
while still allowing manipulated videos with correlated ids to be used
(nothing is dropped, unlike a naive "both ids must be in the same split
independently" rule which would discard a large fraction of the data).
"""
import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def read_manifest(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_leakage_safe_split(
    manifest_rows: List[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Returns {video_path: split_name} covering every row in the manifest."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    uf = UnionFind()
    for row in manifest_rows:
        ids = [i for i in row["source_ids"].split(";") if i]
        for i in ids:
            uf.find(i)  # register node
        for i in ids[1:]:
            uf.union(ids[0], i)

    # group ids by connected component
    components: Dict[str, List[str]] = defaultdict(list)
    for node in uf.parent:
        components[uf.find(node)].append(node)
    group_keys = sorted(components.keys())

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train : n_train + n_val])
    test_groups = set(group_keys[n_train + n_val :])

    id_to_split: Dict[str, str] = {}
    for g in train_groups:
        for node in components[g]:
            id_to_split[node] = "train"
    for g in val_groups:
        for node in components[g]:
            id_to_split[node] = "val"
    for g in test_groups:
        for node in components[g]:
            id_to_split[node] = "test"

    video_to_split: Dict[str, str] = {}
    for row in manifest_rows:
        ids = [i for i in row["source_ids"].split(";") if i]
        if not ids:
            continue
        # all ids of a video are guaranteed (by union) to map to the same split
        video_to_split[row["video_path"]] = id_to_split[ids[0]]

    return video_to_split


def write_split_csvs(manifest_rows: List[dict], video_to_split: Dict[str, str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    per_split_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in manifest_rows:
        split = video_to_split.get(row["video_path"])
        if split:
            per_split_rows[split].append(row)

    fieldnames = ["video_path", "label", "dataset", "method", "source_ids"]
    for split, rows in per_split_rows.items():
        out_path = os.path.join(out_dir, f"{split}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        n_real = sum(1 for r in rows if int(r["label"]) == 0)
        n_fake = sum(1 for r in rows if int(r["label"]) == 1)
        print(f"[dataset_split] {split}: {len(rows)} videos ({n_real} real / {n_fake} fake) -> {out_path}")


def verify_no_leakage(video_to_split: Dict[str, str], manifest_rows: List[dict]) -> bool:
    """Sanity check: no source identity appears in more than one split."""
    id_splits: Dict[str, set] = defaultdict(set)
    for row in manifest_rows:
        split = video_to_split.get(row["video_path"])
        if not split:
            continue
        for i in row["source_ids"].split(";"):
            if i:
                id_splits[i].add(split)
    offenders = {i: s for i, s in id_splits.items() if len(s) > 1}
    if offenders:
        print(f"[dataset_split] LEAKAGE DETECTED for {len(offenders)} ids: {list(offenders.items())[:5]}")
        return False
    print("[dataset_split] OK: no source identity appears in more than one split.")
    return True


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    p = argparse.ArgumentParser(description="Create a leakage-safe train/val/test split from a manifest CSV.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--train_frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from training.config import SPLITS_DIR

    rows = read_manifest(args.manifest)
    mapping = build_leakage_safe_split(rows, args.train_frac, args.val_frac, args.test_frac, args.seed)
    verify_no_leakage(mapping, rows)
    write_split_csvs(rows, mapping, args.out_dir or SPLITS_DIR)
