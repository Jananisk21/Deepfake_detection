"""
PyTorch Dataset/DataLoader for the cached, preprocessed videos produced by
scripts/preprocess_dataset.py (data/processed/<split>/*.npz).

Each .npz holds:
    rgb   : (T, H, W, 3) float32 in [0, 1]
    freq  : (T, H, W)    float32 in [0, 1]
    label : scalar int (0 real / 1 fake)

This module normalizes RGB with standard ImageNet statistics (since the
spatial branch uses an ImageNet-pretrained backbone) and applies light,
video-consistent augmentation (the SAME random flip/brightness jitter is
applied to every frame of a given video, never a different one per frame,
since a real video's frames must stay visually consistent).
"""
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeepfakeVideoDataset(Dataset):
    def __init__(self, processed_dir: str, augment: bool = False, seed: int = 42):
        self.files: List[str] = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No cached .npz files found in {processed_dir}. "
                "Run scripts/preprocess_dataset.py first."
            )
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def _augment_video(self, rgb: np.ndarray, freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """rgb: (T, H, W, 3) in [0,1], freq: (T, H, W) in [0,1].
        Applies moderate, realistic augmentations to improve generalization
        without destroying fine deepfake manipulation boundaries."""
        # 1. Consistent horizontal flip (applied to both spatial and frequency representations)
        if self._rng.random() < 0.5:
            rgb = rgb[:, :, ::-1, :].copy()
            freq = freq[:, :, ::-1].copy()

        # 2. Moderate brightness scaling (+/- 10%)
        if self._rng.random() < 0.5:
            factor = 1.0 + self._rng.uniform(-0.10, 0.10)
            rgb = np.clip(rgb * factor, 0.0, 1.0)

        # 3. Mild realistic blur (3x3 gaussian blur, simulating natural camera focus/mild motion)
        # Keeps kernel small (3x3) so fine manipulation artifacts are not washed out!
        if self._rng.random() < 0.25:
            for t in range(rgb.shape[0]):
                rgb[t] = cv2.GaussianBlur(rgb[t], (3, 3), 0.5)

        # 4. Mild sensor noise (sigma in [0.005, 0.015])
        if self._rng.random() < 0.20:
            sigma = self._rng.uniform(0.005, 0.015)
            noise = self._rng.normal(0, sigma, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb + noise, 0.0, 1.0)

        return rgb, freq

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        rgb = data["rgb"].astype(np.float32)     # (T,H,W,3) in [0,1]
        freq = data["freq"].astype(np.float32)   # (T,H,W)   in [0,1]
        label = float(data["label"])

        if self.augment:
            rgb, freq = self._augment_video(rgb, freq)

        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        rgb_t = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()      # (T,3,H,W)
        freq_t = torch.from_numpy(freq).unsqueeze(1).float()            # (T,1,H,W)
        label_t = torch.tensor(label, dtype=torch.float32)
        return rgb_t, freq_t, label_t

    def labels(self) -> np.ndarray:
        """Fast label lookup (for class-weight computation) without loading full arrays."""
        out = []
        for f in self.files:
            with np.load(f) as d:
                out.append(int(d["label"]))
        return np.array(out)


def build_dataloader(processed_dir: str, batch_size: int, shuffle: bool, augment: bool, num_workers: int = 2) -> DataLoader:
    ds = DeepfakeVideoDataset(processed_dir, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )


def compute_class_weights(dataset: DeepfakeVideoDataset) -> torch.Tensor:
    """Returns a per-class weight tensor [w_real, w_fake] (inverse frequency),
    used to build a WeightedRandomSampler or a weighted BCE loss when the
    real/fake counts are imbalanced (common in FF++: 1 real per 4 fakes)."""
    labels = dataset.labels()
    n_real = max(1, (labels == 0).sum())
    n_fake = max(1, (labels == 1).sum())
    total = n_real + n_fake
    w_real = total / (2.0 * n_real)
    w_fake = total / (2.0 * n_fake)
    return torch.tensor([w_real, w_fake], dtype=torch.float32)
