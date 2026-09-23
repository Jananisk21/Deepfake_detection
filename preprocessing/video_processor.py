"""
End-to-end per-video preprocessing pipeline.

For a single video this module:
  1. reads the video
  2. samples a configurable number of frames (evenly spaced across the clip)
  3. detects + crops the face in each sampled frame (with a centre-crop
     fallback so a single failed detection never crashes the pipeline)
  4. resizes the RGB crop to the model input size and normalises it
  5. computes the frequency-domain representation (FFT or DCT) of the same
     crop
  6. caches the result to a compressed .npz file so training does not
     re-decode video / re-run face detection every epoch

This directly implements the "Video Preprocessing" requirements from the
project brief (frame sampling, face detection, resize, normalize, frequency
transform, caching, configurable parameters). It is our own engineering
pipeline; the paper does not prescribe a concrete implementation, only the
general convention (MTCNN + margin, described in the pipeline's docstring
in face_detector.py).
"""
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from preprocessing.face_detector import (
    BaseFaceDetector,
    center_crop_fallback,
    detect_and_crop,
)
from preprocessing.frequency_transform import rgb_to_frequency_map


@dataclass
class ProcessedVideo:
    video_id: str
    label: int                 # 0 = real, 1 = fake
    rgb_frames: np.ndarray     # (T, H, W, 3) float32 in [0, 1]
    freq_frames: np.ndarray    # (T, H, W)    float32 in [0, 1]
    num_faces_detected: int    # diagnostics: how many of T frames had a real face detection


def sample_frame_indices(total_frames: int, num_samples: int) -> List[int]:
    """Evenly-spaced frame indices across the whole clip (not just the
    first N frames), so short and long videos are sampled comparably."""
    if total_frames <= 0:
        return []
    if total_frames <= num_samples:
        return list(range(total_frames))
    return list(np.linspace(0, total_frames - 1, num_samples).astype(int))


def read_video_frames(video_path: str, indices: List[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx_set = set(indices)
    i = 0
    max_idx = max(indices) if indices else -1
    while cap.isOpened() and i <= max_idx:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i in idx_set:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def process_video(
    video_path: str,
    video_id: str,
    label: int,
    detector: BaseFaceDetector,
    num_frames: int = 8,
    image_size: int = 224,
    freq_type: str = "fft",
    freq_size: int = 224,
    face_margin: float = 0.10,
    log_scale_fft: bool = True,
) -> Optional[ProcessedVideo]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        return None

    indices = sample_frame_indices(total_frames, num_frames)
    raw_frames = read_video_frames(video_path, indices)
    if len(raw_frames) == 0:
        return None

    rgb_out, freq_out = [], []
    n_faces = 0
    for frame in raw_frames:
        face = detect_and_crop(detector, frame, margin=face_margin, target_size=(image_size, image_size))
        if face is None:
            face = center_crop_fallback(frame, target_size=(image_size, image_size))
        else:
            n_faces += 1
        freq_map = rgb_to_frequency_map(face, freq_type=freq_type, size=freq_size, log_scale=log_scale_fft)
        rgb_out.append(face.astype(np.float32) / 255.0)
        freq_out.append(freq_map)

    # pad by repeating the last frame if the video was shorter than num_frames
    while len(rgb_out) < num_frames:
        rgb_out.append(rgb_out[-1])
        freq_out.append(freq_out[-1])

    return ProcessedVideo(
        video_id=video_id,
        label=label,
        rgb_frames=np.stack(rgb_out[:num_frames], axis=0),
        freq_frames=np.stack(freq_out[:num_frames], axis=0),
        num_faces_detected=n_faces,
    )


def save_processed_video(pv: ProcessedVideo, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pv.video_id}.npz")
    np.savez_compressed(
        out_path,
        rgb=pv.rgb_frames,
        freq=pv.freq_frames,
        label=pv.label,
        num_faces_detected=pv.num_faces_detected,
    )
    return out_path


def load_processed_video(path: str) -> ProcessedVideo:
    data = np.load(path)
    video_id = os.path.splitext(os.path.basename(path))[0]
    return ProcessedVideo(
        video_id=video_id,
        label=int(data["label"]),
        rgb_frames=data["rgb"],
        freq_frames=data["freq"],
        num_faces_detected=int(data["num_faces_detected"]),
    )
