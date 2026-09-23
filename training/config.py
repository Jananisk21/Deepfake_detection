"""
Central configuration for the whole project.

Every script (preprocessing, training, evaluation, the Streamlit app)
imports its settings from here instead of hard-coding values, so the
project can be re-tuned for a weaker/stronger machine by editing one file.

This is a deliberately simple dataclass-based config (no external config
library) so it stays easy to read and modify for a final-year project.
"""
import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")


@dataclass
class DataConfig:
    # ---- video / frame sampling ----
    frames_per_video: int = 8          # temporal sequence length (configurable)
    frame_sample_stride: int = 1       # take every Nth sampled frame position (kept simple)
    image_size: int = 224              # spatial branch input resolution (RGB face crop)
    face_margin: float = 0.10          # extra margin kept around detected face box (paper: ~10%)

    # ---- frequency representation ----
    freq_type: str = "fft"             # "fft" or "dct"
    freq_size: int = 224                # frequency map resolution (kept same as RGB for simplicity)
    log_scale_fft: bool = True

    # ---- face detector ----
    face_detector: str = "mtcnn"       # "mtcnn" (facenet-pytorch) or "haar" (OpenCV fallback)

    # ---- classes ----
    real_label: int = 0
    fake_label: int = 1
    manipulation_methods: List[str] = field(
        default_factory=lambda: ["DeepFakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    )


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"           # "adam" | "adamw" | "sgd"
    scheduler: str = "cosine"          # "cosine" | "step" | "none"
    early_stopping_patience: int = 5
    seed: int = 42
    num_workers: int = 2
    use_class_weights: bool = True
    grad_clip: float = 1.0
    fusion_type: str = "gated"         # "gated" (proposed) | "concat" (baseline) | "cross_attention"
    branch_mode: str = "dual"          # "dual" | "spatial_only" | "frequency_only"
    spatial_backbone: str = "mobilenet_v3_small"  # lightweight, practical on CPU
    freeze_backbone_layers: bool = True  # freeze early pretrained layers to keep training light


CFG_DATA = DataConfig()
CFG_TRAIN = TrainConfig()

for _dir in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, CHECKPOINT_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
