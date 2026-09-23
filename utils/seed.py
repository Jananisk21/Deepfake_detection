"""
Reproducibility utilities.

Sets random seeds for Python's `random`, NumPy and PyTorch (CPU + CUDA) so that
training runs, dataset splits and augmentation are repeatable across runs.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all relevant random seeds for reproducibility.

    Note: full determinism on GPU also requires disabling cuDNN
    auto-tuning (`torch.backends.cudnn.benchmark = False`), which we do
    here. This can slightly slow down training but makes results
    reproducible, which matters more for a student project / viva.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    """Auto-detect CUDA GPU, otherwise fall back to CPU.

    The project must run on a CPU-only student laptop, so every training
    and inference script calls this instead of hard-coding a device.
    """
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
