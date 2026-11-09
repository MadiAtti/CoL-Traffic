"""Seed Python and numerical libraries for reproducible experiments."""

import random

import numpy as np
import torch


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch and enable deterministic cuDNN behavior.

    Args:
        seed: Integer seed applied to each supported random-number generator.

    Note:
        Deterministic cuDNN operations can reduce training performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CUDA operations, which can help with reproducibility but may impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False