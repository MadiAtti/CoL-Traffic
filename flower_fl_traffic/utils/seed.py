import random

import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for reproducibility across various libraries (random, numpy, torch).
    This function ensures that the same random seed is used for all relevant libraries, which 
    is crucial for achieving consistent results in machine learning experiments. It also configures 
    PyTorch to ensure deterministic behavior on GPU operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CUDA operations, which can help with reproducibility but may impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False