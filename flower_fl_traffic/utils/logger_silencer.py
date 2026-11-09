"""Configure process-wide logging, warning, and thread limits for experiments."""

import logging
import os
import warnings

def silence_log():
    """Reduce routine Ray, Flower, Lightning, PyTorch, and warning output.

    Environment variables are set before logger-level changes so child
    processes inherit the same limits when they are started afterward.
    """
    # --- 1. Environment variables (Must be set BEFORE imports if possible) ---
    os.environ["RAY_DEDUP_LOGS"] = "1"              # Deduplicate logs
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Critical for avoiding the 'atexit' PyTorch garbage we saw
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["TORCH_LOGS"] = "-all"
    os.environ["TORCH_COMPILE_DEBUG"] = "0"

    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    # CPU threading limits (Keep these, they are good)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # --- 2. Python Warnings ---
    warnings.filterwarnings("ignore") 
    logging.captureWarnings(True) # Captures warnings into the logging system
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

    # Lightning specifikus loggerek
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.trainer").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities").setLevel(logging.ERROR)

    # --- 3. Logging Levels ---
    # Set all noise-makers to ERROR only
    for logger_name in ["flwr", "ray", "torch", "filelock"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False # Prevent logs from bubbling up to console