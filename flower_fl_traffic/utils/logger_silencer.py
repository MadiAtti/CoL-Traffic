import logging
import os
import warnings
import ray
import torch

def silence_log():
    """
    Enhanced silence function to handle Ray, Flower, and PyTorch noise.
    """
    # --- 1. Environment variables (Must be set BEFORE imports if possible) ---
    os.environ["RAY_DEDUP_LOGS"] = "1"              # Deduplicate logs
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Critical for avoiding the 'atexit' PyTorch garbage we saw
    os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    os.environ["TORCH_LOGS"] = "-all"
    os.environ["TORCH_COMPILE_DEBUG"] = "0"

    # CPU threading limits (Keep these, they are good)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # --- 2. Python Warnings ---
    warnings.filterwarnings("ignore") 
    logging.captureWarnings(True) # Captures warnings into the logging system
    
    # --- 3. Logging Levels ---
    # Set all noise-makers to ERROR only
    for logger_name in ["flwr", "ray", "torch", "filelock"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False # Prevent logs from bubbling up to console