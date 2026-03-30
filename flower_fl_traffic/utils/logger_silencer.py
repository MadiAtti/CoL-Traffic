import logging
import os
import warnings
import ray

def silence_log():
    """
    Function to silence logs from Ray, Flower, and other libraries to keep the output clean during federated learning experiments.
    This function sets environment variables, configures Python warnings, and adjusts logging levels to minimize unnecessary output from the libraries used in the project.
    """
    
    # Environment variables to reduce logging from Ray and other libraries
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Filter oit warnings form python, opacus, and torch to keep the output clean during experiments
    warnings.filterwarnings("ignore") 
    warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Turn off logging for Flower to minimize output during federated learning experiments
    logging.getLogger("flwr").setLevel(logging.ERROR)
    
    # Initialize Ray with logging disabled to prevent worker logs from cluttering the output during federated learning simulations. This is especially important when running multiple rounds of training, as Ray can produce a lot of log messages by default.
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR, 
            log_to_driver=False,       # Disable logging to the driver to prevent cluttering the console output
            configure_logging=True,    # Configure logging to ensure that the logging level is set correctly for all Ray components
            include_dashboard=False    # Disable the Ray dashboard to reduce resource usage and prevent additional log messages related to the dashboard
        )