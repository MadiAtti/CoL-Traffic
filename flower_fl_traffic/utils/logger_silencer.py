import logging
import os
import warnings
import ray

def silence_log():
    """Minden felesleges zajt elnyomunk a konzolon (Ray, Flower, Opacus, Torch)."""
    
    # 1. Környezeti változók (Még a Ray/Torch inicializálása előtt)
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Ha esetleg befigyelne a TF
    
    # 2. Python warnings globális némítása
    warnings.filterwarnings("ignore") 
    # Speciális szűrés az Opacus és a Torch backward hook üzeneteire
    warnings.filterwarnings("ignore", category=UserWarning, module="opacus")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # 3. Flower naplózás némítása
    logging.getLogger("flwr").setLevel(logging.ERROR)
    
    # 4. Ray inicializálása csendes módban
    if not ray.is_initialized():
        ray.init(
            logging_level=logging.ERROR, 
            log_to_driver=False,       # Ne küldje a worker logokat a fő képernyőre
            configure_logging=True,    # Engedjük, hogy a Ray saját magát némítsa
            include_dashboard=False    # Dashboard sem kell, kevesebb log
        )