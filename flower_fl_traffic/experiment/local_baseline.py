import torch
import json
import os
import shutil
import logging
import pandas as pd
import multiprocessing as mp
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf
from models.neural_network import TrafficLightningModule

# Silence unnecessary Lightning/Torch warnings for cleaner output
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

def _train_single_local_model(args):
    """
    Worker function to train an independent local model using PyTorch Lightning.
    Uses CSVLogger to capture epoch-by-epoch metrics for the JSON report.
    """
    # Restrict each process to a single thread to optimize CPU parallelization
    torch.set_num_threads(1)

    (i, train_loader, test_loaders, config, client_ids, device) = args
    model_key = f"M{i+1}"
    pid = os.getpid()
    
    # Define validation set (the client's own test data)
    val_loader = test_loaders[i]
    temp_log_dir = f"temp_logs_{pid}"
    
    print(f"🚀 [PID:{pid}] Starting training for {model_key}...")
    
    # 1. Initialize Logger to track history
    logger = CSVLogger(save_dir=temp_log_dir, name=model_key)

    # 2. Setup Model and Callbacks
    model = TrafficLightningModule(
        input_dim=config.dataset.input_dim, 
        num_classes=config.dataset.num_classes,
        lr=config.config.lr
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=5,
        mode="max",
        min_delta=0.005,
        verbose=True
    )
    
    # 3. Initialize Trainer
    trainer = L.Trainer(
        max_epochs=config.config.federated_rounds * config.config.num_epochs,
        callbacks=[early_stop_callback],
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=logger
    )
    
    # 4. Perform Training
    trainer.fit(model, train_loader, val_loader)
    
    # 5. Extract Training History from CSV logs
    training_history = []
    log_path = os.path.join(temp_log_dir, model_key, f"version_{logger.version}", "metrics.csv")
    if os.path.exists(log_path):
        metrics_df = pd.read_csv(log_path)
        # Drop empty rows and convert to list of dictionaries
        training_history = metrics_df.where(pd.notnull(metrics_df), None).to_dict(orient="records")

    # 6. Final Multi-Client Evaluation
    evaluation_results = {}
    for j, loader in enumerate(test_loaders):
        target_name = client_ids[j]
        val_metrics = trainer.validate(model, dataloaders=loader, verbose=False)[0]
        evaluation_results[target_name] = {
            "loss": float(val_metrics["val_loss"]),
            "accuracy": float(val_metrics["val_acc"])
        }

    # Cleanup temporary logs
    if os.path.exists(temp_log_dir):
        shutil.rmtree(temp_log_dir)

    return model_key, {"training": training_history, "evaluation": evaluation_results}

def run_local_experiment(config, train_loaders, test_loaders, subdir, client_ids=None):
    """
    Main entry point for local baseline execution. 
    Parallelizes client training using a multiprocessing Pool.
    """
    if client_ids is None:
        client_ids = ["P1", "P2"]

    # Serialization-friendly config
    clean_params = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cpu")
    
    # Prepare task arguments
    tasks = []
    for i, train_loader in enumerate(train_loaders):
        tasks.append((i, train_loader, test_loaders, config, client_ids, device))

    print(f"\n--- Starting Lightning Local Baseline (Subdir: {subdir}) ---")
    
    # Execute training tasks in parallel
    with mp.Pool(processes=len(tasks)) as pool:
        results_list = pool.map(_train_single_local_model, tasks)

    # Compile final results structure
    final_results = {
        "parameters": clean_params,
        "models": {model_key: data for model_key, data in results_list}
    }

    # Save to JSON
    target_path = os.path.join("1_local_baseline", subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")

    with open(file_path, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"✅ Baseline experiment completed. Data saved to: {file_path}")