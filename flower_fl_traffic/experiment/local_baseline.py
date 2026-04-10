import torch
import json
import os
import multiprocessing as mp
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from omegaconf import OmegaConf
from models.neural_network import TrafficLightningModule

def _train_single_local_model(args):
    """
    Trains a single model using PyTorch Lightning with Early Stopping 
    within a separate process.
    """
    # Isolate CPU resources for this specific process
    torch.set_num_threads(1)

    (i, train_loader, test_loaders, config, client_ids, device) = args
    model_key = f"M{i+1}"
    pid = os.getpid()
    
    # Use the client's own test loader as the validation set for Early Stopping
    val_loader = test_loaders[i]
    
    print(f"🚀 [PID:{pid}] Starting Lightning training for {model_key}...")
    
    # 1. Initialize the Lightning Module
    model = TrafficLightningModule(
        input_dim=config.dataset.input_dim, 
        num_classes=config.dataset.num_classes,
        lr=config.config.lr
    )
    
    # 2. Configure Early Stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=False
    )
    
    # 3. Initialize the Trainer
    # We disable progress bars (enable_progress_bar=False) to avoid cluttered terminal output
    trainer = L.Trainer(
        max_epochs=config.config.federated_rounds * config.config.num_epochs,
        callbacks=[early_stop_callback],
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False 
    )
    
    # 4. Run Training (Fit)
    trainer.fit(model, train_loader, val_loader)
    
    print(f"  └─ {model_key} (PID:{pid}): Training finished (Early Stopping might have triggered).")

    # 5. Final Evaluation on all provided test loaders
    evaluation_results = {}
    for j, loader in enumerate(test_loaders):
        target_name = client_ids[j]
        # trainer.validate returns a list of dictionaries containing logged metrics
        val_metrics = trainer.validate(model, dataloaders=loader, verbose=False)[0]
        
        evaluation_results[target_name] = {
            "loss": float(val_metrics["val_loss"]),
            "accuracy": float(val_metrics["val_acc"])
        }

    # Note: training_history can be extracted from trainer.logger if enabled,
    # but for simplicity in this parallel setup, we return evaluation results.
    return model_key, {"evaluation": evaluation_results}

def run_local_experiment(config, train_loaders, test_loaders, subdir, client_ids=None):
    """
    Parallel execution manager for baseline experiments.
    """
    if client_ids is None:
        client_ids = ["P1", "P2"]

    clean_params = OmegaConf.to_container(config, resolve=True)
    device = torch.device("cpu")
    
    tasks = []
    for i, train_loader in enumerate(train_loaders):
        tasks.append((i, train_loader, test_loaders, config, client_ids, device))

    print(f"\n--- Starting Lightning Local Baseline Execution (Subdir: {subdir}) ---")
    
    with mp.Pool(processes=len(tasks)) as pool:
        results_list = pool.map(_train_single_local_model, tasks)

    final_results = {
        "parameters": clean_params,
        "models": {model_key: data for model_key, data in results_list}
    }

    target_path = os.path.join("1_local_baseline", subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")

    with open(file_path, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"✅ Baseline finished and saved to {file_path}")