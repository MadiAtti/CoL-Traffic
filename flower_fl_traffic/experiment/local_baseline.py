import torch
import torch.nn as nn
import json
import os
import multiprocessing as mp
from omegaconf import OmegaConf
from models.neural_network import TrafficNN
from utils.evaluation import evaluate_model

def _train_single_local_model(args):
    """
    Helper function to train a single local model in a separate process.
    This ensures that independent client training can happen in parallel.
    """

    torch.set_num_threads(1)  # Ensure that each process uses only one thread to prevent oversubscription of CPU resources

    (i, train_loader, test_loaders, config, client_ids, device) = args
    model_key = f"M{i+1}"
    pid = os.getpid()
    
    print(f"🚀 [PID:{pid}] Starting parallel local training for {model_key}...")
    
    # Initialize model, optimizer and loss function locally within the process
    model = TrafficNN(input_dim=config.dataset.input_dim, num_classes=config.dataset.num_classes)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.config.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Total epochs calculated the same way as in the sequential version
    total_epochs = config.config.federated_rounds * config.config.num_epochs 
    training_history = []

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        
        # Track metrics for each epoch
        training_history.append({
            "epoch": epoch,
            "loss": running_loss / total,
            "accuracy": correct / total
        })
        
        if epoch % 20 == 0:
            print(f"  └─ {model_key} (PID:{pid}): Epoch {epoch}/{total_epochs} completed.")

    # Evaluation on all provided test loaders (P1, P2, etc.)
    evaluation_results = {}
    for j, test_loader in enumerate(test_loaders):
        target_name = client_ids[j]
        loss, acc = evaluate_model(model, test_loader, criterion)
        evaluation_results[target_name] = {
            "loss": float(loss),
            "accuracy": float(acc)
        }

    return model_key, {"training": training_history, "evaluation": evaluation_results}

def run_local_experiment(config, train_loaders, test_loaders, subdir, client_ids=None):
    """
    Runs local baseline experiments in parallel using a multiprocessing Pool.
    Each client trains its model independently on its own CPU core.
    """
    if client_ids is None:
        client_ids = ["P1", "P2"]

    # Convert OmegaConf to container for easier handling and serialization
    clean_params = OmegaConf.to_container(config, resolve=True)
    
    # Device selection (CPU is recommended for parallel local training to avoid OOM)
    device = torch.device("cpu")
    
    # Prepare arguments for each worker process
    tasks = []
    for i, train_loader in enumerate(train_loaders):
        tasks.append((i, train_loader, test_loaders, config, client_ids, device))

    print(f"\n--- Starting Parallel Local Baseline Execution (Subdir: {subdir}) ---")
    
    # Initialize a pool of workers. The number of processes matches the number of tasks (clients).
    with mp.Pool(processes=len(tasks)) as pool:
        # map() handles the distribution of tasks and collects the returned values
        results_list = pool.map(_train_single_local_model, tasks)

    # Reconstruct the final results dictionary from the returned list of tuples
    final_results = {
        "parameters": clean_params,
        "models": {model_key: data for model_key, data in results_list}
    }

    # Save results to a JSON file
    target_path = os.path.join("1_local_baseline", subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")

    with open(file_path, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"✅ Parallel Local Baseline finished and saved to {file_path}")