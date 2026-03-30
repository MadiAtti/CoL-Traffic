import os
import json
from omegaconf import OmegaConf

def setup_file(config, subdir, base_dir):
    '''
    Initializes the JSON file for saving federated learning experiment results.
    This function creates the necessary directory structure based on the provided base directory and subdirectory, 
    and then creates an initial JSON file with the experiment parameters and an empty list for storing experiment results.
    '''
    target_path = os.path.join(base_dir, subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")
    
    # Create an initial JSON structure with the experiment parameters and an empty list for results
    initial_log = {
        "parameters": OmegaConf.to_container(config, resolve=True),
        "experiments": []
    }
    with open(file_path, "w") as f:
        json.dump(initial_log, f, indent=4)

def save_federated_history(history, config, val1, val2, subdir, base_dir, metric_name):
    """
    Saves the federated learning history to a JSON file.
    val1, val2: the current values for the two clients (noise level OR feature count)
    metric_name: "noise" or "features"
    """
    # Collect round data into a structured format
    rounds_list = []
    for r in range(len(history.losses_distributed)):
        rounds_list.append({
            "round": r + 1,
            "global_evaluation": {
                "P1": {"accuracy": history.metrics_distributed["client1_accuracy"][r][1]},
                "P2": {"accuracy": history.metrics_distributed["client2_accuracy"][r][1]}
            },
            "loss": history.losses_distributed[r][1]
        })

    # Create the experiment entry with the current scenario's parameters and results
    experiment_entry = {
        f"{metric_name}_p1": val1,
        f"{metric_name}_p2": val2,
        "rounds": rounds_list,
        "final_evaluation": {
            "P1": {"accuracy": history.metrics_distributed["client1_accuracy"][-1][1]},
            "P2": {"accuracy": history.metrics_distributed["client2_accuracy"][-1][1]}
        }
    }

    # Get the file path for the current experiment's results
    file_path = os.path.join(base_dir, subdir, f"{config.config.seed}.json")

    # Load and update the existing log file with the new experiment entry
    with open(file_path, "r") as f:
        full_log = json.load(f)
    
    full_log["experiments"].append(experiment_entry)

    # Save the updated log back to the JSON file
    with open(file_path, "w") as f:
        json.dump(full_log, f, indent=4)
    
    return file_path