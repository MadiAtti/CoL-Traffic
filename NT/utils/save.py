"""Initialize and atomically update per-seed federated experiment JSON logs."""

import os
import json
import tempfile
from omegaconf import OmegaConf

def setup_file(config, subdir, base_dir):
    """Create or validate the result log for the configured experiment.

    Args:
        config: Resolved Hydra configuration, including its seed.
        subdir: Result subdirectory within ``base_dir``.
        base_dir: Root directory where experiment results are stored.

    Returns:
        The path to the per-seed JSON log.

    Raises:
        ValueError: If an existing log is malformed or uses different
            experiment parameters.
    """
    target_path = os.path.join(base_dir, subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")
    parameters = OmegaConf.to_container(config, resolve=True)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_log = json.load(f)
        if not isinstance(existing_log.get("experiments"), list):
            raise ValueError(f"Invalid experiment log structure: {file_path}")
        if existing_log.get("parameters") != parameters:
            raise ValueError(
                f"Existing experiment results use different parameters: {file_path}"
            )
        return file_path

    initial_log = {
        "parameters": parameters,
        "experiments": []
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(initial_log, f, indent=4)
    return file_path

def save_federated_history(history, config, val1, val2, subdir, base_dir, metric_name):
    """Append or replace one scenario result in its per-seed JSON log.

    Args:
        history: Flower history with distributed losses and client metrics.
        config: Hydra configuration containing the random seed.
        val1: Privacy setting used by the first client.
        val2: Privacy setting used by the second client.
        subdir: Result subdirectory within ``base_dir``.
        base_dir: Root directory where experiment results are stored.
        metric_name: Parameter prefix used in the log, such as ``"noise"``
            or ``"features"``.

    Returns:
        The path to the updated JSON log.
    """
    # Collect round data into a structured format
    rounds_list = []
    for r in range(len(history.losses_distributed)):
        rounds_list.append({
            "round": r + 1,
            "global_evaluation": {
                "P1": {"accuracy": history.metrics_distributed["client1_accuracy"][r][1],
                        "loss": history.metrics_distributed["client1_loss"][r][1],
                        "epsilon": history.metrics_distributed["client1_epsilon"][r][1]},
                "P2": {"accuracy": history.metrics_distributed["client2_accuracy"][r][1],
                        "loss": history.metrics_distributed["client2_loss"][r][1],
                        "epsilon": history.metrics_distributed["client2_epsilon"][r][1]}
            },
            "loss": history.losses_distributed[r][1]
        })

    # Create the experiment entry with the current scenario's parameters and results
    experiment_entry = {
        f"{metric_name}_p1": val1,
        f"{metric_name}_p2": val2,
        "rounds": rounds_list,
        "final_evaluation": {
            "P1": {"accuracy": history.metrics_distributed["client1_accuracy"][-1][1],
                    "loss": history.metrics_distributed["client1_loss"][-1][1],
                    "epsilon": history.metrics_distributed["client1_epsilon"][-1][1]},
            "P2": {"accuracy": history.metrics_distributed["client2_accuracy"][-1][1],
                    "loss": history.metrics_distributed["client2_loss"][-1][1],
                    "epsilon": history.metrics_distributed["client2_epsilon"][-1][1]}
        }
    }

    # Get the file path for the current experiment's results
    file_path = os.path.join(base_dir, subdir, f"{config.config.seed}.json")

    # Load and update the existing log file with the new experiment entry
    with open(file_path, "r") as f:
        full_log = json.load(f)
    
    parameter_keys = (f"{metric_name}_p1", f"{metric_name}_p2")
    existing_index = next(
        (
            index
            for index, existing in enumerate(full_log["experiments"])
            if (existing.get(parameter_keys[0]), existing.get(parameter_keys[1]))
            == (val1, val2)
        ),
        None,
    )
    if existing_index is None:
        full_log["experiments"].append(experiment_entry)
    else:
        full_log["experiments"][existing_index] = experiment_entry

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(file_path),
            prefix=f".{os.path.basename(file_path)}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(full_log, f, indent=4)
        os.replace(temp_path, file_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return file_path