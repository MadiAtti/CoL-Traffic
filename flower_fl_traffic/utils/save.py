import os
import json
from omegaconf import OmegaConf

def setup_file(config, subdir, base_dir):
    target_path = os.path.join(base_dir, subdir)
    os.makedirs(target_path, exist_ok=True)
    file_path = os.path.join(target_path, f"{config.config.seed}.json")
    
    # Létrehozunk egy alap JSON-t, csak a paraméterekkel, üres kísérlet listával
    # Ez felülírja a korábbi (pl. tegnapi) futtatás eredményeit
    initial_log = {
        "parameters": OmegaConf.to_container(config, resolve=True),
        "experiments": []
    }
    with open(file_path, "w") as f:
        json.dump(initial_log, f, indent=4)

def save_federated_history(history, config, val1, val2, subdir, base_dir, metric_name):
    """
    val1, val2: a két kliens aktuális értéke (zajszint VAGY feature szám)
    metric_name: "noise" vagy "features"
    """
    # 1. Körök adatainak kigyűjtése
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

    # 2. Az experiment bejegyzés összeállítása dinamikus kulcsokkal
    experiment_entry = {
        f"{metric_name}_p1": val1,
        f"{metric_name}_p2": val2,
        "rounds": rounds_list,
        "final_evaluation": {
            "P1": {"accuracy": history.metrics_distributed["client1_accuracy"][-1][1]},
            "P2": {"accuracy": history.metrics_distributed["client2_accuracy"][-1][1]}
        }
    }

    # 3. Útvonal meghatározása
    file_path = os.path.join(base_dir, subdir, f"{config.config.seed}.json")

    # 4. Betöltés és HOZZÁADÁS
    with open(file_path, "r") as f:
        full_log = json.load(f)
    
    full_log["experiments"].append(experiment_entry)

    # 5. Mentés
    with open(file_path, "w") as f:
        json.dump(full_log, f, indent=4)
    
    return file_path