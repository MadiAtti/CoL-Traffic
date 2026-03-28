import os
import json
from omegaconf import OmegaConf

def setup_file(config, subdir):
    base_dir = "4_noise"
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

def save_federated_history(history, config, c1_noise, c2_noise, subdir):
    """
    A Flower history objektumát a 0.json formátumra alakítja és menti.
    """
    # 1. Adatok kigyűjtése (ugyanaz mint eddig)
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

    experiment_entry = {
        "noise_multiplier_p1": c1_noise,
        "noise_multiplier_p2": c2_noise,
        "rounds": rounds_list,
        "final_evaluation": {
            "P1": {"accuracy": history.metrics_distributed["client1_accuracy"][-1][1]},
            "P2": {"accuracy": history.metrics_distributed["client2_accuracy"][-1][1]}
        }
    }

    # 2. Útvonal (mint eddig)
    file_path = os.path.join("4_noise", subdir, f"{config.config.seed}.json")

    # 3. Betöltés és HOZZÁADÁS
    # Itt már feltételezzük, hogy a fájl létezik (a runner létrehozta)
    with open(file_path, "r") as f:
        full_log = json.load(f)
    
    # Egyszerűen hozzáfűzzük (nem vizsgáljuk, volt-e már ilyen zaj)
    full_log["experiments"].append(experiment_entry)

    # 4. Mentés
    with open(file_path, "w") as f:
        json.dump(full_log, f, indent=4)
    
    return file_path