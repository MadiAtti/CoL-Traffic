import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from data.dataset import prepare_data_and_loaders, setup_directories
from experiment.differential_privacy import run_dp_experiment
from utils.logger_silencer import silence_log


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    ## 1. load configuration and setup directories
    #print("Betöltött konfiguráció:")
    #print(OmegaConf.to_yaml(config))
    silence_log()
    OmegaConf.set_struct(config, False)

    setup_directories(config)


    ## 2. prepare dataset
    loaders = prepare_data_and_loaders(config)               

    sample_batch_x, sample_batch_y = next(iter(loaders["p1"]["train"]))
    if "dataset" not in config:
        config.dataset = {}
    
    config.dataset.input_dim = sample_batch_x.shape[1]
    config.dataset.num_classes = len(torch.unique(sample_batch_y))

    experiments = [
        ("P1-P2",   "p1",  "p2",  ""),      # Gyökér (4_noise/)
        ("P11-P12", "p11", "p12", "P1"),    # 4_noise/P1/
        ("P21-P22", "p21", "p22", "P2")     # 4_noise/P2/
    ]

    ## 3. Run experiments with different noise levels

    for exp_name, client1_key, client2_key, subdir in experiments:
        print(f"\n{'#'*80}")
        print(f"🚀 INDUL AZ EXPERIMENT: {exp_name} (DP)🚀")
        print(f"{'#'*80}\n")

        train_loaders = [loaders[client1_key]["train"], loaders[client2_key]["train"]]
        test_loaders = [loaders[client1_key]["test"], loaders[client2_key]["test"]]

        run_dp_experiment(config, train_loaders, test_loaders, subdir=subdir)
    

if __name__ == "__main__":
    
    main()