import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from data.dataset import prepare_data_and_loaders, setup_directories
from experiment.experiment_runner import run_experiment
from experiment.local_baseline import run_local_experiment
from utils.logger_silencer import silence_log


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):

    ## Load configuration and setup directories
    silence_log()
    OmegaConf.set_struct(config, False)
    setup_directories(config)

    ## Prepare dataset and update config with dataset info (input_dim, num_classes)
    loaders = prepare_data_and_loaders(config)               

    sample_batch_x, sample_batch_y = next(iter(loaders["p1"]["train"]))
    if "dataset" not in config:
        config.dataset = {}
    
    config.dataset.input_dim = sample_batch_x.shape[1]
    config.dataset.num_classes = len(torch.unique(sample_batch_y))

    ## Define experiments (client pairs and their corresponding dataset keys)

    experiments = [
        ("P1-P2",   "p1",  "p2",  ""),      
        ("P11-P12", "p11", "p12", "P1"),   
        ("P21-P22", "p21", "p22", "P2") 
    ]

    ## Run local baseline experiments (M1, M2, M11, M12, M21, M22)

    for exp_name, client1_key, client2_key, subdir in experiments:
        print(f"\n{'#'*80}")
        print(f"🚀 INDUL AZ EXPERIMENT: {exp_name} (BASELINE)🚀")
        print(f"{'#'*80}\n")

        # Listákat készítünk a loaderekből
        train_loaders = [loaders[client1_key]["train"], loaders[client2_key]["train"]]
        test_loaders = [loaders[client1_key]["test"], loaders[client2_key]["test"]]
        
        # Átadjuk a kliens ID-kat is a JSON címkézéshez
        run_local_experiment(
            config=config, 
            train_loaders=train_loaders, 
            test_loaders=test_loaders, 
            subdir=subdir,
            client_ids=[client1_key, client2_key]
        )

    ## Run experiments with different noise levels

    # for exp_name, client1_key, client2_key, subdir in experiments:
    #     print(f"\n{'#'*80}")
    #     print(f"🚀 INDUL AZ EXPERIMENT: {exp_name} (DP)🚀")
    #     print(f"{'#'*80}\n")

    #     train_loaders = [loaders[client1_key]["train"], loaders[client2_key]["train"]]
    #     test_loaders = [loaders[client1_key]["test"], loaders[client2_key]["test"]]

    #     run_experiment(
    #         config=config, 
    #         train_loaders=train_loaders, 
    #         test_loaders=test_loaders, 
    #         subdir=subdir, mode="dp")

    ## Run experiments with different suppression levels
    
    # for exp_name, client1_key, client2_key, subdir in experiments:
    #     print(f"\n{'#'*80}")
    #     print(f"🚀 INDUL AZ EXPERIMENT: {exp_name} (SUP)🚀")
    #     print(f"{'#'*80}\n")

    #     train_loaders = [loaders[client1_key]["train"], loaders[client2_key]["train"]]
    #     test_loaders = [loaders[client1_key]["test"], loaders[client2_key]["test"]]

    #     run_experiment(
    #         config=config, 
    #         train_loaders=train_loaders, 
    #         test_loaders=test_loaders, 
    #         subdir=subdir, mode="sup")

if __name__ == "__main__":
    
    main()