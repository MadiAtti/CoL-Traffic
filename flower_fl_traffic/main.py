import hydra
from omegaconf import DictConfig, OmegaConf
from data.dataset import prepare_data_and_loaders, setup_directories

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig):
    # 1. Konfiguráció betöltése és mappastruktúra létrehozása
    print("Betöltött konfiguráció:")
    print(OmegaConf.to_yaml(config))
    setup_directories(config)

    ## 2. prepare dataset
    loaders = prepare_data_and_loaders(config)

    loaders_keys = list(loaders.keys())
    print(f"Elérhető loaderek: {loaders_keys}")

if __name__ == "__main__":
    
    main()