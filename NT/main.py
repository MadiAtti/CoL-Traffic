"""Run the configured multi-seed federated traffic-classification study.

The pipeline creates participant data loaders, trains local baselines, runs
regular and maximum-privacy Flower simulations, and builds game matrices.
Visualization and Nash-equilibrium analysis are intentionally separate tools.
"""

import multiprocessing as mp
import os
from time import perf_counter

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.logger_silencer import silence_log

silence_log()

from data.dataset import prepare_data_and_loaders, setup_directories
from data.splitter import dataset_splitter
from experiment.experiment_runner import run_experiment
from experiment.local_baseline import run_local_experiment
from experiment.run_max_privacy_experiment import run_max_privacy_experiment
from utils.average_plots import build_matrices_for_mode
from utils.seed import set_seed


def run_pipeline(config: DictConfig) -> None:
    """Run all baseline and federated scenarios for one configured seed.

    The seed's cached participant splits are prepared before training. Each
    experiment uses the full participant pair and both self-division pairs.

    Args:
        config: Resolved Hydra configuration for the current seed.
    """
    silence_log()
    start_time = perf_counter()
    if config.num_clients != 2:
        raise ValueError("This experiment pipeline requires exactly two clients.")
    set_seed(config.config.seed)

    OmegaConf.set_struct(config, False)
    setup_directories(config)
    dataset_splitter(config)

    loaders = prepare_data_and_loaders(config)
    p1_train = loaders["p1"]["train"].dataset
    p2_train = loaders["p2"]["train"].dataset
    p1_test = loaders["p1"]["test"].dataset
    p2_test = loaders["p2"]["test"].dataset
    if not len(p1_train) or not len(p2_train):
        raise ValueError("The configured training split is empty.")

    config.dataset.input_dim = len(config.dataset.feature_columns)
    labels = set(p1_train.y.tolist())
    labels.update(p2_train.y.tolist())
    labels.update(p1_test.y.tolist())
    labels.update(p2_test.y.tolist())
    config.dataset.num_classes = len(labels)

    experiments = [
        ("P1-P2", "p1", "p2", ""),
        ("P11-P12", "p11", "p12", "P1"),
        ("P21-P22", "p21", "p22", "P2"),
    ]

    for experiment_name, client1_key, client2_key, subdir in experiments:
        print(f"\n{'#' * 80}\nStarting {experiment_name}\n{'#' * 80}")
        train_loaders = [
            loaders[client1_key]["train"],
            loaders[client2_key]["train"],
        ]
        test_loaders = [
            loaders[client1_key]["test"],
            loaders[client2_key]["test"],
        ]

        run_local_experiment(
            config=config,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            subdir=subdir,
            client_ids=[client1_key, client2_key],
        )
        run_experiment(config, train_loaders, test_loaders, subdir, mode="dp")
        run_experiment(config, train_loaders, test_loaders, subdir, mode="sup")
        run_max_privacy_experiment(
            config, train_loaders, test_loaders, subdir, mode="dp"
        )
        run_max_privacy_experiment(
            config, train_loaders, test_loaders, subdir, mode="sup"
        )

    elapsed = perf_counter() - start_time
    print(f"Total execution time: {elapsed / 3600:.2f} hours")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    """Execute the pipeline for the inclusive configured seed range.

    After all seeds finish, construct per-seed and averaged payoff matrices
    for the selected dataset mode.

    Args:
        config: Hydra configuration loaded from ``conf/base.yaml``.
    """
    os.environ["RAY_GCS_SERVER_PORT"] = "0"
    seed_start = int(config.config.seed_start)
    seed_end = int(config.config.seed_end)
    if seed_end < seed_start:
        raise ValueError("config.seed_end must be greater than or equal to config.seed_start.")

    for seed in range(seed_start, seed_end + 1):
        config.config.seed = seed
        print(f"\n{'#' * 80}\nRunning seed {seed}\n{'#' * 80}")
        run_pipeline(config)

    build_matrices_for_mode(
        config.dataset.mode,
        seeds=range(seed_start, seed_end + 1),
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
