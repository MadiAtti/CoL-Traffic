"""
run_max_privacy_experiment.py
------------------------------
Max-privacy kísérlet: minden privacy szint párosítva az echo klienssel.
"""

import logging
import multiprocessing as mp
from itertools import chain
from pathlib import Path

from utils.logger_silencer import silence_log
silence_log()

import ray
import flwr as fl
from omegaconf import OmegaConf

from federated.echo_client import create_echo_client_fn, EchoClient
from federated.server import get_on_fit_config
from federated.universal_client import create_client_fn
from models.neural_network import TrafficNN
from utils.metrics import player_specific_metrics
from utils.save import save_federated_history, setup_file

ECHO = "echo"  # Sentinel érték None helyett


def build_max_privacy_scenarios(levels):
    """
    L-alakú scenario lista, None helyett "echo" sentinel értékkel.
    """
    c1_echo = [(ECHO, v) for v in levels]
    c2_echo = [(v, ECHO) for v in levels]
    both_echo = [(ECHO, ECHO)]
    return list(chain(c1_echo, c2_echo, both_echo))


def create_mixed_client_fn(train_loaders, test_loaders, cfg, val1, val2, param_key):
    vals = [val1, val2]

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        if vals[idx] == ECHO:
            return EchoClient(
                cid=cid,
                model=TrafficNN(
                    input_dim=cfg.dataset.input_dim,
                    num_classes=cfg.dataset.num_classes,
                ),
                testloader=test_loaders[idx],
                cfg=cfg,
            ).to_client()
        else:
            return create_client_fn(train_loaders, test_loaders, cfg)(cid)

    return client_fn


def _run_single_scenario(args):
    silence_log()

    (val1, val2, config, raw_train_loaders, test_loaders,
     subdir, mode, base_dir, metric_name, param_key, lock) = args

    active_loaders = raw_train_loaders

    print(
        f"\n🔇 Max-privacy scenario ({mode.upper()}) "
        f"| C1: {val1} | C2: {val2}",
        flush=True,
    )

    fit_kwargs = {
        f"client1_{param_key}": 0.0 if val1 == ECHO else val1,
        f"client2_{param_key}": 0.0 if val2 == ECHO else val2,
    }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        on_fit_config_fn=get_on_fit_config(**fit_kwargs),
        evaluate_metrics_aggregation_fn=player_specific_metrics,
    )

    history = fl.simulation.start_simulation(
        client_fn=create_mixed_client_fn(
            active_loaders, test_loaders, config, val1, val2, param_key
        ),
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.config.federated_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={
            "logging_level": logging.ERROR,
            "log_to_driver": True,
            "num_cpus": 2,
            "runtime_env": {"env_vars": {"OMP_NUM_THREADS": "1"}},
        },
    )

    with lock:
        save_federated_history(
            history, config, val1, val2, subdir,
            base_dir=base_dir, metric_name=metric_name,
        )

    res_1 = history.metrics_distributed["client1_accuracy"][-1][1]
    res_2 = history.metrics_distributed["client2_accuracy"][-1][1]
    print(
        f"✅ Kész | C1: {val1}, C2: {val2} "
        f"| Acc: {res_1:.2%}, {res_2:.2%}",
        flush=True,
    )
    return (val1, val2, res_1, res_2)


def run_max_privacy_experiment(config, train_loaders, test_loaders, subdir, mode):
    ds_mode = config.dataset.mode

    if mode == "dp":
        base_dir = f"results/{ds_mode}/4_max_privacy_noise"
        metric_name = "noise"
        param_key = "noise"
        if ds_mode == "full":
            levels = config.config.full_noise_levels
        elif ds_mode == "half":
            levels = config.config.half_noise_levels
        else:
            raise ValueError(f"Ismeretlen dataset mode: {ds_mode}")
    else:
        base_dir = f"results/{ds_mode}/4_max_privacy_suppression"
        metric_name = "features"
        param_key = "features"
        levels = config.config.sup_levels

    setup_file(config, subdir, base_dir=base_dir)

    scenarios = build_max_privacy_scenarios(levels)

    print(f"\n{'#'*60}")
    print(f"🔇 Max-privacy runner | Mode: {mode.upper()} | {len(scenarios)} scenario")
    print(f"   Szintek : {list(levels)}")
    print(f"   Összes  : 2×{len(levels)}+1 = {len(scenarios)}")
    print(f"{'#'*60}\n")

    manager = mp.Manager()
    lock = manager.Lock()

    tasks = [
        (val1, val2, config, train_loaders, test_loaders,
         subdir, mode, base_dir, metric_name, param_key, lock)
        for val1, val2 in scenarios
    ]

    num_parallel_scenarios = 4

    # FIX: ray.shutdown() kivéve a finally-ből, csak a pool után hívjuk
    try:
        with mp.Pool(processes=num_parallel_scenarios) as pool:
            pool.map(_run_single_scenario, tasks)
    except Exception as e:
        print(f"Hiba a párhuzamos futtatás során: {e}")

    ray.shutdown()  # Csak egyszer, a pool teljes befejezése után

    print(f"\n✨ Minden max-privacy scenario kész ({subdir}, {mode.upper()}).")