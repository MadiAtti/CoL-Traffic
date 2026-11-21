"""Run scenarios that pair configured privacy levels with an echo client.

The echo boundary models full protection by returning the received global
weights unchanged. Regular privacy levels are crossed with echo in both player
positions, with an additional echo-versus-echo scenario.
"""

import logging
import multiprocessing as mp
from itertools import chain

from utils.logger_silencer import silence_log
silence_log()

import ray
import flwr as fl

from federated.echo_client import EchoClient
from federated.server import get_on_fit_config
from federated.universal_client import create_client_fn
from models.neural_network import TrafficNN
from utils.metrics import player_specific_metrics
from utils.save import save_federated_history, setup_file

ECHO = "echo"


def build_max_privacy_scenarios(levels):
    """Create the L-shaped grid of regular and maximum-privacy scenarios.

    Args:
        levels: Configured suppression counts or DP noise multipliers.

    Returns:
        Ordered pairs containing echo-versus-level, level-versus-echo, and
        echo-versus-echo settings.
    """
    c1_echo = [(ECHO, v) for v in levels]
    c2_echo = [(v, ECHO) for v in levels]
    both_echo = [(ECHO, ECHO)]
    return list(chain(c1_echo, c2_echo, both_echo))


def create_mixed_client_fn(train_loaders, test_loaders, cfg, val1, val2):
    """Build a Flower client factory for one mixed echo/active scenario.

    The echo client's reported example count matches the active client's
    training-set size. This gives both clients equal weight in FedAvg for
    equal-sized participant splits.

    Args:
        train_loaders: Training loaders for both clients.
        test_loaders: Evaluation loaders for both clients.
        cfg: Experiment configuration used to create each client model.
        val1: First client's privacy level or ``"echo"``.
        val2: Second client's privacy level or ``"echo"``.

    Returns:
        A Flower ``client_fn`` accepting a client ID.
    """
    vals = [val1, val2]

    def client_fn(cid: str) -> fl.client.Client:
        """Create the active or echo client associated with ``cid``."""
        idx = int(cid)
        other_idx = 1 - idx  # Index of the other client.

        if vals[idx] == ECHO:
            return EchoClient(
                cid        = cid,
                model      = TrafficNN(
                    input_dim   = cfg.dataset.input_dim,
                    num_classes = cfg.dataset.num_classes,
                ),
                testloader = test_loaders[idx],
                cfg        = cfg,
                n_samples  = len(train_loaders[other_idx].dataset),
            ).to_client()
        else:
            return create_client_fn(train_loaders, test_loaders, cfg)(cid)

    return client_fn


def _run_single_scenario(args):
    """Run one maximum-privacy scenario and save its federated history.

    Args:
        args: Worker tuple containing privacy parameters, client loaders,
            configuration, output metadata, and a shared file lock.

    Returns:
        The scenario parameters and final per-client accuracies.
    """
    silence_log()

    (val1, val2, config, raw_train_loaders, test_loaders,
     subdir, mode, base_dir, metric_name, param_key, lock) = args

    # Apply feature suppression only to active clients, never to echo clients.
    if mode == "sup":
        from data.custom_dataset import CustomDataset
        from torch.utils.data import DataLoader

        total_f = config.dataset.input_dim
        active_loaders = []

        for i, val in enumerate([val1, val2]):
            if val == ECHO:
                active_loaders.append(raw_train_loaders[i])
            else:
                limit = int(val)
                feature_indices = list(range(limit))

                X_orig = raw_train_loaders[i].dataset.X.cpu().numpy().copy()
                y_orig = raw_train_loaders[i].dataset.y.cpu().numpy().copy()
                X_cut  = X_orig[:, feature_indices]

                new_ds = CustomDataset(
                    X=X_cut,
                    y=y_orig,
                    feature_indices=feature_indices,
                    total_features=total_f,
                )
                active_loaders.append(DataLoader(
                    new_ds,
                    batch_size=config.config.batch_size,
                    shuffle=True,
                ))
    else:
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
            active_loaders, test_loaders, config, val1, val2
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
    """Run all echo-boundary scenarios for one method and client split.

    Args:
        config: Hydra experiment configuration.
        train_loaders: Two client training DataLoaders.
        test_loaders: Two client evaluation DataLoaders.
        subdir: Empty for the full participant pair or ``P1``/``P2`` for
            self-division games.
        mode: Either ``"dp"`` or ``"sup"``.

    Raises:
        ValueError: If the dataset or experiment mode is unsupported.
    """
    ds_mode = config.dataset.mode
    if ds_mode not in {"full", "half", "80percent", "30percent"}:
        raise ValueError(f"Ismeretlen dataset mode: {ds_mode}")

    if mode == "dp":
        base_dir  = f"results/{ds_mode}/4_max_privacy_noise"
        metric_name = "noise"
        param_key   = "noise"
        if ds_mode == "full":
            levels = config.config.full_noise_levels
        elif ds_mode == "half":
            levels = config.config.half_noise_levels
        elif ds_mode == "80percent":
            levels = config.config.percent80_noise_levels
        elif ds_mode == "30percent":
            levels = config.config.percent30_noise_levels
        else:
            raise ValueError(f"Ismeretlen dataset mode: {ds_mode}")
    elif mode == "sup":
        base_dir    = f"results/{ds_mode}/4_max_privacy_suppression"
        metric_name = "features"
        param_key   = "features"
        levels      = config.config.sup_levels
    else:
        raise ValueError(f"Unsupported experiment mode: {mode}")

    setup_file(config, subdir, base_dir=base_dir)

    scenarios = build_max_privacy_scenarios(levels)

    print(f"\n{'#'*60}")
    print(f"🔇 Max-privacy runner | Mode: {mode.upper()} | {len(scenarios)} scenario")
    print(f"   Szintek : {list(levels)}")
    print(f"   Összes  : 2×{len(levels)}+1 = {len(scenarios)}")
    print(f"{'#'*60}\n")

    manager = mp.Manager()
    lock    = manager.Lock()

    tasks = [
        (val1, val2, config, train_loaders, test_loaders,
         subdir, mode, base_dir, metric_name, param_key, lock)
        for val1, val2 in scenarios
    ]

    num_parallel_scenarios = 4

    try:
        with mp.Pool(processes=num_parallel_scenarios) as pool:
            pool.map(_run_single_scenario, tasks)
    finally:
        ray.shutdown()

    print(f"\n✨ Minden max-privacy scenario kész ({subdir}, {mode.upper()}).")