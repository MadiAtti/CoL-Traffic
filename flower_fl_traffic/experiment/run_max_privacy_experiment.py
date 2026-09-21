"""
run_max_privacy_experiment.py
------------------------------
Max-privacy kísérlet: minden privacy szint párosítva az echo klienssel.

Scenario lista (N=5 szint esetén, 2N+1 = 11 db):
  (echo, p1), (echo, p2), ..., (echo, pN)   <- C1 csendes, C2 normál
  (p1, echo), (p2, echo), ..., (pN, echo)   <- C1 normál,  C2 csendes
  (echo, echo)                               <- mindkettő csendes (baseline)

Azonos struktúra mint a meglévő run_experiment.py, csak a scenario
generálás tér el: product() helyett az "él" kombinációk.

Output: results/<ds_mode>/4_max_privacy/max_privacy_results.csv
"""

import logging
import multiprocessing as mp
from itertools import chain
from pathlib import Path

from utils.logger_silencer import silence_log
silence_log()

import flwr as fl
from omegaconf import OmegaConf

from federated.echo_client import create_echo_client_fn, EchoClient
from federated.server import get_on_fit_config
from federated.universal_client import create_client_fn
from models.neural_network import TrafficNN
from utils.metrics import player_specific_metrics
from utils.save import save_federated_history, setup_file


# ──────────────────────────────────────────────────────────────────────────────
# Scenario generálás
# ──────────────────────────────────────────────────────────────────────────────

def build_max_privacy_scenarios(levels):
    """
    Visszaadja az L-alakú scenario listát.

    Minden elem: (val1, val2) ahol a None az echo klienst jelenti.

    Pl. levels=[0.0, 0.5, 1.0, 1.5, 2.0] esetén:
      (None, 0.0), (None, 0.5), ..., (None, 2.0)   <- C1 echo
      (0.0, None), (0.5, None), ..., (2.0, None)   <- C2 echo
      (None, None)                                   <- mindkettő echo
    """
    c1_echo = [(None, v) for v in levels]   # C1 csendes, C2 normál
    c2_echo = [(v, None) for v in levels]   # C1 normál,  C2 csendes
    both_echo = [(None, None)]              # mindkettő csendes
    return list(chain(c1_echo, c2_echo, both_echo))


# ──────────────────────────────────────────────────────────────────────────────
# Mixed client_fn: None → EchoClient, érték → UniversalTrafficClient
# ──────────────────────────────────────────────────────────────────────────────

def create_mixed_client_fn(train_loaders, test_loaders, cfg, val1, val2, param_key):
    """
    val1, val2: a privacy paraméter értéke (pl. noise vagy features),
                vagy None ha az adott kliens echo.
    param_key:  "client1_noise" / "client2_noise"  (dp módban)
                "client1_features" / "client2_features"  (sup módban)
    """
    vals = [val1, val2]
    param_keys = [f"client1_{param_key}", f"client2_{param_key}"]

    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        if vals[idx] is None:
            # Echo kliens: visszaadja a szerver modelljét változatlanul
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
            # Normál kliens: helyi tanítás a megadott privacy szinttel
            return create_client_fn(train_loaders, test_loaders, cfg)(cid)

    return client_fn


# ──────────────────────────────────────────────────────────────────────────────
# Egyetlen scenario futtatása (külön folyamatban)
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_scenario(args):
    """
    A meglévő _run_single_scenario-val azonos szignatúra és struktúra,
    de val1/val2 lehet None (echo kliens).
    """
    silence_log()

    (val1, val2, config, raw_train_loaders, test_loaders,
     subdir, mode, base_dir, metric_name, param_key, lock) = args

    # Echo kliensnél nem kell adat-előkészítés (sup módban sincs suppression)
    active_loaders = raw_train_loaders

    # Szép label a loghoz
    def _label(v):
        return "echo" if v is None else str(v)

    print(
        f"\n🔇 Max-privacy scenario ({mode.upper()}) "
        f"| C1: {_label(val1)} | C2: {_label(val2)}",
        flush=True,
    )

    # on_fit_config: echo klienseknél 0.0 noise-t adunk, így a szerver
    # konfigja konzisztens marad; az EchoClient.fit() úgyis figyelmen
    # kívül hagyja a tanítást
    fit_kwargs = {
        f"client1_{param_key}": val1 if val1 is not None else 0.0,
        f"client2_{param_key}": val2 if val2 is not None else 0.0,
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
        f"✅ Kész | C1: {_label(val1)}, C2: {_label(val2)} "
        f"| Acc: {res_1:.2%}, {res_2:.2%}",
        flush=True,
    )
    return (val1, val2, res_1, res_2)


# ──────────────────────────────────────────────────────────────────────────────
# Fő belépési pont
# ──────────────────────────────────────────────────────────────────────────────

def run_max_privacy_experiment(config, train_loaders, test_loaders, subdir, mode):
    """
    Leváltja / kiegészíti a run_experiment()-et a max-privacy scenáriókhoz.

    Hívás ugyanolyan mint run_experiment():
        run_max_privacy_experiment(config, train_loaders, test_loaders, subdir, mode)

    mode: "dp"  → noise szintek, param_key = "noise"
          "sup" → feature szintek, param_key = "features"
    """
    import ray

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

    # L-alakú scenario lista: 2N+1 db
    scenarios = build_max_privacy_scenarios(levels)

    print(f"\n{'#'*60}")
    print(f"🔇 Max-privacy runner | Mode: {mode.upper()} | {len(scenarios)} scenario")
    print(f"   Szintek : {list(levels)}")
    print(f"   Összes  : 2×{len(levels)}+1 = {len(scenarios)} (vs {len(levels)**2} normál gridnél)")
    print(f"{'#'*60}\n")

    manager = mp.Manager()
    lock = manager.Lock()

    tasks = [
        (val1, val2, config, train_loaders, test_loaders,
         subdir, mode, base_dir, metric_name, param_key, lock)
        for val1, val2 in scenarios
    ]

    num_parallel_scenarios = 4

    try:
        with mp.Pool(processes=num_parallel_scenarios) as pool:
            pool.map(_run_single_scenario, tasks)
    except Exception as e:
        print(f"Hiba a párhuzamos futtatás során: {e}")
    finally:
        ray.shutdown()

    print(f"\n✨ Minden max-privacy scenario kész ({subdir}, {mode.upper()}).")