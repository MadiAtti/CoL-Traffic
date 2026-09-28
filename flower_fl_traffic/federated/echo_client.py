"""
echo_client.py
--------------
Max-privacy ("echo") Flower client, illeszkedve a meglévő
UniversalTrafficClient architektúrához.

A kliens megkapja a szerver globális modelljét minden körben,
visszaadja változatlanul — helyi tanítás nem történik, semmilyen
gradiens vagy adat-információ nem szivárog ki.

evaluate() továbbra is fut, hogy a szerver mérni tudja a globális
modell teljesítményét ezen a kliensen.
"""

from __future__ import annotations

from collections import OrderedDict

import flwr as fl
import torch

from models.neural_network import TrafficNN
from utils.evaluation import evaluate_model


class EchoClient(fl.client.NumPyClient):
    """
    Passzív / max-privacy kliens.

    fit()      → visszaadja a szerver súlyait változatlanul, n_samples-lel.
    evaluate() → normál kiértékelés a helyi tesztadaton.
    """

    def __init__(self, cid, model, testloader, cfg, n_samples):
        super().__init__()
        self.cid        = cid
        self.model      = model
        self.testloader = testloader
        self.cfg        = cfg
        self.n_samples  = n_samples  # tanuló kliens train dataset mérete a 50-50 súlyozáshoz
        self.device     = torch.device("cpu")
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Súlykezelés — azonos a UniversalTrafficClient-tel
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # fit: echo — betölti a szerver súlyait, azonnal visszaadja
    # A szerver azt hiszi tanított, de valójában nem történt semmi
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        return self.get_parameters(config={}), self.n_samples, {}

    # ------------------------------------------------------------------
    # evaluate: normál kiértékelés — azonos a UniversalTrafficClient-tel
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = torch.nn.CrossEntropyLoss()

        loss, accuracy = evaluate_model(self.model, self.testloader, criterion)

        return (
            float(loss),
            len(self.testloader.dataset),
            {"accuracy": float(accuracy), "loss": float(loss)},
        )


# ------------------------------------------------------------------
# Factory — azonos szignatúra mint a create_client_fn-ben
# ------------------------------------------------------------------

def create_echo_client_fn(trainloader, testloader, cfg):
    """
    Visszaad egy client_fn-t, amit közvetlenül át lehet adni a
    fl.simulation.start_simulation()-nek.
    """
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return EchoClient(
            cid        = cid,
            model      = TrafficNN(
                input_dim   = cfg.dataset.input_dim,
                num_classes = cfg.dataset.num_classes,
            ),
            testloader = testloader[idx],
            cfg        = cfg,
            n_samples  = len(trainloader[idx].dataset),
        ).to_client()

    return client_fn