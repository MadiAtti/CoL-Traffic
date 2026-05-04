# Price of Privacy in Collaborative Drug Discovery

This repository implements the experimental framework detailed in the paper:
**"Approximating the Accuracy Gain in Collaborative Learning with Two Users"**

It utilizes the [Flower (flwr.ai)](https://flower.ai/) federated learning framework to simulate a two-player Collaborative Learning (CoL) game in a drug discovery (QSAR) context. 

## Theoretical Context
The goal of this codebase is to empirically calculate the **Privacy-Accuracy Trade-off Function (PATF)** and the **Price of Privacy (PoP)**. The simulation evaluates how local data suppression (attribute reduction) and Bounded Differential Privacy (bDP) impact the Normalized Accuracy Improvement:

`Normalized Improvement = (theta - Theta) / |o - theta|`

Where:
* `o` = Oracle loss (untrained baseline model)
* `theta` = Local training loss (training alone)
* `Theta` = Federated training loss (training collaboratively)

## Architecture Overview
This project transitions from a serialized, manual training loop to a fully decentralized simulation using Flower:
* **`src/main.py`**: The orchestrator. Sets up the scenarios (Real P1+P2, Sim P11+P12) and triggers `fl.simulation.start_simulation`.
* **`src/client.py`**: Contains `DrugDiscoveryClient`. Manages local Stochastic Gradient Descent (SGD) and local Differential Privacy noise injection. Critically, it handles the statefulness of the task-specific Neural Network "Head" across Flower's stateless Ray actors.
* **`src/strategy.py`**: Implements `SaveModelStrategy`, a custom extension of `FedAvg` designed to extract the final global `trunk` parameters at the end of the specified rounds.

## Quickstart
To run a quick simulation with reduced parameters (3x3 grids, 50 rounds):
```bash
python src/main.py --quick --n_samples 50000 --seed_range 3 --rounds 50
```