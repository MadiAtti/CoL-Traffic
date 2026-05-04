# CoL-Traffic: Federated Learning for Network Traffic Classification

This project implements a Federated Learning framework using **Flower** to investigate model robustness and performance under various constraints, such as feature suppression and differential privacy (noise injection), specifically for network traffic classification tasks.

## Project Structure

```text
flower_fl_traffic/
│
├── main.py                
├── conf/
│   └── base.yaml           
├── models/
│   └── neural_network.py  
├── data/
│   ├── dataset.py         
│   └── custom_dataset.py 
├── experiment/
│   ├── experiment_runner.py 
│   └── local_baseline.py   
├── federated/
│   ├── universal_client.py  
│   └── server.py            
├── utils/
│   ├── evaluation.py       
│   ├── logger_silencer.py            
│   ├── training.py        
│   ├── save.py             
│   ├── seed.py             
│   └── metrics.py          
└── dataset/                
```

---

## Dataset

The `dataset/` directory acts as a placeholder and does not contain the actual data files due to size constraints. To execute the experiments, you must provide the dataset manually by following these steps:

1.  **Download**: Obtain the `dataset.parquet` file from the project's official data source.
2.  **Placement**: Move the downloaded `.parquet` file directly into the `dataset/` folder.
3.  **Source Link**: [INSERT_LINK_HERE]

The system expects the primary file at: ./dataset/dataset.parquet.

---

## Installation & Environment Setup

This project uses **Conda** to manage dependencies and **Hydra** for configuration. Follow these steps to set up your local environment:

### Create the Environment
Use the provided `enviroment.yml` to recreate the exact Python environment. This ensures all versions (Flower, PyTorch, Opacus) are compatible.
```bash
conda env create -f enviroment.yml
conda activate flower-fl-traffic
```

---

## Running Experiments

Once the environment and dataset are ready, you can start the experiments using the main entry point:

```bash
python main.py
```

### Configuring the Seed

To run the code with a different seed value for reproducibility:

- Option A: Change the seed value directly in the conf/base.yaml file.

- Option B (Recommended): Override it directly from the terminal without touching the config file:

```bash
python main.py config.seed=5
```

By default, the system will run:

1. **Local Baselines**:

    - Process: Training on all datasets individually using cross-validation, without any Federated Learning (FL).

    - Goal: Establish a performance benchmark to see the maximum accuracy achievable in a traditional, non-distributed setup.

2. **Feature Suppression**: 
    - Process: An FL simulation where features are systematically removed from the system in steps (14, 12, 10... down to 2).

    - Mechanism: The system tests every possible pairing of these feature counts across two clients, resulting in a 7x7 results matrix.

    - Goal: To observe how the global model performs when clients have different or limited sets of available features.

3. **Noise Injection**: 
    - Process: Similar to the suppression logic, but instead of removing features, the system injects varying levels of Gaussian noise into the training process.

    - Mechanism: Seven different noise levels are tested in all possible combinations between clients, producing another 7x7 matrix.

    - Goal: To measure the impact of privacy-preserving noise on the final classification accuracy.

Note: You *can* modify hyperparameters and experiment settings in conf/base.yaml.

---

## Results & Output

After execution, the results are stored in the following directory structure:

```text
flower_fl_traffic/
├── 1_local_baseline/   
├── 2_suppression/    
└── 3_noise/           
```

### Key Logging Features:

- **Deterministic Naming**: The output filename is directly tied to the initial seed value used for the run (e.g., starting with `seed: 0` generates `0.json`). This ensures that results from different stochastic initializations do not overwrite each other.

- **Data Consistency**: Each JSON file contains comprehensive metrics, including:

    - Training/Validation loss per round.

    - Final classification accuracy.

    - Specific parameters for the 7x7 matrix (feature counts or noise levels).
---