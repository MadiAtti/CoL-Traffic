# Drug Discovery Experiments

## Setup

From the `drug_discovery` directory, create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r ../requirements.txt
```

## Run `main.py`

Run this command from the `drug_discovery` directory:

```bash
python src/main.py \
    --data_root src/Datasets/data_30 \
    --seed_range 10 \
    --workers 70 \
    --out_csv_dir src/Results/data_30/CSV \
    --out_plot_dir src/Results/data_30/Plots
```

**Important:** Check how many CPU workers are available before running the experiment. Change `--workers 70` to a value allowed by your computer or SLURM allocation; do not assume that 70 workers are available.

To use the larger dataset, replace `data_30` with `data_80` in `--data_root`.