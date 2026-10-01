# Recommender System Experiments

## Setup

From the `recommender_system` directory, create and activate a virtual environment:

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

Install the requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r ../requirements.txt
```

## Run `main.py`

Run this command from the `recommender_system` directory:

```bash
python src/main.py \
    --data_root data/NF_1_IID \
    --out_csv_dir Results/NF_1_IID/CSV \
    --out_plot_dir Results/NF_1_IID/Plots \
    --seed_range 10 \
    --rounds 20 \
    --workers 70
```

Replace the `--data_root` and output paths to use the other dataset.

**Important:** Check how many CPU workers are available before running the experiment. Change `--workers 70` to a value allowed by your computer or SLURM allocation; do not assume that 70 workers are available.