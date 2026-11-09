# CoL-Traffic

CoL-Traffic is a Flower-based federated-learning research codebase for
multiclass network-traffic classification. It measures the accuracy benefit
of collaborative training under feature suppression and differentially
private (DP) training, and can construct the payoff matrices used in the
paper's Collaborative Learning (CoL) game and Self-Division (SD) analysis.

The main experiment runner is `main.py`. Plotting, Nash analysis, auxiliary
experiments, and dataset-comparison tools are separate scripts; they are not
silently run as part of the main experiment.

## Project layout

```text
flower_fl_traffic/
├── conf/base.yaml                 Hydra experiment configuration
├── data/                          Dataset splitting and PyTorch datasets
├── experiment/                    Local, federated, and maximum-privacy runs
├── federated/                     Flower clients and server configuration
├── models/                        Feed-forward traffic classifiers
├── utils/                          Matrix building, analysis, plots, and helpers
├── dataset/                        Input data and cached participant splits
├── docs/                           Sphinx documentation sources
├── appendix_pipeline.py           Supplementary paper experiments
└── main.py                         Main multi-seed experiment entry point
```

## Requirements and installation

Use the supplied Conda environment from this directory:

```bash
conda env create -f enviroment.yml
conda activate flower-fl-traffic
```

Download [`dataset.parquet`](https://bmeedu-my.sharepoint.com/personal/madarasz_attila_edu_bme_hu/Documents/dataset.parquet)
from the BME OneDrive account and place it at `dataset/dataset.parquet`. Access
requires an authorized institutional account. The file must contain the
feature columns and target named in `conf/base.yaml`. The splitter drops
rows containing missing values, standardizes the configured features, encodes
the application labels, and writes cached participant splits under
`dataset/<mode>/`.

Supported `dataset.mode` values are `full`, `half`, `80percent`, and
`30percent`. The default configuration selects `30percent`; use `full` and
`half` for the two dataset sizes described in the network-traffic section of
the paper. Cached P1/P2 Parquet files are reused if both are present. Remove or
replace those specific cached files yourself when intentionally rebuilding a
split; do not mix cached splits from different source data or settings.

## Run the main experiments

Run commands from this directory so configured relative paths resolve:

```bash
python main.py
```

By default, this runs seeds 0 through 9 inclusive for the selected dataset
mode. For a single seed:

```bash
python main.py config.seed_start=5 config.seed_end=5
```

For another supported dataset mode:

```bash
python main.py dataset.mode=full
```

Each seed runs the local baselines and the configured suppression, DP, and
maximum-privacy scenario grids for the full-participant game and the two
self-division games. After all requested seeds complete, the runner builds
per-seed and seed-averaged payoff matrices. The grid sizes and training
parameters are controlled by `conf/base.yaml`.

**The default run is computationally intensive.** It runs many independent
Flower simulations for every seed and should not be treated as a quick smoke
test. The runner does not automatically invoke plotting or Nash-equilibrium
analysis.

## Metrics and privacy parameterization

For accuracy, the normalized collaborative gain for one participant is

```text
Delta = (accuracy_federated - accuracy_local) / (1 - accuracy_local)
```

This is the fraction of the local model's remaining accuracy error removed by
collaboration. The matrix builder also contains an optional loss-analysis path;
the paper's traffic-classification analysis uses accuracy.

For DP, the configured value is the Gaussian noise multiplier `sigma`. The
game-analysis scripts map it to the unit interval using

```text
p = sigma / (sigma + 1)
```

For feature suppression, the configured feature count determines which
features are retained. The analysis maps a retained-feature count `k` to
`p = (14 - k) / 14`. Maximum privacy is represented separately by an `echo`
client: it returns the server model unchanged during fitting. The accountant's
cumulative epsilon is logged at `config.dp_delta`.

The Nash utility and Self-Division transformation are implemented in separate
post-processing scripts. Their formulas and input/output conventions are
described in the generated API documentation; review those scripts and their
configuration before interpreting results as a reproduction of a specific
paper table.

## Results

Experiment logs are stored under `results/<mode>/`:

```text
results/<mode>/
├── 1_local_baseline/                 Local models, one JSON file per seed
├── 2_suppression/                    Regular suppression grid
├── 3_noise/                          Regular DP/noise grid
├── 4_max_privacy_suppression/        Suppression scenarios with echo
├── 4_max_privacy_noise/              DP scenarios with echo
└── accuracy/games/
    ├── seed<seed>/                   Per-seed bimatrices
    └── average/                      Averaged matrices for analysis and plots
```

Federated JSON logs contain each scenario's privacy parameters, per-round
client metrics, final client metrics, and cumulative DP epsilon. Existing
scenario results are updated when the same seed and scenario are rerun.

## Appendix experiments

The supplementary experiments are separate from `main.py` so their larger
training runs do not affect the paper's main FL results.

### Centralized accuracy by dataset size

This experiment trains a fresh centralized model at each 5% increment of a
seed-shuffled dataset, evaluates it on a stratified holdout set, and averages
the per-seed curves:

```bash
python appendix_pipeline.py saturation --mode run --seeds 0 1 2 --workers 2
python appendix_pipeline.py saturation --mode plot --seeds 0 1 2
```

The per-seed CSVs, averaged CSV, and figure are written to `results/baseline/`.

### Unequal participant data sizes

This experiment compares local and federated accuracy for P1/P2 data ratios
from `1/8` through `8/1`, on 1x, 5x, and 10x dataset sizes. Each client uses a
stratified 80/20 train/holdout split for both the local baseline and federated
evaluation, so the accuracy comparison uses held-out data in both cases.

```bash
python appendix_pipeline.py imbalance --mode run --seeds 10 11 12 --workers 2
python appendix_pipeline.py imbalance --mode plot --seeds 10 11 12
```

This is computationally intensive: each seed includes many Flower simulations.
Per-seed results, summary CSVs, and the appendix figure are saved under
`results/multi-split/`.

## Separate analysis scripts

Run these from this directory, after producing the required experiment
results:

```bash
python utils/visualizer.py       # Heatmaps and fitted SD-to-real transform
python utils/evaluate_game.py    # Nash best-response analysis
python utils/generate_nash_tables.py
                                # Nash tables for measured and transformed matrices
```

The visualizer expects the configured full/half (or 80percent/30percent)
averaged matrices. Nash analysis reads averaged bimatrices for all modes
listed in its script. These commands do not run automatically from `main.py`.
The `DIFF_NT_*` figures show the post-correction residual
`Delta - f(Delta_tilde)`: the empirical matrix minus the fitted, corrected
simulation. The correction is fitted before those residual maps are generated.
The Nash search evaluates the measured NT games first, then the fitted,
transformed Self-Division games, for the four ordered coefficient pairs
`(C1, C2) = (0, 0)`, `(0, 0.1)`, `(0.1, 0)`, and `(0.1, 0.1)`.
All equilibrium data is stored in the single file
`results/nash/nash_equilibria.json`. The original `utils/nesh_visualisation.py`
seven-column simulation-versus-real table renderer is used by
`utils/generate_nash_tables.py`. The script saves the full/half tables in
`results/final_100-50/` and the 80%/30% tables in `results/final_80-30/`;
each mode has both a measured and an `f`-transformed version. The transformed
tables apply the fitted correction to all three comparison matrices.
Each pair folder also contains its pair-specific Nash JSON and a complete
`nash_equilibria_all_sizes.json`, so the folder can be interpreted without
loading results from the other split. Universal figures are present in both
folders; older shared-final heatmaps and parameter snapshots are preserved in
`results/final_100-50/legacy_final_outputs/`.

The appendix workflows above are the supplementary dataset-size and
participant-imbalance experiments. Other unrelated legacy plotting and generic
Parquet-comparison helpers are not part of this codebase.

## Generate the API documentation

Install the documentation-only dependency and build HTML from the Python
docstrings:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The Sphinx configuration uses
autodoc and Napoleon to render module, class, and function documentation.
Generated files remain in `docs/_build/` and are not source files.
