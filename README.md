# CoL-Traffic

CoL-Traffic is a research codebase for collaborative and federated learning
experiments on network-traffic classification. The implementation compares
local training with Flower-based federated training and studies feature
suppression, differential privacy, and the resulting collaborative-learning
game matrices.

## Documentation

The traffic-classification implementation and its complete usage guide are in
[`flower_fl_traffic/`](./flower_fl_traffic/):

- [Experiment guide, setup, data requirements, and commands](./flower_fl_traffic/README.md)
- [Sphinx API documentation sources](./flower_fl_traffic/docs/)

The experiment guide covers the main pipeline, appendix experiments, result
files, privacy parameterization, and analysis scripts. Python modules contain
English docstrings used to generate the API reference.

## Quick start

Run commands from the implementation directory:

```bash
cd flower_fl_traffic
conda env create -f enviroment.yml
conda activate flower-fl-traffic
```

Download and place the required dataset as described in the
[experiment guide](./flower_fl_traffic/README.md#requirements-and-installation),
then run the main experiment:

```bash
python main.py
```

The main experiment is computationally intensive. The guide documents how to
select seeds and dataset modes, run the supplementary workflows, and generate
plots and game-analysis results.

## Build the API documentation

From `flower_fl_traffic/`, install the documentation dependency and build the
HTML reference:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to browse the generated documentation.
