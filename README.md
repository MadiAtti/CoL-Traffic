# CoL-Traffic

CoL-Traffic is a research codebase for collaborative and federated learning
experiments on network-traffic classification. The implementation compares
local training with Flower-based federated training and studies feature
suppression, differential privacy, and the resulting collaborative-learning
game matrices.

## Repository layout

| Folder | Content |
|---|---|
| [`NT/`](./NT/) | Network Traffic experiments (Flower), their post-processing and fit of the SD operator |
| [`DD/`](./DD/) | Drug Discovery experiments and their own post-processing (`DD/results/`) |
| [`RS/`](./RS/) | Recommender System experiments and their own post-processing (`RS/results/`) |
| [`GT/`](./GT/) | shared game-theory analysis and cross-domain validation for all three use-cases |
| `DD_old/` | an earlier copy of the DD code, kept for reference |

Each use-case produces its gain surfaces and its fit of the Self-Division
operator with its own pipeline; `GT/` reads those results (DD and RS through
`GT/import_usecase.py`) and runs the game theory and the cross-validation. See
[`GT/README.md`](./GT/README.md).

## Documentation

The traffic-classification implementation and its complete usage guide are in
[`NT/`](./NT/):

- [Experiment guide, setup, data requirements, and commands](./NT/README.md)
- [Sphinx API documentation sources](./NT/docs/)

The experiment guide covers the main pipeline, appendix experiments, result
files, privacy parameterization, and analysis scripts. Python modules contain
English docstrings used to generate the API reference.

## Quick start

Run commands from the implementation directory:

```bash
cd NT
conda env create -f enviroment.yml
conda activate flower-fl-traffic
```

Download and place the required dataset as described in the
[experiment guide](./NT/README.md#requirements-and-installation),
then run the main experiment:

```bash
python main.py
```

The main experiment is computationally intensive. The guide documents how to
select seeds and dataset modes, run the supplementary workflows, and generate
plots and game-analysis results.

## Build the API documentation

From `NT/`, install the documentation dependency and build the
HTML reference:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to browse the generated documentation.
