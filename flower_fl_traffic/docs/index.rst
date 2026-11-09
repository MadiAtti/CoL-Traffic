CoL-Traffic documentation
=========================

CoL-Traffic is a Flower-based research implementation for federated network
traffic classification, privacy-preserving training, payoff-matrix
construction, and Self-Division analysis.

The ``README.md`` alongside this documentation covers installation, data
preparation, experiment commands, and result locations. The API pages below
are generated from the English-language Python docstrings.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

Core workflow
-------------

The standard experiment entry point is :mod:`main`. It prepares data, runs
local baselines and federated scenario grids for the configured seed range,
then builds averaged game matrices. Visualization and Nash analysis are
separate commands documented in the README.
