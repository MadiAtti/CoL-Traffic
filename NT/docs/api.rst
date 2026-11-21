API reference
=============

Experiment entry point
----------------------

.. automodule:: main
   :members:

Appendix experiment entry point
-------------------------------

.. automodule:: appendix_pipeline
   :members:

Data preparation
----------------

.. automodule:: data.splitter
   :members:

.. automodule:: data.dataset
   :members:

.. automodule:: data.custom_dataset
   :members:

Models
------

.. automodule:: models.neural_network
   :members:

Experiment runners
------------------

.. automodule:: experiment.local_baseline
   :members:

.. automodule:: experiment.experiment_runner
   :members:

.. automodule:: experiment.run_max_privacy_experiment
   :members:

Federated clients and server
----------------------------

.. automodule:: federated.universal_client
   :members:

.. automodule:: federated.echo_client
   :members:

.. automodule:: federated.server
   :members:

Training and metrics
--------------------

.. automodule:: utils.training
   :members:

.. automodule:: utils.evaluation
   :members:

.. automodule:: utils.metrics
   :members:

.. automodule:: utils.save
   :members:

.. automodule:: utils.seed
   :members:

.. automodule:: utils.logger_silencer
   :members:

Result matrices and the SD operator f (NT)
------------------------------------------

.. automodule:: utils.average_plots
   :members:

.. automodule:: utils.visualizer
   :members:

Game theory and cross-validation
--------------------------------

The game-theory analysis (paper Section 7) and the cross-domain validation are
shared by NT, DD and RS and live in the repository's ``GT/`` folder; see its
README.

Appendix experiments
--------------------

.. automodule:: utils.central_baseline
   :members:

.. automodule:: utils.multi_split
   :members:
