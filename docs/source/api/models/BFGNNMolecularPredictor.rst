BFGNNMolecularPredictor
=======================

Graph learning aligned with the Bellman–Ford algorithm.

.. code-block:: python

   from torch_molecule import BFGNNMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor.predict

Example
-------

:download:`BFGNN example <../../../../examples/prediction/bfgnn.py>`.
