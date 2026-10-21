GNNMolecularPredictor
=====================

GIN or GCN message passing, with optional virtual nodes.

.. code-block:: python

   from torch_molecule import GNNMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor.predict

Example
-------

:download:`GNN example <../../../../examples/prediction/gnn.py>`.
