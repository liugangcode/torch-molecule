RPGNNMolecularPredictor
=======================

Relational pooling over node permutations.

.. code-block:: python

   from torch_molecule import RPGNNMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor.predict

Example
-------

:download:`RPGNN example <../../../../examples/prediction/rpgnn.py>`.
