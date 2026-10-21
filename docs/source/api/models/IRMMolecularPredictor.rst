IRMMolecularPredictor
=====================

Invariant risk minimization with graph neural networks.

.. code-block:: python

   from torch_molecule import IRMMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor.predict

Example
-------

:download:`IRM example <../../../../examples/prediction/irm.py>`.
