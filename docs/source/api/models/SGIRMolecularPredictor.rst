SGIRMolecularPredictor
======================

Semi-supervised learning for imbalanced regression targets.

.. code-block:: python

   from torch_molecule import SGIRMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor.predict

.. _SGIRMolecularPredictor-example:

Example
-------

.. literalinclude:: ../../../../examples/prediction/sgir.py
   :language: python
   :caption: examples/prediction/sgir.py
   :class: tm-example
