GRINMolecularPredictor
======================

Repetition-invariant graph representations for polymers.

.. code-block:: python

   from torch_molecule import GRINMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor.predict

.. _GRINMolecularPredictor-example:

Example
-------

.. literalinclude:: ../../../../examples/prediction/grin.py
   :language: python
   :caption: examples/prediction/grin.py
   :class: tm-example
