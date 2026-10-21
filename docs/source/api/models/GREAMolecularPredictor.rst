GREAMolecularPredictor
======================

Graph rationales and environment-based data augmentation.

.. code-block:: python

   from torch_molecule import GREAMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor.predict

.. _GREAMolecularPredictor-example:

Example
-------

.. literalinclude:: ../../../../examples/prediction/grea.py
   :language: python
   :caption: examples/prediction/grea.py
   :class: tm-example
