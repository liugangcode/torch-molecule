SSRMolecularPredictor
=====================

Regularization for generalization across graph sizes.

.. code-block:: python

   from torch_molecule import SSRMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor.predict

Example
-------

:download:`SSR example <../../../../examples/prediction/ssr.py>`.
