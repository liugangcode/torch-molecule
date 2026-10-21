DIRMolecularPredictor
=====================

Learning invariant graph rationales.

.. code-block:: python

   from torch_molecule import DIRMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor.predict

Example
-------

:download:`DIR example <../../../../examples/prediction/dir.py>`.
