SMILESTransformerMolecularPredictor
===================================

Transformer sequence learning from SMILES.

.. code-block:: python

   from torch_molecule import SMILESTransformerMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor.predict

Example
-------

:download:`Transformer example <../../../../examples/prediction/smilestransformer.py>`.
