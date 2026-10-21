LSTMMolecularPredictor
======================

Recurrent sequence learning from SMILES.

.. code-block:: python

   from torch_molecule import LSTMMolecularPredictor

Constructor
-----------

.. autoclass:: torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.fit>`
     - Fit the model to the supplied data.
   * - :meth:`autofit <torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.autofit>`
     - Search hyperparameters and fit a model.
   * - :meth:`predict <torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.predict>`
     - Predict molecular properties.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.fit

autofit
~~~~~~~

.. automethod:: torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.autofit

predict
~~~~~~~

.. automethod:: torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor.predict

Example
-------

:download:`LSTM example <../../../../examples/prediction/lstm.py>`.
