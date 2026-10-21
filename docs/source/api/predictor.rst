Predictor API
=============

Train molecular property models with ``fit(X_train, y_train)`` and predict on
new SMILES with ``predict(X)``. Predictions are returned in a dictionary; see
each class for its output fields and task-specific options.

.. list-table:: Common operations
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``fit(...)``
     - Train with SMILES and regression or classification labels.
   * - ``autofit(...)``
     - Search hyperparameters with Optuna and train a model.
   * - ``predict(X)``
     - Predict properties for a list of SMILES.
   * - ``save_to_local(path)`` / ``load_from_local(path)``
     - Save or restore a trained model.

Compare all 11 predictors in the :doc:`model catalog <../models>`. Checkpoint
and Hugging Face methods are documented in :doc:`base`.

Graph models
------------

.. rubric:: Graph Neural Networks
.. autoclass:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Graph Rationalization with Environment-based Data Augmentation
.. autoclass:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Semi-Supervised Graph Imbalanced Regression Models
.. autoclass:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Learning Repetition-Invariant Representations for Polymer Informatics
.. autoclass:: torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Graph neural networks extrapolate out-of-distribution for shortest paths
.. autoclass:: torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Discovering Invariant Rationales for Graph Neural Networks
.. autoclass:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Invariant Risk Minimization with GNNs
.. autoclass:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Relational Pooling for Graph Representations with GNNs
.. autoclass:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: SizeShiftReg: a Regularization Method for Improving Size-Generalization in GNNs
.. autoclass:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor  
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

Sequence models
---------------

.. rubric:: LSTM models based on SMILES
.. autoclass:: torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor
   :members: fit, autofit, predict
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Transformer models based on SMILES
.. autoclass:: torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor
   :members: fit, autofit, predict
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:
