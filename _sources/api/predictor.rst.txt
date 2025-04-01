Molecular Property Prediction Models
=====================================

The predictor models inherit from the :class:`torch_molecule.base.predictor.BaseMolecularPredictor` class and share common methods for model training, evaluation, prediction and persistence.

.. rubric:: Training and Prediction

- ``fit(X, y, **kwargs)``: Train the model on given data, where X contains SMILES strings and y contains target values
- ``autofit(X, y, search_parameters, n_trials=50, **kwargs)``: Automatically search for optimal hyperparameters using Optuna and train the model
- ``predict(X, **kwargs)``: Make predictions on new SMILES strings and return a dictionary containing predictions and optional uncertainty estimates

.. rubric:: Model Persistence

inherited from :class:`torch_molecule.base.base.BaseModel`

- ``save_to_local(path)``: Save the trained model to a local file
- ``load_from_local(path)``: Load a trained model from a local file
- ``save_to_hf(repo_id)``: Push the model to Hugging Face Hub
- ``load_from_hf(repo_id, local_cache)``: Load a model from Hugging Face Hub and save it to a local file
- ``save(path, repo_id)``: Save the model to either local storage or Hugging Face
- ``load(path, repo_id)``: Load a model from either local storage or Hugging Face

Modeling Molecules as Graphs with Graph Neural Networks 
-------------------------------------------------------

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

.. rubric:: Discovering Invariant Rationales
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

Modeling Molecules as Sequences with RNNs
-----------------------------------------

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