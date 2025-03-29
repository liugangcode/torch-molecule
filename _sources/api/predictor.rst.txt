Molecular Property Prediction Models
=====================================

This section documents the predictor models in `torch_molecule.predictor`.

.. (Add predictors here as needed. You might write summaries and link submodules.)

.. rubric:: Graph Neural Networks
.. autoclass:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: Graph Rationalization with Environment-based Data Augmentation
.. autoclass:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: Semi-Supervised Graph Imbalanced Regression Models
.. autoclass:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: Discovering Invariant Rationales
.. autoclass:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: Invariant Risk Minimization Models
.. autoclass:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: LSTM models based on SMILES
.. autoclass:: torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: Relational Pooling for Graph Representations with GNNs
.. autoclass:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch  
   :show-inheritance:

.. rubric:: SizeShiftReg: a Regularization Method for Improving Size-Generalization in GNNs
.. autoclass:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor  
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch  
   :show-inheritance:
