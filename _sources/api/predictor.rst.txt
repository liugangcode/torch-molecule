Molecular Property Prediction Models
=====================================

This section documents the predictor models in `torch_molecule.predictor`.

.. contents::
   :local:
   :depth: 1

.. (Add predictors here as needed. You might write summaries and link submodules.)

.. rubric:: Graph Neural Networks
.. .. autoclass:: torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor
..    :members: fit, autofit, predict
..    :undoc-members: fitting_loss, fitting_epoch
..    :show-inheritance:
.. automodule:: torch_molecule.predictor.gnn
   :members:
   :exclude-members: fitting_epoch, fitting_loss
   :undoc-members:
   :show-inheritance:

.. rubric:: Graph Rationalization with Environment-based Data Augmentation
.. autoclass:: torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Semi-Supervised Graph Imbalanced Regression Models
.. autoclass:: torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Discovering Invariant Rationales
.. autoclass:: torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Invariant Risk Minimization with GNNs
.. autoclass:: torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: LSTM models based on SMILES
.. automodule:: torch_molecule.predictor.lstm
   :members:
   :exclude-members: fitting_epoch, fitting_loss
   :undoc-members:
   :show-inheritance:

.. rubric:: Relational Pooling for Graph Representations with GNNs
.. autoclass:: torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: SizeShiftReg: a Regularization Method for Improving Size-Generalization in GNNs
.. autoclass:: torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor  
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
