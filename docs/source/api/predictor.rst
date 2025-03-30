Molecular Property Prediction Models
=====================================

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
   :members: fit, autofit, predict
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
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
.. automodule:: torch_molecule.predictor.lstm
   :members:
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance: