Molecular Property Prediction Models
=====================================

This section documents the predictor models in `torch_molecule.predictor`.

.. (Add predictors here as needed. You might write summaries and link submodules.)

.. rubric:: Graph Neural Networks
.. automodule:: torch_molecule.predictor.gnn
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:

.. rubric:: Graph Rationalization with Environment-based Data Augmentation
.. automodule:: torch_molecule.predictor.grea
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Semi-Supervised Graph Imabalanced Regression Models
.. automodule:: torch_molecule.predictor.sgir
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Discovering Invariant Rationales
.. automodule:: torch_molecule.predictor.dir
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Invariant Risk Minimization Models
.. automodule:: torch_molecule.predictor.irm
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: LSTM models based on SMILES
.. automodule:: torch_molecule.predictor.lstm
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: Relational Pooling for Graph Representations with GNNs
.. automodule:: torch_molecule.predictor.rpgnn
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:

.. rubric:: SizeShiftReg: a Regularization Method for Improving Size-Generalization in GNNs
.. automodule:: torch_molecule.predictor.ssr
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
