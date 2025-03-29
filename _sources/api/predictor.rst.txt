Molecular Property Prediction Models
=====================================

This section documents the predictor models in `torch_molecule.predictor`.

.. (Add predictors here as needed. You might write summaries and link submodules.)

.. rubric:: Graph Neural Networks
   
.. autoclass:: torch_molecule.predictor.gnn
   :members: fit, autofit, predict
   :undoc-members: fitting_loss, fitting_epoch
   :show-inheritance:
   :member-order: bysource

.. rubric:: Graph Rationalization with Environment-based Data Augmentation
.. autoclass:: torch_molecule.predictor.grea
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. rubric:: Semi-Supervised Graph Imabalanced Regression Models
.. autoclass:: torch_molecule.predictor.sgir
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. rubric:: Discovering Invariant Rationales
.. autoclass:: torch_molecule.predictor.dir
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. rubric:: Invariant Risk Minimization Models
.. autoclass:: torch_molecule.predictor.irm
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. rubric:: LSTM models based on SMILES
.. autoclass:: torch_molecule.predictor.lstm
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. rubric:: Relational Pooling for Graph Representations with GNNs
.. autoclass:: torch_molecule.predictor.rpgnn
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. rubric:: SizeShiftReg: a Regularization Method for Improving Size-Generalization in GNNs
.. autoclass:: torch_molecule.predictor.ssr
   :members: fit, autofit, predict
   :undoc-members:
   :show-inheritance:
   :member-order: bysource