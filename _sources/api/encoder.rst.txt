Molecular Encoder Models
====================================================

The encoder models inherit from the :class:`torch_molecule.base.encoder.BaseMolecularEncoder` class and share common methods for model pretraining and encoding, as well as model persistence.

.. rubric:: Training and Encoding

- ``fit(X, **kwargs)``: Pretrain the model on given data, where X contains SMILES strings

  Not implemented for:
  - N/A

- ``encode(X, **kwargs)``: Encode new SMILES strings and return a dictionary containing encoded representations

.. rubric:: Model Persistence

inherited from :class:`torch_molecule.base.base.BaseModel`

- ``save_to_local(path)``: Save the trained model to a local file
- ``load_from_local(path)``: Load a trained model from a local file
- ``save_to_hf(repo_id)``: Push the model to Hugging Face Hub
- ``load_from_hf(repo_id, local_cache)``: Load a model from Hugging Face Hub and save it to a local file
- ``save(path, repo_id)``: Save the model to either local storage or Hugging Face
- ``load(path, repo_id)``: Load a model from either local storage or Hugging Face


Self-supervised Molecular Representation Learning
-------------------------------------------------

.. rubric:: MoAma for Molecular Representation Learning
.. automodule:: torch_molecule.encoder.moama
   :members: 
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Attribute Masking for Molecular Representation Learning

.. automodule:: torch_molecule.encoder.attrmask
   :members:
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Context Prediction for Molecular Representation Learning

.. automodule:: torch_molecule.encoder.contextpred
   :members:
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Edge Prediction for Molecular Representation Learning

.. automodule:: torch_molecule.encoder.edgepred
   :members:
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

Supervised Pretraining for Molecules
------------------------------------

.. rubric:: Supervised/Pseudolabeled Pretraining for Molecules
.. automodule:: torch_molecule.encoder.supervised
   :members:
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance: