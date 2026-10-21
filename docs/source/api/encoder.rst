Encoder API
===========

Learn molecular representations and return one embedding per input SMILES with
``encode(X)``. Trainable encoders learn from your data; the Hugging Face encoder
loads an existing checkpoint with ``fit()``.

.. list-table:: Common operations
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``fit(...)``
     - Train an encoder. Supervised models need labels; SAND needs pairs and similarity scores.
   * - ``encode(X, return_type="np")``
     - Return molecular embeddings as a NumPy array; use ``"pt"`` for a PyTorch tensor.
   * - ``save_to_local(path)`` / ``load_from_local(path)``
     - Save or restore a trained model.

Compare all 9 encoders in the :doc:`model catalog <../models>`. See
:doc:`../install` for optional pretrained-model dependencies.

Self-supervised encoders
------------------------

.. rubric:: MoAmA: motif-aware attribute masking
.. autoclass:: torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Attribute Masking for Molecular Representation Learning

.. autoclass:: torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Graph masked autoencoder

.. autoclass:: torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Context Prediction

.. autoclass:: torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: Edge Prediction

.. autoclass:: torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: InfoGraph

.. autoclass:: torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

Supervised encoders
-------------------

.. rubric:: Pretraining with Supervised/Pseudolabeled Data
.. autoclass:: torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder
   :members: fit, encode
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :undoc-members:
   :show-inheritance:

.. rubric:: SAND Shape Aware Neural Descriptor

.. autoclass:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder
   :members: fit, encode, load_from_pretrained, save_to_local, load_from_local
   :show-inheritance:

Pretrained encoders
-------------------

.. rubric:: Sequence-based Pretrained Transformers from Hugging Face

.. autoclass:: torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder
   :members: fit, encode, load, load_from_hf
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
