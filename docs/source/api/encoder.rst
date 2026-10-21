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

Compare all 9 encoders in the :doc:`model collection <../models>`. See
:doc:`../install` for optional pretrained-model dependencies.


Model reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Class
     - Purpose
   * - :class:`MoamaMolecularEncoder <torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder>`
     - Motif-aware attribute masking and fingerprint reconstruction.

   * - :class:`AttrMaskMolecularEncoder <torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder>`
     - Reconstruct masked atom attributes.

   * - :class:`GraphMAEMolecularEncoder <torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder>`
     - Graph masked autoencoding.

   * - :class:`ContextPredMolecularEncoder <torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder>`
     - Predict molecular subgraph contexts.

   * - :class:`EdgePredMolecularEncoder <torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder>`
     - Predict whether pairs of atoms are connected.

   * - :class:`InfoGraphMolecularEncoder <torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder>`
     - Learn agreement between graph and substructure representations.

   * - :class:`SupervisedMolecularEncoder <torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder>`
     - Molecular property labels or computed molecular descriptors.

   * - :class:`SANDMolecularEncoder <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder>`
     - Pairwise similarity scores; dense or compressed molecular descriptors.

   * - :class:`HFPretrainedMolecularEncoder <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder>`
     - Load a pretrained sequence encoder, such as ChemBERTa or ChemGPT.

.. toctree::
   :hidden:

   MoAmA <models/MoamaMolecularEncoder>
   AttrMask <models/AttrMaskMolecularEncoder>
   GraphMAE <models/GraphMAEMolecularEncoder>
   ContextPred <models/ContextPredMolecularEncoder>
   EdgePred <models/EdgePredMolecularEncoder>
   InfoGraph <models/InfoGraphMolecularEncoder>
   Supervised <models/SupervisedMolecularEncoder>
   SAND <models/SANDMolecularEncoder>
   Hugging Face <models/HFPretrainedMolecularEncoder>

.. raw:: html

   <a class="tm-api-redirect" id="torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder" href="models/MoamaMolecularEncoder.html#torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder">MoamaMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.fit" href="models/MoamaMolecularEncoder.html#torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.fit">MoamaMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.encode" href="models/MoamaMolecularEncoder.html#torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.encode">MoamaMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder" href="models/AttrMaskMolecularEncoder.html#torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder">AttrMaskMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.fit" href="models/AttrMaskMolecularEncoder.html#torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.fit">AttrMaskMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.encode" href="models/AttrMaskMolecularEncoder.html#torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.encode">AttrMaskMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder" href="models/GraphMAEMolecularEncoder.html#torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder">GraphMAEMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.fit" href="models/GraphMAEMolecularEncoder.html#torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.fit">GraphMAEMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.encode" href="models/GraphMAEMolecularEncoder.html#torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.encode">GraphMAEMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder" href="models/ContextPredMolecularEncoder.html#torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder">ContextPredMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.fit" href="models/ContextPredMolecularEncoder.html#torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.fit">ContextPredMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.encode" href="models/ContextPredMolecularEncoder.html#torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.encode">ContextPredMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder" href="models/EdgePredMolecularEncoder.html#torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder">EdgePredMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.fit" href="models/EdgePredMolecularEncoder.html#torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.fit">EdgePredMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.encode" href="models/EdgePredMolecularEncoder.html#torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.encode">EdgePredMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder" href="models/InfoGraphMolecularEncoder.html#torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder">InfoGraphMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.fit" href="models/InfoGraphMolecularEncoder.html#torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.fit">InfoGraphMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.encode" href="models/InfoGraphMolecularEncoder.html#torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.encode">InfoGraphMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder" href="models/SupervisedMolecularEncoder.html#torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder">SupervisedMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.fit" href="models/SupervisedMolecularEncoder.html#torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.fit">SupervisedMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.encode" href="models/SupervisedMolecularEncoder.html#torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.encode">SupervisedMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder" href="models/SANDMolecularEncoder.html#torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder">SANDMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.fit" href="models/SANDMolecularEncoder.html#torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.fit">SANDMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.encode" href="models/SANDMolecularEncoder.html#torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.encode">SANDMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_pretrained" href="models/SANDMolecularEncoder.html#torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_pretrained">SANDMolecularEncoder.load_from_pretrained reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.save_to_local" href="models/SANDMolecularEncoder.html#torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.save_to_local">SANDMolecularEncoder.save_to_local reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_local" href="models/SANDMolecularEncoder.html#torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_local">SANDMolecularEncoder.load_from_local reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder" href="models/HFPretrainedMolecularEncoder.html#torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder">HFPretrainedMolecularEncoder reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.fit" href="models/HFPretrainedMolecularEncoder.html#torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.fit">HFPretrainedMolecularEncoder.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.encode" href="models/HFPretrainedMolecularEncoder.html#torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.encode">HFPretrainedMolecularEncoder.encode reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load" href="models/HFPretrainedMolecularEncoder.html#torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load">HFPretrainedMolecularEncoder.load reference</a>
   <a class="tm-api-redirect" id="torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load_from_hf" href="models/HFPretrainedMolecularEncoder.html#torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load_from_hf">HFPretrainedMolecularEncoder.load_from_hf reference</a>
