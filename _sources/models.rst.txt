Model collection
================

Choose from **31 model classes** for property prediction, molecular representations,
and generation. Models accept SMILES strings and handle their own graph or sequence
representations. Select a model below to see its parameters and method signatures.
Example links open the code directly in each model’s reference page.

Predict properties
------------------

Fit on molecules and labels, then predict regression or classification targets.
Start with GNN for a graph baseline or a SMILES model for a sequence baseline.

.. list-table:: 11 predictors
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Approach
     - Example
   * - :class:`GNN <torch_molecule.predictor.gnn.modeling_gnn.GNNMolecularPredictor>`
     - GIN or GCN message passing, with optional virtual nodes.
     - :ref:`GNN example <GNNMolecularPredictor-example>`
   * - :class:`GREA <torch_molecule.predictor.grea.modeling_grea.GREAMolecularPredictor>`
     - Graph rationales and environment-based data augmentation.
     - :ref:`GREA example <GREAMolecularPredictor-example>`
   * - :class:`SGIR <torch_molecule.predictor.sgir.modeling_sgir.SGIRMolecularPredictor>`
     - Semi-supervised learning for imbalanced regression targets.
     - :ref:`SGIR example <SGIRMolecularPredictor-example>`
   * - :class:`GRIN <torch_molecule.predictor.grin.modeling_grin.GRINMolecularPredictor>`
     - Repetition-invariant graph representations for polymers.
     - :ref:`GRIN example <GRINMolecularPredictor-example>`
   * - :class:`BFGNN <torch_molecule.predictor.bfgnn.modeling_bfgnn.BFGNNMolecularPredictor>`
     - Graph learning aligned with the Bellman–Ford algorithm.
     - :ref:`BFGNN example <BFGNNMolecularPredictor-example>`
   * - :class:`DIR <torch_molecule.predictor.dir.modeling_dir.DIRMolecularPredictor>`
     - Learning invariant graph rationales.
     - :ref:`DIR example <DIRMolecularPredictor-example>`
   * - :class:`IRM <torch_molecule.predictor.irm.modeling_irm.IRMMolecularPredictor>`
     - Invariant risk minimization with graph neural networks.
     - :ref:`IRM example <IRMMolecularPredictor-example>`
   * - :class:`RPGNN <torch_molecule.predictor.rpgnn.modeling_rpgnn.RPGNNMolecularPredictor>`
     - Relational pooling over node permutations.
     - :ref:`RPGNN example <RPGNNMolecularPredictor-example>`
   * - :class:`SSR <torch_molecule.predictor.ssr.modeling_ssr.SSRMolecularPredictor>`
     - Regularization for generalization across graph sizes.
     - :ref:`SSR example <SSRMolecularPredictor-example>`
   * - :class:`LSTM <torch_molecule.predictor.lstm.modeling_lstm.LSTMMolecularPredictor>`
     - Recurrent sequence learning from SMILES.
     - :ref:`LSTM example <LSTMMolecularPredictor-example>`
   * - :class:`SMILES Transformer <torch_molecule.predictor.smiles_transformer.modeling_transformer.SMILESTransformerMolecularPredictor>`
     - Transformer sequence learning from SMILES.
     - :ref:`Transformer example <SMILESTransformerMolecularPredictor-example>`

Learn molecular representations
-------------------------------

Encode molecules as vectors for downstream prediction, clustering, or similarity
analysis. Self-supervised encoders train on unlabeled SMILES; supervised encoders
need the labels or pairs described below.

.. list-table:: 9 encoders
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Training signal
     - Example
   * - :class:`AttrMask <torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder>`
     - Reconstruct masked atom attributes.
     - :ref:`AttrMask example <AttrMaskMolecularEncoder-example>`
   * - :class:`MoAmA <torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder>`
     - Motif-aware attribute masking and fingerprint reconstruction.
     - :ref:`MoAmA example <MoamaMolecularEncoder-example>`
   * - :class:`GraphMAE <torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder>`
     - Graph masked autoencoding.
     - :ref:`GraphMAE example <GraphMAEMolecularEncoder-example>`
   * - :class:`ContextPred <torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder>`
     - Predict molecular subgraph contexts.
     - :ref:`ContextPred example <ContextPredMolecularEncoder-example>`
   * - :class:`EdgePred <torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder>`
     - Predict whether pairs of atoms are connected.
     - :ref:`EdgePred example <EdgePredMolecularEncoder-example>`
   * - :class:`InfoGraph <torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder>`
     - Learn agreement between graph and substructure representations.
     - :ref:`InfoGraph example <InfoGraphMolecularEncoder-example>`
   * - :class:`Supervised <torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder>`
     - Molecular property labels or computed molecular descriptors.
     - :ref:`Supervised example <SupervisedMolecularEncoder-example>`
   * - :class:`SAND <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder>`
     - Pairwise similarity scores; dense or compressed molecular descriptors.
     - :doc:`API and input format <api/models/SANDMolecularEncoder>`
   * - :class:`Hugging Face <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder>`
     - Load a pretrained sequence encoder, such as ChemBERTa or ChemGPT.
     - :ref:`Pretrained encoder example <HFPretrainedMolecularEncoder-example>`

Generate molecules
------------------

Train a generator on molecules or load a pretrained model. Conditioning varies by
method: property labels, molecular scaffolds, and scored demonstrations serve
different purposes and have different input formats.

.. list-table:: 11 generators
   :header-rows: 1
   :widths: 18 62 20

   * - Model
     - Approach and supported conditioning
     - Example
   * - :class:`DemoDiff <torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator>`
     - Graph diffusion with optional molecular demonstrations and normalized ``score`` values.
     - :ref:`DemoDiff example <DemoDiffMolecularGenerator-example>`
   * - :class:`GenMol <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator>`
     - Masked diffusion over SAFE fragments, optionally retaining a supplied scaffold.
     - :doc:`API and usage <api/models/GenMolMolecularGenerator>`
   * - :class:`GraphDiT <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator>`
     - Graph diffusion transformer with optional property labels.
     - :ref:`GraphDiT example <GraphDITMolecularGenerator-example>`
   * - :class:`DeFoG <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator>`
     - Discrete graph flow matching with optional property labels.
     - :ref:`DeFoG example <DeFoGMolecularGenerator-example>`
   * - :class:`DiGress <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator>`
     - Unconditional discrete graph diffusion.
     - :ref:`DiGress example <DigressMolecularGenerator-example>`
   * - :class:`GDSS <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator>`
     - Unconditional score-based graph diffusion.
     - :ref:`GDSS example <GDSSMolecularGenerator-example>`
   * - :class:`JT-VAE <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator>`
     - Unconditional junction-tree variational autoencoding.
     - :ref:`JT-VAE example <JTVAEMolecularGenerator-example>`
   * - :class:`Graph GA <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator>`
     - Genetic search using property labels or a scoring oracle.
     - :ref:`Graph GA example <GraphGAMolecularGenerator-example>`
   * - :class:`MolGPT <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator>`
     - Autoregressive SMILES transformer with optional properties and scaffold tokens.
     - :ref:`MolGPT example <MolGPTMolecularGenerator-example>`
   * - :class:`LSTM <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator>`
     - Recurrent SMILES generation with optional property labels.
     - :ref:`LSTM example <LSTMMolecularGenerator-example>`
   * - :class:`Hugging Face <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator>`
     - NovoMolGen, MolGen, Molexar, or SAFE-GPT checkpoints; constraints depend on the model family.
     - :ref:`SAFE-GPT example <HFPretrainedMolecularGenerator-example>`

See :doc:`install` for optional dependencies and :doc:`example` for complete
workflows, including dataset preparation and checkpoint loading.
