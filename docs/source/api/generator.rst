Generator API
=============

Fit a model on SMILES with ``fit(...)`` and sample molecules with
``generate(...)``. Conditional inputs and sample-count arguments differ by
model; use each model’s reference page when switching methods.

.. list-table:: Generation and conditioning
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Optional conditioning
   * - :class:`DemoDiff <torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator>`
     - Demonstration SMILES in ``contexts`` and normalized ``score`` values.
   * - :class:`GenMol <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator>`
     - A molecular substructure in ``scaffold``.
   * - :class:`GraphDiT <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator>`
     - Property targets in ``labels``.
   * - :class:`DeFoG <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator>`
     - Property targets in ``labels``.
   * - :class:`Graph GA <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator>`
     - Property targets in ``labels`` or a scoring oracle configured on the estimator.
   * - :class:`MolGPT <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator>`
     - ``properties`` and ``scaffolds``.
   * - :class:`LSTM <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator>`
     - Property targets in ``labels``.
   * - :class:`Hugging Face <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator>`
     - Scaffold, fragment, or property inputs, depending on the checkpoint family.
   * - :class:`DiGress <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator>`
     - Unconditional generation.
   * - :class:`GDSS <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator>`
     - Unconditional generation.
   * - :class:`JT-VAE <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator>`
     - Unconditional generation.

Use ``save_to_local(path)`` and ``load_from_local(path)`` to save and restore
trained models. Additional persistence methods are documented in :doc:`base`;
Graph GA does not support Hugging Face uploads or downloads.

See the :doc:`model collection <../models>` for method summaries and links to
examples. Optional dependencies are listed in :doc:`../install`.


Model reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Class
     - Purpose
   * - :class:`DemoDiffMolecularGenerator <torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator>`
     - Graph diffusion with optional molecular demonstrations and normalized ``score`` values.

   * - :class:`GraphDITMolecularGenerator <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator>`
     - Graph diffusion transformer with optional property labels.

   * - :class:`DeFoGMolecularGenerator <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator>`
     - Discrete graph flow matching with optional property labels.

   * - :class:`GraphGAMolecularGenerator <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator>`
     - Genetic search using property labels or a scoring oracle.

   * - :class:`DigressMolecularGenerator <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator>`
     - Unconditional discrete graph diffusion.

   * - :class:`GDSSMolecularGenerator <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator>`
     - Unconditional score-based graph diffusion.

   * - :class:`JTVAEMolecularGenerator <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator>`
     - Unconditional junction-tree variational autoencoding.

   * - :class:`GenMolMolecularGenerator <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator>`
     - Masked diffusion over SAFE fragments, optionally retaining a supplied scaffold.

   * - :class:`MolGPTMolecularGenerator <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator>`
     - Autoregressive SMILES transformer with optional properties and scaffold tokens.

   * - :class:`LSTMMolecularGenerator <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator>`
     - Recurrent SMILES generation with optional property labels.

   * - :class:`HFPretrainedMolecularGenerator <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator>`
     - NovoMolGen, MolGen, Molexar, or SAFE-GPT checkpoints; constraints depend on the model family.

.. toctree::
   :hidden:

   DemoDiff <models/DemoDiffMolecularGenerator>
   GraphDiT <models/GraphDITMolecularGenerator>
   DeFoG <models/DeFoGMolecularGenerator>
   Graph GA <models/GraphGAMolecularGenerator>
   DiGress <models/DigressMolecularGenerator>
   GDSS <models/GDSSMolecularGenerator>
   JT-VAE <models/JTVAEMolecularGenerator>
   GenMol <models/GenMolMolecularGenerator>
   MolGPT <models/MolGPTMolecularGenerator>
   LSTM <models/LSTMMolecularGenerator>
   Hugging Face <models/HFPretrainedMolecularGenerator>

.. raw:: html

   <a class="tm-api-redirect" id="torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator" href="models/DemoDiffMolecularGenerator.html#torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator">DemoDiffMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator.fit" href="models/DemoDiffMolecularGenerator.html#torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator.fit">DemoDiffMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator.generate" href="models/DemoDiffMolecularGenerator.html#torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator.generate">DemoDiffMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator" href="models/GraphDITMolecularGenerator.html#torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator">GraphDITMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.fit" href="models/GraphDITMolecularGenerator.html#torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.fit">GraphDITMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.generate" href="models/GraphDITMolecularGenerator.html#torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.generate">GraphDITMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator" href="models/DeFoGMolecularGenerator.html#torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator">DeFoGMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.fit" href="models/DeFoGMolecularGenerator.html#torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.fit">DeFoGMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.generate" href="models/DeFoGMolecularGenerator.html#torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.generate">DeFoGMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator" href="models/GraphGAMolecularGenerator.html#torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator">GraphGAMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.fit" href="models/GraphGAMolecularGenerator.html#torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.fit">GraphGAMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.generate" href="models/GraphGAMolecularGenerator.html#torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.generate">GraphGAMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator" href="models/DigressMolecularGenerator.html#torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator">DigressMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.fit" href="models/DigressMolecularGenerator.html#torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.fit">DigressMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.generate" href="models/DigressMolecularGenerator.html#torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.generate">DigressMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator" href="models/GDSSMolecularGenerator.html#torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator">GDSSMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.fit" href="models/GDSSMolecularGenerator.html#torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.fit">GDSSMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.generate" href="models/GDSSMolecularGenerator.html#torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.generate">GDSSMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator" href="models/JTVAEMolecularGenerator.html#torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator">JTVAEMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.fit" href="models/JTVAEMolecularGenerator.html#torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.fit">JTVAEMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.generate" href="models/JTVAEMolecularGenerator.html#torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.generate">JTVAEMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator" href="models/GenMolMolecularGenerator.html#torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator">GenMolMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.fit" href="models/GenMolMolecularGenerator.html#torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.fit">GenMolMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.generate" href="models/GenMolMolecularGenerator.html#torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.generate">GenMolMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.load_from_pretrained" href="models/GenMolMolecularGenerator.html#torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.load_from_pretrained">GenMolMolecularGenerator.load_from_pretrained reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator" href="models/MolGPTMolecularGenerator.html#torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator">MolGPTMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.fit" href="models/MolGPTMolecularGenerator.html#torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.fit">MolGPTMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.generate" href="models/MolGPTMolecularGenerator.html#torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.generate">MolGPTMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator" href="models/LSTMMolecularGenerator.html#torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator">LSTMMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.fit" href="models/LSTMMolecularGenerator.html#torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.fit">LSTMMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.generate" href="models/LSTMMolecularGenerator.html#torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.generate">LSTMMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator" href="models/HFPretrainedMolecularGenerator.html#torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator">HFPretrainedMolecularGenerator reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.fit" href="models/HFPretrainedMolecularGenerator.html#torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.fit">HFPretrainedMolecularGenerator.fit reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.generate" href="models/HFPretrainedMolecularGenerator.html#torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.generate">HFPretrainedMolecularGenerator.generate reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.save_to_local" href="models/HFPretrainedMolecularGenerator.html#torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.save_to_local">HFPretrainedMolecularGenerator.save_to_local reference</a>
   <a class="tm-api-redirect" id="torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.load_from_local" href="models/HFPretrainedMolecularGenerator.html#torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.load_from_local">HFPretrainedMolecularGenerator.load_from_local reference</a>
