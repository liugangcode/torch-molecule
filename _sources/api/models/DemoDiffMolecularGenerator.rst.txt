DemoDiffMolecularGenerator
==========================

Graph diffusion with optional molecular demonstrations and normalized ``score`` values.

Usage
-----

DemoDiff trains on SMILES and optionally uses other molecules as demonstrations.
In ``fit()``, ``contexts[i]`` lists indices of other training molecules for target
``i``; in ``generate()``, ``contexts`` contains demonstration SMILES. Supply
``score`` as normalized values in [0, 1], with higher values meaning more
target-like demonstrations. The target's score is fixed to 1 internally.
Use the same score definition in training and generation. The local example
uses Morgan fingerprint Tanimoto similarity: 1 means identical fingerprints
and 0 means no shared fingerprint bits. During generation, scores describe the
demonstrations in the desired task; they do not guarantee measured similarities
of generated molecules. Omit both ``contexts`` and ``score`` for
unconditional training or generation.

.. code-block:: python

   from torch_molecule import DemoDiffMolecularGenerator

   smiles = ["CCOCCN", "c1ccccc1CCO", "CC(=O)NCCO"]
   model = DemoDiffMolecularGenerator(
       num_layer=2, hidden_size=64, num_head=4,
       diffusion_steps=50, epochs=20, batch_size=3,
   )
   model.fit(
       smiles,
       contexts=[[1], [0], [0, 1]],
       score=[[0.8], [0.8], [0.6, 0.7]],
   )
   generated = model.generate(
       n_samples=10, contexts=[smiles[0]], score=[0.8],
   )

The scores above are illustrative. Demonstrations guide generation;
they are not guaranteed substructures. The tokenizer can lose stereochemistry
and formal charges, while polymer attachment atoms are supported.

.. model-reference:: torch_molecule.generator.demodiff.modeling_demodiff.DemoDiffMolecularGenerator
   :methods: fit, generate
   :example: examples/generation/demodiff.py
