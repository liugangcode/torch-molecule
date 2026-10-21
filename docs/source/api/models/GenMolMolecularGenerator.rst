GenMolMolecularGenerator
========================

Masked diffusion over SAFE fragments, optionally retaining a supplied scaffold.

Usage
-----

GenMol trains on SMILES strings and generates molecules using masked diffusion
over SAFE fragments. Pass ``scaffold="c1ccccc1"`` to ``generate()`` to retain a
benzene substructure in generated molecules. Install its optional packages with
``pip install ".[genmol]"`` from the source checkout.

.. model-reference:: torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator
   :methods: fit, generate, load_from_pretrained
