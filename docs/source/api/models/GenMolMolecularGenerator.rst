GenMolMolecularGenerator
========================

Masked diffusion over SAFE fragments, optionally retaining a supplied scaffold.

.. code-block:: python

   from torch_molecule import GenMolMolecularGenerator

Usage
-----

GenMol trains on SMILES strings and generates molecules using masked diffusion
over SAFE fragments. Pass ``scaffold="c1ccccc1"`` to ``generate()`` to retain a
benzene substructure in generated molecules. Install its optional packages with
``pip install ".[genmol]"`` from the source checkout.

Constructor
-----------

.. autoclass:: torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.
   * - :meth:`load_from_pretrained <torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.load_from_pretrained>`
     - Load a pretrained checkpoint.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.generate

load_from_pretrained
~~~~~~~~~~~~~~~~~~~~

.. automethod:: torch_molecule.generator.genmol.modeling_genmol.GenMolMolecularGenerator.load_from_pretrained
