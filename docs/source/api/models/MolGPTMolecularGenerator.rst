MolGPTMolecularGenerator
========================

Autoregressive SMILES transformer with optional properties and scaffold tokens.

.. code-block:: python

   from torch_molecule import MolGPTMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.molgpt.modeling_molgpt.MolGPTMolecularGenerator.generate

.. _MolGPTMolecularGenerator-example:

Example
-------

.. literalinclude:: ../../../../examples/generation/molgpt.py
   :language: python
   :caption: examples/generation/molgpt.py
   :class: tm-example
