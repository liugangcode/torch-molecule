JTVAEMolecularGenerator
=======================

Unconditional junction-tree variational autoencoding.

.. code-block:: python

   from torch_molecule import JTVAEMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.jtvae.modeling_jtvae.JTVAEMolecularGenerator.generate

.. _JTVAEMolecularGenerator-example:

Example
-------

.. literalinclude:: ../../../../examples/generation/jtvae.py
   :language: python
   :caption: examples/generation/jtvae.py
   :class: tm-example
