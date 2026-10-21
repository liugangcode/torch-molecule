GDSSMolecularGenerator
======================

Unconditional score-based graph diffusion.

.. code-block:: python

   from torch_molecule import GDSSMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.gdss.modeling_gdss.GDSSMolecularGenerator.generate

.. _GDSSMolecularGenerator-example:

Example
-------

.. literalinclude:: ../../../../examples/generation/gdss.py
   :language: python
   :caption: examples/generation/gdss.py
   :class: tm-example
