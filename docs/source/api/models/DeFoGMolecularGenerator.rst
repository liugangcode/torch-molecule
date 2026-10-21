DeFoGMolecularGenerator
=======================

Discrete graph flow matching with optional property labels.

.. code-block:: python

   from torch_molecule import DeFoGMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.defog.modeling_defog.DeFoGMolecularGenerator.generate

Example
-------

:download:`DeFoG example <../../../../examples/generation/defog.py>`.
