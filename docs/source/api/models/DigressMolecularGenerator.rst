DigressMolecularGenerator
=========================

Unconditional discrete graph diffusion.

.. code-block:: python

   from torch_molecule import DigressMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator.generate

Example
-------

:download:`DiGress example <../../../../examples/generation/digress.py>`.
