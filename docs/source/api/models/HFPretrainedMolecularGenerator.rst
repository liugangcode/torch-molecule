HFPretrainedMolecularGenerator
==============================

NovoMolGen, MolGen, Molexar, or SAFE-GPT checkpoints; constraints depend on the model family.

.. code-block:: python

   from torch_molecule import HFPretrainedMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.fit>`
     - Load the pretrained model and tokenizer.
   * - :meth:`generate <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.
   * - :meth:`save_to_local <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.save_to_local>`
     - Save model weights and configuration.
   * - :meth:`load_from_local <torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.load_from_local>`
     - Restore a saved model.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.generate

save_to_local
~~~~~~~~~~~~~

.. automethod:: torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.save_to_local

load_from_local
~~~~~~~~~~~~~~~

.. automethod:: torch_molecule.generator.pretrained.modeling_pretrained.HFPretrainedMolecularGenerator.load_from_local

.. _HFPretrainedMolecularGenerator-example:

Example
-------

.. literalinclude:: ../../../../examples/generation/pretrained_safe_gpt.py
   :language: python
   :caption: examples/generation/pretrained_safe_gpt.py
   :class: tm-example
