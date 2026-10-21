HFPretrainedMolecularEncoder
============================

Load a pretrained sequence encoder, such as ChemBERTa or ChemGPT.

.. code-block:: python

   from torch_molecule import HFPretrainedMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.fit>`
     - Load the pretrained model and tokenizer.
   * - :meth:`encode <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.encode>`
     - Return a representation for each molecule.
   * - :meth:`load <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load>`
     - Load a pretrained model and tokenizer.
   * - :meth:`load_from_hf <torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load_from_hf>`
     - Load a model from Hugging Face.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.encode

load
~~~~

.. automethod:: torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load

load_from_hf
~~~~~~~~~~~~

.. automethod:: torch_molecule.encoder.pretrained.modeling_pretrained.HFPretrainedMolecularEncoder.load_from_hf

.. _HFPretrainedMolecularEncoder-example:

Example
-------

.. literalinclude:: ../../../../examples/encoder/hfpretrained.py
   :language: python
   :caption: examples/encoder/hfpretrained.py
   :class: tm-example
