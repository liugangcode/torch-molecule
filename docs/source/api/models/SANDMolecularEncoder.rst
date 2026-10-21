SANDMolecularEncoder
====================

Pairwise similarity scores; dense or compressed molecular descriptors.

.. code-block:: python

   from torch_molecule import SANDMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.encode>`
     - Return a representation for each molecule.
   * - :meth:`load_from_pretrained <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_pretrained>`
     - Load a pretrained checkpoint.
   * - :meth:`save_to_local <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.save_to_local>`
     - Save model weights and configuration.
   * - :meth:`load_from_local <torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_local>`
     - Restore a saved model.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.encode

load_from_pretrained
~~~~~~~~~~~~~~~~~~~~

.. automethod:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_pretrained

save_to_local
~~~~~~~~~~~~~

.. automethod:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.save_to_local

load_from_local
~~~~~~~~~~~~~~~

.. automethod:: torch_molecule.encoder.sand.modeling_sand.SANDMolecularEncoder.load_from_local
