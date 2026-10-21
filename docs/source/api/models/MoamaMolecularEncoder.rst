MoamaMolecularEncoder
=====================

Motif-aware attribute masking and fingerprint reconstruction.

.. code-block:: python

   from torch_molecule import MoamaMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.moama.modeling_moama.MoamaMolecularEncoder.encode

.. _MoamaMolecularEncoder-example:

Example
-------

.. literalinclude:: ../../../../examples/encoder/moama.py
   :language: python
   :caption: examples/encoder/moama.py
   :class: tm-example
