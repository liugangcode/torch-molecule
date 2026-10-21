AttrMaskMolecularEncoder
========================

Reconstruct masked atom attributes.

.. code-block:: python

   from torch_molecule import AttrMaskMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.attrmask.modeling_attrmask.AttrMaskMolecularEncoder.encode

Example
-------

:download:`AttrMask example <../../../../examples/encoder/attrmask.py>`.
