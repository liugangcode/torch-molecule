InfoGraphMolecularEncoder
=========================

Learn agreement between graph and substructure representations.

.. code-block:: python

   from torch_molecule import InfoGraphMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.infograph.modeling_infograph.InfoGraphMolecularEncoder.encode

Example
-------

:download:`InfoGraph example <../../../../examples/encoder/infograph.py>`.
