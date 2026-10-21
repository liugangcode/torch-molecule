SupervisedMolecularEncoder
==========================

Molecular property labels or computed molecular descriptors.

.. code-block:: python

   from torch_molecule import SupervisedMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.supervised.modeling_supervised.SupervisedMolecularEncoder.encode

Example
-------

:download:`Supervised example <../../../../examples/encoder/supervised.py>`.
