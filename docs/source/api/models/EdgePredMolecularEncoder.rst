EdgePredMolecularEncoder
========================

Predict whether pairs of atoms are connected.

.. code-block:: python

   from torch_molecule import EdgePredMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.edgepred.modeling_edgepred.EdgePredMolecularEncoder.encode

Example
-------

:download:`EdgePred example <../../../../examples/encoder/edgepred.py>`.
