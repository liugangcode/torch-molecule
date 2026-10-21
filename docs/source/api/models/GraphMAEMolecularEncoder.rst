GraphMAEMolecularEncoder
========================

Graph masked autoencoding.

.. code-block:: python

   from torch_molecule import GraphMAEMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.graphmae.modeling_graphmae.GraphMAEMolecularEncoder.encode

.. _GraphMAEMolecularEncoder-example:

Example
-------

.. literalinclude:: ../../../../examples/encoder/graphmae.py
   :language: python
   :caption: examples/encoder/graphmae.py
   :class: tm-example
