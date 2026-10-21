ContextPredMolecularEncoder
===========================

Predict molecular subgraph contexts.

.. code-block:: python

   from torch_molecule import ContextPredMolecularEncoder

Constructor
-----------

.. autoclass:: torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.fit>`
     - Fit the model to the supplied data.
   * - :meth:`encode <torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.encode>`
     - Return a representation for each molecule.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.fit

encode
~~~~~~

.. automethod:: torch_molecule.encoder.contextpred.modeling_contextpred.ContextPredMolecularEncoder.encode

Example
-------

:download:`ContextPred example <../../../../examples/encoder/contextpred.py>`.
