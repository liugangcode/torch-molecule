LSTMMolecularGenerator
======================

Recurrent SMILES generation with optional property labels.

.. code-block:: python

   from torch_molecule import LSTMMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.lstm.modeling_lstm.LSTMMolecularGenerator.generate

Example
-------

:download:`LSTM example <../../../../examples/generation/lstm.py>`.
