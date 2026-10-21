GraphGAMolecularGenerator
=========================

Genetic search using property labels or a scoring oracle.

.. code-block:: python

   from torch_molecule import GraphGAMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator.generate

.. _GraphGAMolecularGenerator-example:

Example
-------

.. literalinclude:: ../../../../examples/generation/graphga.py
   :language: python
   :caption: examples/generation/graphga.py
   :class: tm-example

Scoring oracles
---------------

.. automodule:: torch_molecule.generator.graph_ga.oracle
   :members:
