GraphDITMolecularGenerator
==========================

Graph diffusion transformer with optional property labels.

.. code-block:: python

   from torch_molecule import GraphDITMolecularGenerator

Constructor
-----------

.. autoclass:: torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Purpose
   * - :meth:`fit <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.fit>`
     - Fit the model to the supplied data.
   * - :meth:`generate <torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.generate>`
     - Sample molecules with the supported conditioning inputs.

For shared checkpoint and parameter operations, see :doc:`../base`.

fit
~~~

.. automethod:: torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.fit

generate
~~~~~~~~

.. automethod:: torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator.generate

.. _GraphDITMolecularGenerator-example:

Example
-------

.. literalinclude:: ../../../../examples/generation/graphdit.py
   :language: python
   :caption: examples/generation/graphdit.py
   :class: tm-example
