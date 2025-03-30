Molecular Generation Models
=============================

Modeling Molecules as Graphs with Graph Neural Networks 
-------------------------------------------------------

.. rubric:: GraphDiT for Un/Multi-conditional Molecular Generation
.. autoclass:: torch_molecule.generator.graph_dit.modeling_graph_dit.GraphDITMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_name, model_class
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

.. rubric:: DiGress for Unconditional Molecular Generation
.. autoclass:: torch_molecule.generator.digress.modeling_digress.DigressMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, model_class, dataset_info, model_name
   :members: fit, generate
   :undoc-members:
   :show-inheritance:

Modeling Molecules with Heuristic-based Generators
---------------------------------------------------

.. rubric:: Graph Genetic Algorithm for Un/Multi-conditional Molecular Generation
.. autoclass:: torch_molecule.generator.graph_ga.modeling_graph_ga.GraphGAMolecularGenerator
   :exclude-members: fitting_epoch, fitting_loss, push_to_huggingface, load_from_huggingface
   :members: fit, generate
   :undoc-members:
   :show-inheritance:
