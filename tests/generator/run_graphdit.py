import os
import numpy as np
from tqdm import tqdm

import torch
from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.utils.search import ParameterType, ParameterSpec

EPOCHS = 10
BATCH_SIZE = 32

def test_graph_dit_generator():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    smiles_list = smiles_list * 25  # Create 100 molecules for training
    properties = None

    # 1. Basic initialization test
    print('smiles_list', smiles_list, 'properties', properties)
    print("\n=== Testing GraphDIT model initialization ===")
    model = GraphDITMolecularGenerator(
        # hidden_size=384,
        # num_layer=6,
        # num_head=16,
        # mlp_ratio=4.0,
        # dropout=0.0,
        # drop_condition=0.1,
        max_node=50,
        # X_dim=118,
        # E_dim=5,
        y_dim=None,
        task_type=['regression'],
        timesteps=500,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=True,
        guide_scale=2.0
    )
    print("GraphDIT Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing GraphDIT model fitting ===")
    model.fit(smiles_list, properties)
    print("GraphDIT Model fitting completed")

    # 3. Generation test
    print("\n=== Testing GraphDIT model generation ===")
    target_properties = None
    generated_smiles = model.generate(target_properties, batch_size=BATCH_SIZE)
    print(f"Generated {len(generated_smiles)} molecules")
    print("Example generated SMILES:", generated_smiles)

    # 5. Model saving and loading test
    print("\n=== Testing GraphDIT model saving and loading ===")
    save_path = "graph_dit_test_model.pt"
    model.save_to_local(save_path)
    print(f"GraphDIT Model saved to {save_path}")

    new_model = GraphDITMolecularGenerator()
    new_model.load_from_local(save_path)
    print("GraphDIT Model loaded successfully")

    # Test generation with loaded model
    generated_smiles = new_model.generate(target_properties)
    print("Generated molecules with loaded model:", len(generated_smiles))

    # 6. Test generation with specific number of nodes
    print("\n=== Testing generation with specific node counts ===")
    num_nodes = np.array([[20], [25], [30], [35]])  # Specify different node counts
    generated_smiles = model.generate(target_properties, num_nodes=num_nodes)
    print(f"Generated molecules with specific node counts: {len(generated_smiles)}, {generated_smiles}")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_graph_dit_generator()