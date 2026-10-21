import os
import numpy as np
from tqdm import tqdm

import torch
from torch_molecule import GDSSMolecularGenerator

EPOCHS = 5
BATCH_SIZE = 16

def test_gdss_generator():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    smiles_list = smiles_list * 25  # Create 100 molecules for training

    # 1. Basic initialization test
    print("\n=== Testing GDSS model initialization ===")
    model = GDSSMolecularGenerator(
        num_layer=3,
        hidden_size_adj=8,
        hidden_size=16,
        attention_dim=16,
        num_head=4,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=0.005,
        verbose="print_statement"
    )
    print("GDSS Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing GDSS model fitting ===")
    model.fit(smiles_list)
    print("GDSS Model fitting completed")

    # 3. Generation test
    print("\n=== Testing GDSS model generation ===")
    generated_smiles = model.generate(batch_size=BATCH_SIZE)
    print(f"Generated {len(generated_smiles)} molecules")
    print("Example generated SMILES:", generated_smiles[:5])

    # 4. Model saving and loading test
    print("\n=== Testing GDSS model saving and loading ===")
    save_path = "gdss_test_model.pt"
    model.save_to_local(save_path)
    print(f"GDSS Model saved to {save_path}")

    new_model = GDSSMolecularGenerator()
    new_model.load_from_local(save_path)
    print("GDSS Model loaded successfully")

    # Test generation with loaded model
    generated_smiles = new_model.generate(batch_size=BATCH_SIZE)
    print(f"Generated molecules with loaded model: {len(generated_smiles)}")
    print("Example generated SMILES:", generated_smiles[:5])

    # 5. Test generation with specific number of nodes
    print("\n=== Testing generation with specific node counts ===")
    num_nodes = np.array([[20], [25], [30], [35]])  # Specify different node counts
    generated_smiles = model.generate(num_nodes=num_nodes)
    print(f"Generated molecules with specific node counts: {len(generated_smiles)}")
    print("Example generated SMILES:", generated_smiles)

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_gdss_generator()