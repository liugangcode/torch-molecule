import os
import numpy as np
from tqdm import tqdm

import torch
from torch_molecule import GraphDITMolecularGenerator

EPOCHS = 2
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
    # properties = [1.0, 2.0, 3.0, 4.0] * 25  # Create 100 properties for training
    properties = [0, 0, 1, 1] * 25  # Create 100 properties for training

    # 1. Basic initialization test - Conditional Model
    print('smiles_list', len(smiles_list), smiles_list[:5], 'properties', len(properties), properties[:5])
    print("\n=== Testing Conditional GraphDIT model initialization ===")
    conditional_model = GraphDITMolecularGenerator(
        task_type=['classification'],
        drop_condition=0.1,
        timesteps=500,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose="progress_bar",
        guide_scale=2.0,
        lw_X=1,
        lw_E=5
    )
    print("Conditional GraphDIT Model initialized successfully")

    # 2. Basic fitting test - Conditional Model
    print("\n=== Testing Conditional GraphDIT model fitting ===")
    conditional_model.fit(smiles_list, properties)
    print("Conditional GraphDIT Model fitting completed")

    # 3. Conditional generation test
    print("\n=== Testing Conditional GraphDIT generation ===")
    target_properties = [0, 0, 1, 1]
    generated_smiles = conditional_model.generate(target_properties, batch_size=BATCH_SIZE)
    print(f"Conditionally generated {len(generated_smiles)} molecules")
    print("Example conditionally generated SMILES:", generated_smiles[:2])
    
    # 4. Model saving and loading test - Conditional Model
    print("\n=== Testing Conditional GraphDIT model saving and loading ===")
    save_path = "conditional_graph_dit_test_model.pt"
    conditional_model.save_to_local(save_path)
    print(f"Conditional GraphDIT Model saved to {save_path}")

    new_conditional_model = GraphDITMolecularGenerator()
    new_conditional_model.load_from_local(save_path)
    print("Conditional GraphDIT Model loaded successfully")

    # Test generation with loaded conditional model
    generated_smiles = new_conditional_model.generate(target_properties)
    print("Generated molecules with loaded conditional model:", len(generated_smiles))
    
    # 5. Test generation with specific node counts - Conditional Model
    print("\n=== Testing conditional generation with specific node counts ===")
    num_nodes = np.array([[20], [25], [30], [35]])  # Specify different node counts
    
    # Conditional generation with specific node counts
    generated_smiles = conditional_model.generate(target_properties, num_nodes=num_nodes)
    print(f"Conditionally generated molecules with specific node counts: {len(generated_smiles)}")
    
    # Clean up conditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")
    
    # 6. Basic initialization test - Unconditional Model
    print("\n=== Testing Unconditional GraphDIT model initialization ===")
    unconditional_model = GraphDITMolecularGenerator(
        timesteps=500,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose="progress_bar"
    )
    print("Unconditional GraphDIT Model initialized successfully")

    # 7. Basic fitting test - Unconditional Model
    print("\n=== Testing Unconditional GraphDIT model fitting ===")
    unconditional_model.fit(smiles_list)
    print("Unconditional GraphDIT Model fitting completed")

    # 8. Unconditional generation test
    print("\n=== Testing Unconditional GraphDIT generation ===")
    generated_smiles_uncond = unconditional_model.generate(batch_size=BATCH_SIZE)
    print(f"Unconditionally generated {len(generated_smiles_uncond)} molecules")
    print("Example unconditionally generated SMILES:", generated_smiles_uncond[:2])
    
    # 9. Model saving and loading test - Unconditional Model
    print("\n=== Testing Unconditional GraphDIT model saving and loading ===")
    save_path = "unconditional_graph_dit_test_model.pt"
    unconditional_model.save_to_local(save_path)
    print(f"Unconditional GraphDIT Model saved to {save_path}")

    new_unconditional_model = GraphDITMolecularGenerator()
    new_unconditional_model.load_from_local(save_path)
    print("Unconditional GraphDIT Model loaded successfully")

    # Test generation with loaded unconditional model
    generated_smiles_uncond = new_unconditional_model.generate()
    print("Generated molecules with loaded unconditional model:", len(generated_smiles_uncond))
    
    # 10. Test generation with specific node counts - Unconditional Model
    print("\n=== Testing unconditional generation with specific node counts ===")
    
    # Unconditional generation with specific node counts
    generated_smiles_uncond = unconditional_model.generate(num_nodes=num_nodes)
    print(f"Unconditionally generated molecules with specific node counts: {len(generated_smiles_uncond)}")

    # Clean up unconditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_graph_dit_generator()