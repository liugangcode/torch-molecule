import os
import numpy as np
import torch

from torch_molecule import DeFoGMolecularGenerator

EPOCHS = 10
BATCH_SIZE = 32

def test_defog_generator():
    """
    Test suite for the DeFoGMolecularGenerator, covering conditional and unconditional
    initialization, fitting, generation, saving, and loading.
    """
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    smiles_list = smiles_list * 25  # Create 100 molecules for training
    
    # Multi-dimensional properties: each row is a molecule, each column is a property
    # Properties range from 0 to 1
    np.random.seed(42)  # For reproducible results
    properties = np.random.rand(100, 3)  # 100 molecules, 3 properties each

    # 1. Multi-Conditional Model Testing
    print("\n=== Testing Multi-Conditional DeFoG Model ===")
    conditional_model = DeFoGMolecularGenerator(
        task_type=['regression', 'regression', 'regression'],  # 3 regression tasks
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=5e-4,
        sample_steps=10,  # Fewer steps for faster testing
        guidance_weight=0.2,
        verbose=True,
    )
    print("Multi-Conditional DeFoG Model initialized successfully.")
    print(f"Input dim y: {conditional_model.input_dim_y}")

    print("\n--- Fitting multi-conditional model ---")
    conditional_model.fit(smiles_list, properties)
    print("Multi-Conditional DeFoG Model fitting completed.")

    print("\n--- Testing multi-conditional generation ---")
    # Generate molecules with specific multi-dimensional properties
    target_properties = [
        [0.1, 0.2, 0.3],  # Low values for all properties
        [0.4, 0.5, 0.6],  # Medium values for all properties
        [0.7, 0.8, 0.9],  # High values for all properties
        [0.9, 0.1, 0.5]   # Mixed values
    ]
    generated_smiles = conditional_model.generate(labels=target_properties)
    print(f"Multi-conditionally generated {len(generated_smiles)} molecules.")
    assert len(generated_smiles) == len(target_properties)
    print("Example SMILES:", generated_smiles[:2])
    print("Target properties for first molecule:", target_properties[0])

    print("\n--- Testing model saving and loading ---")
    save_path = "conditional_defog_test_model.pt"
    conditional_model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_conditional_model = DeFoGMolecularGenerator()
    new_conditional_model.load_from_local(save_path)
    print("Model loaded successfully.")
    
    loaded_generated_smiles = new_conditional_model.generate(labels=target_properties)
    print(f"Generated {len(loaded_generated_smiles)} molecules with loaded model.")
    assert len(loaded_generated_smiles) > 0

    print("\n--- Testing generation with specified node counts ---")
    num_nodes = np.array([[20], [25], [30], [35]])
    generated_smiles_nodes = conditional_model.generate(labels=target_properties, num_nodes=num_nodes)
    print(f"Generated {len(generated_smiles_nodes)} molecules with specified node counts.")
    assert len(generated_smiles_nodes) == len(num_nodes)
    
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

    # 2. Single-property conditional testing (backwards compatibility)
    print("\n=== Testing Single-Property Conditional DeFoG Model ===")
    single_properties = properties[:, 0:1]  # Use only first property
    single_conditional_model = DeFoGMolecularGenerator(
        task_type=['regression'],  # Single regression task
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=5e-4,
        sample_steps=10,
        guidance_weight=0.2,
        verbose=True,
    )
    print("Single-Property Conditional DeFoG Model initialized successfully.")
    
    single_conditional_model.fit(smiles_list, single_properties)
    print("Single-Property DeFoG Model fitting completed.")
    
    single_target_properties = [[0.2], [0.5], [0.8], [0.1]]
    single_generated_smiles = single_conditional_model.generate(labels=single_target_properties)
    print(f"Single-conditionally generated {len(single_generated_smiles)} molecules.")
    assert len(single_generated_smiles) == len(single_target_properties)

    # 3. Unconditional Model Testing
    print("\n=== Testing Unconditional DeFoG Model ===")
    unconditional_model = DeFoGMolecularGenerator(
        task_type=[],  # Empty task_type for unconditional generation
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=5e-4,
        sample_steps=10,  # Fewer steps for faster testing
        guidance_weight=0.0,  # No guidance for unconditional generation
        verbose=True,
    )
    print("Unconditional DeFoG Model initialized successfully.")
    print(f"Input dim y: {unconditional_model.input_dim_y}")

    print("\n--- Fitting unconditional model ---")
    # For unconditional model, we don't need properties
    unconditional_model.fit(smiles_list)
    print("Unconditional DeFoG Model fitting completed.")

    print("\n--- Testing unconditional generation (no parameters) ---")
    unconditional_smiles = unconditional_model.generate()
    print(f"Unconditionally generated {len(unconditional_smiles)} molecules (no parameters).")
    assert len(unconditional_smiles) > 0
    print("Example SMILES:", unconditional_smiles[:2])

    print("\n--- Testing unconditional generation with batch_size ---")
    unconditional_smiles_batch = unconditional_model.generate(batch_size=5)
    print(f"Unconditionally generated {len(unconditional_smiles_batch)} molecules (batch_size=5).")
    assert len(unconditional_smiles_batch) == 5
    print("Example SMILES:", unconditional_smiles_batch[:2])

    print("\n--- Testing unconditional generation with specified node counts ---")
    num_nodes_uncond = np.array([[15], [20], [25]])
    unconditional_smiles_nodes = unconditional_model.generate(num_nodes=num_nodes_uncond)
    print(f"Unconditionally generated {len(unconditional_smiles_nodes)} molecules with specified node counts.")
    assert len(unconditional_smiles_nodes) == len(num_nodes_uncond)

    print("\n--- Testing unconditional model saving and loading ---")
    save_path_uncond = "unconditional_defog_test_model.pt"
    unconditional_model.save_to_local(save_path_uncond)
    print(f"Unconditional model saved to {save_path_uncond}")

    new_unconditional_model = DeFoGMolecularGenerator()
    new_unconditional_model.load_from_local(save_path_uncond)
    print("Unconditional model loaded successfully.")
    
    loaded_unconditional_smiles = new_unconditional_model.generate(batch_size=3)
    print(f"Generated {len(loaded_unconditional_smiles)} molecules with loaded unconditional model.")
    assert len(loaded_unconditional_smiles) == 3
    
    if os.path.exists(save_path_uncond):
        os.remove(save_path_uncond)
        print(f"Cleaned up {save_path_uncond}")

    print("\n=== All DeFoG tests completed successfully! ===")

if __name__ == "__main__":
    test_defog_generator()
