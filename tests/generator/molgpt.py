import os
import re
import numpy as np
from tqdm import tqdm

import torch
from torch_molecule import MolGPTMolecularGenerator

# EPOCHS = 1000
EPOCHS = 5
BATCH_SIZE = 32

def test_molgpt_generator():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    smiles_list = smiles_list * 25  # Create 100 molecules for training
    properties = [[1.0], [2.0], [3.0], [4.0]] * 25  # Create 100 properties for training
    
    # Create scaffold examples
    scaffold_list = [
        'c1ccccc1',  # Benzene
        'c1ccncc1',  # Pyridine
        'c1cccnc1',  # Pyridine (different position)
        'c1cnccn1'   # Pyrazine
    ] * 25
    
    # 1. Basic initialization test - Unconditional Model
    print('smiles_list', len(smiles_list), smiles_list[:5])
    print("\n=== Testing Unconditional MolGPT model initialization ===")
    unconditional_model = MolGPTMolecularGenerator(
        num_layer=4,
        num_head=4,
        hidden_size=128,
        max_len=64,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose="progress_bar"
    )
    print("Unconditional MolGPT Model initialized successfully")

    # 2. Basic fitting test - Unconditional Model
    print("\n=== Testing Unconditional MolGPT model fitting ===")
    unconditional_model.fit(smiles_list)
    print("Unconditional MolGPT Model fitting completed")

    # 3. Unconditional generation test
    print("\n=== Testing Unconditional MolGPT generation ===")
    generated_smiles_uncond = unconditional_model.generate(n_samples=BATCH_SIZE)
    print(f"Unconditionally generated {len(generated_smiles_uncond)} molecules")
    print("Example unconditionally generated SMILES:", generated_smiles_uncond[:2])
    
    # 4. Model saving and loading test - Unconditional Model
    print("\n=== Testing Unconditional MolGPT model saving and loading ===")
    save_path = "unconditional_molgpt_test_model.pt"
    unconditional_model.save_to_local(save_path)
    print(f"Unconditional MolGPT Model saved to {save_path}")

    new_unconditional_model = MolGPTMolecularGenerator()
    new_unconditional_model.load_from_local(save_path)
    print("Unconditional MolGPT Model loaded successfully")

    # Test generation with loaded unconditional model
    generated_smiles_uncond = new_unconditional_model.generate(n_samples=5)
    print("Generated molecules with loaded unconditional model:", len(generated_smiles_uncond))
    print("Example generated SMILES:", generated_smiles_uncond[:2])

    # Clean up unconditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")
    
    # 5. Basic initialization test - Property Conditional Model
    print("\n=== Testing Property Conditional MolGPT model initialization ===")
    prop_conditional_model = MolGPTMolecularGenerator(
        num_layer=4,
        num_head=4,
        hidden_size=128,
        max_len=64,
        num_task=1,  # Enable property conditioning
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose="progress_bar"
    )
    print("Property Conditional MolGPT Model initialized successfully")

    # 6. Basic fitting test - Property Conditional Model
    print("\n=== Testing Property Conditional MolGPT model fitting ===")
    prop_conditional_model.fit(smiles_list, properties)
    print("Property Conditional MolGPT Model fitting completed")

    # 7. Property conditional generation test
    print("\n=== Testing Property Conditional MolGPT generation ===")
    target_properties = [[1.5], [2.5], [3.5], [4.5], [5.0]]
    generated_smiles = prop_conditional_model.generate(n_samples=5, properties=target_properties)
    print(f"Property conditionally generated {len(generated_smiles)} molecules")
    print("Example property conditionally generated SMILES:", generated_smiles[:2])
    
    # 8. Model saving and loading test - Property Conditional Model
    print("\n=== Testing Property Conditional MolGPT model saving and loading ===")
    save_path = "prop_conditional_molgpt_test_model.pt"
    prop_conditional_model.save_to_local(save_path)
    print(f"Property Conditional MolGPT Model saved to {save_path}")

    new_prop_conditional_model = MolGPTMolecularGenerator()
    new_prop_conditional_model.load_from_local(save_path)
    print("Property Conditional MolGPT Model loaded successfully")

    # Test generation with loaded property conditional model
    generated_smiles = new_prop_conditional_model.generate(n_samples=5, properties=target_properties)
    print("Generated molecules with loaded property conditional model:", len(generated_smiles))
    print("Example generated SMILES:", generated_smiles[:2])

    # Clean up property conditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")
    
    # 9. Basic initialization test - Scaffold Conditional Model
    print("\n=== Testing Scaffold Conditional MolGPT model initialization ===")
    scaffold_conditional_model = MolGPTMolecularGenerator(
        num_layer=4,
        num_head=4,
        hidden_size=128,
        max_len=64,
        use_scaffold=True,  # Enable scaffold conditioning
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose="progress_bar"
    )
    print("Scaffold Conditional MolGPT Model initialized successfully")

    # 10. Basic fitting test - Scaffold Conditional Model
    print("\n=== Testing Scaffold Conditional MolGPT model fitting ===")
    scaffold_conditional_model.fit(smiles_list, X_scaffold=scaffold_list)
    print("Scaffold Conditional MolGPT Model fitting completed")

    # 11. Scaffold conditional generation test
    print("\n=== Testing Scaffold Conditional MolGPT generation ===")
    target_scaffolds = ['c1ccccc1', 'c1ccncc1']  # Benzene and pyridine
    generated_smiles = scaffold_conditional_model.generate(n_samples=2, scaffolds=target_scaffolds)
    print(f"Scaffold conditionally generated {len(generated_smiles)} molecules")
    print("Example scaffold conditionally generated SMILES:", generated_smiles[:2])
    
    # 12. Model saving and loading test - Scaffold Conditional Model
    print("\n=== Testing Scaffold Conditional MolGPT model saving and loading ===")
    save_path = "scaffold_conditional_molgpt_test_model.pt"
    scaffold_conditional_model.save_to_local(save_path)
    print(f"Scaffold Conditional MolGPT Model saved to {save_path}")

    new_scaffold_conditional_model = MolGPTMolecularGenerator()
    new_scaffold_conditional_model.load_from_local(save_path)
    print("Scaffold Conditional MolGPT Model loaded successfully")

    # Test generation with loaded scaffold conditional model
    generated_smiles = new_scaffold_conditional_model.generate(n_samples=2, scaffolds=target_scaffolds)
    print("Generated molecules with loaded scaffold conditional model:", len(generated_smiles))
    print("Example generated SMILES:", generated_smiles[:2])

    # Clean up scaffold conditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")
    
    # 13. Basic initialization test - Combined Conditional Model (Property + Scaffold)
    print("\n=== Testing Combined Conditional MolGPT model initialization ===")
    combined_conditional_model = MolGPTMolecularGenerator(
        num_layer=4,
        num_head=4,
        hidden_size=128,
        max_len=64,
        num_task=1,  # Enable property conditioning
        use_scaffold=True,  # Enable scaffold conditioning
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose="progress_bar"
    )
    print("Combined Conditional MolGPT Model initialized successfully")

    # 14. Basic fitting test - Combined Conditional Model
    print("\n=== Testing Combined Conditional MolGPT model fitting ===")
    combined_conditional_model.fit(smiles_list, properties, X_scaffold=scaffold_list)
    print("Combined Conditional MolGPT Model fitting completed")

    # 15. Combined conditional generation test
    print("\n=== Testing Combined Conditional MolGPT generation ===")
    target_properties = [[1.5], [2.5]]
    target_scaffolds = ['c1ccccc1', 'c1ccncc1']
    generated_smiles = combined_conditional_model.generate(
        n_samples=2, 
        properties=target_properties,
        scaffolds=target_scaffolds
    )
    print(f"Combined conditionally generated {len(generated_smiles)} molecules")
    print("Example combined conditionally generated SMILES:", generated_smiles[:2])
    
    # 16. Model saving and loading test - Combined Conditional Model
    print("\n=== Testing Combined Conditional MolGPT model saving and loading ===")
    save_path = "combined_conditional_molgpt_test_model.pt"
    combined_conditional_model.save_to_local(save_path)
    print(f"Combined Conditional MolGPT Model saved to {save_path}")

    new_combined_conditional_model = MolGPTMolecularGenerator()
    new_combined_conditional_model.load_from_local(save_path)
    print("Combined Conditional MolGPT Model loaded successfully")

    # Test generation with loaded combined conditional model
    generated_smiles = new_combined_conditional_model.generate(
        n_samples=2, 
        properties=target_properties,
        scaffolds=target_scaffolds
    )
    print("Generated molecules with loaded combined conditional model:", len(generated_smiles))
    print("Example generated SMILES:", generated_smiles[:2])

    # Clean up combined conditional model
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_molgpt_generator()