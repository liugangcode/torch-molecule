import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_molecule.generator.jtvae import JTVAEMolecularGenerator

EPOCHS = 3  # Reduced for faster testing
BATCH_SIZE = 8

def test_jtvae_generator():
    # Load data from polymer100.csv
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            "data", "polymer100.csv")
    print(f"Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    smiles_list = df['smiles'].tolist()
    
    print(f"Loaded {len(smiles_list)} molecules")
    print(f"First 3 SMILES: {smiles_list[:3]}")
    
    # 1. Basic initialization test
    print("\n=== Testing JTVAE model initialization ===")
    jtvae_model = JTVAEMolecularGenerator(
        hidden_size=300,  # Reduced for faster testing
        latent_size=32,   # Reduced for faster testing
        depthT=10,        # Reduced for faster testing
        depthG=2,         # Reduced for faster testing
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=True
    )
    print("JTVAE Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing JTVAE model fitting ===")
    jtvae_model.fit(smiles_list[:50])  # Using a subset for faster testing
    print("JTVAE Model fitting completed")

    # 3. Generation test
    print("\n=== Testing JTVAE generation ===")
    generated_smiles = jtvae_model.generate(batch_size=5)
    print(f"Generated {len(generated_smiles)} molecules")
    print("Example generated SMILES:", generated_smiles)
    
    # 4. Model saving and loading test
    print("\n=== Testing JTVAE model saving and loading ===")
    save_path = "jtvae_test_model.pt"
    jtvae_model.save_to_local(save_path)
    print(f"JTVAE Model saved to {save_path}")

    new_jtvae_model = JTVAEMolecularGenerator()
    new_jtvae_model.load_from_local(save_path)
    print("JTVAE Model loaded successfully")

    # Test generation with loaded model
    generated_smiles = new_jtvae_model.generate(batch_size=5)
    print("Generated molecules with loaded model:", len(generated_smiles))
    print("Example generated SMILES:", generated_smiles)

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_jtvae_generator()