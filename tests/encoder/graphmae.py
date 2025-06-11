import numpy as np
import csv
import os
from torch_molecule import GraphMAEMolecularEncoder

EPOCHS = 5

def test_graphmae_encoder():
    # Load molecules from CSV file
    data_path = "data/molecule100.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        # Use simple molecules as fallback
        molecules = [
            "CC(=O)O",  # Acetic acid
            "CCO",      # Ethanol
            "CCCC",     # Butane
            "c1ccccc1", # Benzene
            "CCN",      # Ethylamine
        ]
    else:
        molecules = []
        with open(data_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for i, row in enumerate(csv_reader):
                if i >= 50:  # Use first 50 molecules
                    break
                molecules.append(row['smiles'])
        print(f"Loaded {len(molecules)} molecules from {data_path}")
    # Initialize GraphMAE model
    model = GraphMAEMolecularEncoder(
        num_layer=3,
        hidden_size=128,
        batch_size=16,
        epochs=EPOCHS,  # Small number for testing
        mask_rate=0.15,
        verbose=True,
        # device="cpu"
    )
    print("GraphMAE model initialized successfully")
    
    # Test fitting
    print("\n=== Testing GraphMAE model self-supervised fitting ===")
    model.fit(molecules)
    
    # Test encoding
    print("\n=== Testing molecule encoding ===")
    encodings = model.encode(molecules[:5])
    print(f"Encoding shape: {encodings.shape}")
    
    # Test saving and loading
    print("\n=== Testing model saving and loading ===")
    save_path = "graphmae_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")
    
    new_model = GraphMAEMolecularEncoder()
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    # Test encoding with loaded model
    new_encodings = new_model.encode(molecules[:5])
    print(f"New encoding shape: {new_encodings.shape}")
    
    # Verify encodings are the same (or very close)
    encoding_diff = (encodings - new_encodings).abs().max().item()
    print(f"Max difference between encodings: {encoding_diff}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

def test_graphmae_with_edge_masking():
    # Load molecules from CSV file
    data_path = "data/molecule100.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        # Use simple molecules as fallback
        molecules = [
            "CC(=O)O",  # Acetic acid
            "CCO",      # Ethanol
            "CCCC",     # Butane
            "c1ccccc1", # Benzene
            "CCN",      # Ethylamine
        ]
    else:
        molecules = []
        with open(data_path, 'r') as file:
            csv_reader = csv.DictReader(file)
            for i, row in enumerate(csv_reader):
                if i >= 50:  # Use first 50 molecules
                    break
                molecules.append(row['smiles'])
        print(f"Loaded {len(molecules)} molecules from {data_path}")
    
    # Initialize GraphMAE model with edge masking enabled
    model = GraphMAEMolecularEncoder(
        num_layer=3,
        hidden_size=128,
        batch_size=16,
        epochs=EPOCHS,  # Small number for testing
        mask_rate=0.15,
        mask_edge=True,  # Enable edge masking
        verbose=True,
        # device="cpu"
    )
    print("GraphMAE model with edge masking initialized successfully")
    
    # Test fitting
    print("\n=== Testing GraphMAE model with edge masking ===")
    model.fit(molecules)
    
    # Test encoding
    print("\n=== Testing molecule encoding with edge masking model ===")
    encodings = model.encode(molecules[:5])
    print(f"Encoding shape: {encodings.shape}")
    
    # Test saving and loading
    print("\n=== Testing edge masking model saving and loading ===")
    save_path = "graphmae_edge_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")
    
    new_model = GraphMAEMolecularEncoder()
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    # Verify edge masking parameter was preserved
    print(f"Loaded model mask_edge parameter: {new_model.mask_edge}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    print("=== Testing GraphMAE Encoder (Default Configuration) ===")
    test_graphmae_encoder()
    
    print("\n=== Testing GraphMAE Encoder with Edge Masking ===")
    test_graphmae_with_edge_masking()