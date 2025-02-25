import numpy as np
from torch_molecule import MoamaMolecularEncoder
from torch_molecule.utils.search import ParameterType, ParameterSpec

def test_moama_encoder():
    # Test molecules (simple examples)
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]

    # Basic initilization test
    model = MoamaMolecularEncoder(
        num_task=119,
        task_type="classification",
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,  # Small number for testing
        verbose=True
    )
    print("Model initialized successfully")
    
    # Basic self-supervised fitting test
    print("\n=== Testing MoAMa model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    # Model saving and loading test
    print("\n=== Testing model saving and loading ===")
    save_path = "test_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = MoamaMolecularEncoder(
        num_task=1,
        task_type="classification"
    )
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_moama_encoder()