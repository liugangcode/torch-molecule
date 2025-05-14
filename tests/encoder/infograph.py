import numpy as np
from torch_molecule import InfoGraphMolecularEncoder

def test_infograph_encoder():
    # Test molecules (simple examples)
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ] * 10

    # Basic initialization test
    model = InfoGraphMolecularEncoder(
        num_layer=3,
        embedding_dim=300,  # Must be divisible by num_layer
        batch_size=5,
        epochs=5,  # Small number for testing
        verbose=True,
        lw_prior=0.1
    )
    print("Model initialized successfully")
    
    # Basic self-supervised fitting test
    print("\n=== Testing Infograph model self-supervised fitting ===")
    model.fit(molecules)
    
    # Encoding test
    print("\n=== Testing molecule encoding ===")
    encodings = model.encode(molecules)
    print(f"Encoded {len(molecules)} molecules to shape {encodings.shape}")
    
    # Model saving and loading test
    print("\n=== Testing model saving and loading ===")
    save_path = "test_infograph_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = InfoGraphMolecularEncoder()
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    # Test encoding with loaded model
    new_encodings = new_model.encode(molecules)
    print(f"Encoded {len(molecules)} molecules with loaded model")
    
    # Verify encodings are the same
    if np.allclose(encodings.numpy(), new_encodings.numpy()):
        print("Encodings from original and loaded models match!")
    else:
        print("Warning: Encodings from original and loaded models differ")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_infograph_encoder()