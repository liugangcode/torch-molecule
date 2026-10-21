import numpy as np
from torch_molecule import AttrMaskMolecularEncoder

def test_attrmask_encoder():
    # Test molecules (simple examples)
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]

    # Basic initilization test
    model = AttrMaskMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,  # Small number for testing
        verbose='print_statement'
    )
    print("Model initialized successfully")
    
    # Basic self-supervised fitting test
    print("\n=== Testing AttrMask model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    # Model saving and loading test
    print("\n=== Testing model saving and loading ===")
    save_path = "test_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = AttrMaskMolecularEncoder()
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

def test_attrmask_encoder_polymers():
    # Test molecules (simple examples)
    polymers = [
        "*Nc1ccc([C@H](CCC)c2ccc(C3(c4ccc([C@@H](CCC)c5ccc(N*)cc5)cc4)CCC(CCCCC)CC3)cc2)cc1",
        "*Nc1ccc(-c2c(-c3ccc(C)cc3)c(-c3ccc(C)cc3)c(N*)c(-c3ccc(C)cc3)c2-c2ccc(C)cc2)cc1",
        "*CC(*)(C)C(=O)OCCCCCCCCCOc1ccc2cc(C(=O)Oc3ccccc3)ccc2c1"
    ]
    model = AttrMaskMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,  # Small number for testing
        verbose='progress_bar'
    )
    model.fit(polymers)
    vectors = model.encode(polymers)
    print(f"Representation shape: {vectors.shape}")
    print(f"Representation for new molecule: {vectors[0]}")


if __name__ == "__main__":
    test_attrmask_encoder_polymers()
    test_attrmask_encoder()
