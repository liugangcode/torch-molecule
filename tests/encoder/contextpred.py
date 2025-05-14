import numpy as np
from torch_molecule import ContextPredMolecularEncoder

def test_contextpred_encoder():
    # Test molecules (simple examples)
    molecules = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]

    # Basic initilization test
    model = ContextPredMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,  # Small number for testing
        mode='cbow', # 'cbow' or 'skipgram'
        context_size=2,
        neg_samples=1,
        verbose=True
    )
    print("Model initialized successfully")
    
    # Basic self-supervised fitting test
    print("\n=== Testing ContextPred model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    # Model saving and loading test
    print("\n=== Testing model saving and loading ===")
    save_path = "test_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = ContextPredMolecularEncoder()
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_contextpred_encoder()