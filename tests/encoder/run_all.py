import numpy as np
import os
from torch_molecule import (
    AttrMaskMolecularEncoder,
    ContextPredMolecularEncoder,
    EdgePredMolecularEncoder,
    MoamaMolecularEncoder,
    SupervisedMolecularEncoder
)

def test_attrmask_encoder():
    # Test molecules (simple examples)
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]

    # Basic initialization test
    model = AttrMaskMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,
        verbose=True
    )
    print("AttrMask model initialized successfully")
    
    # Basic self-supervised fitting test
    print("\n=== Testing AttrMask model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    # Test save/load
    _test_save_load(model, AttrMaskMolecularEncoder, "attrmask_test.pt")

def test_contextpred_encoder():
    molecules = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]

    model = ContextPredMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,
        mode='cbow',
        context_size=2,
        neg_samples=1,
        verbose=True
    )
    print("\nContextPred model initialized successfully")
    
    print("\n=== Testing ContextPred model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    _test_save_load(model, ContextPredMolecularEncoder, "contextpred_test.pt")

def test_edgepred_encoder():
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]

    model = EdgePredMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,
        verbose=True
    )
    print("\nEdgePred model initialized successfully")
    
    print("\n=== Testing EdgePred model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    _test_save_load(model, EdgePredMolecularEncoder, "edgepred_test.pt")

def test_moama_encoder():
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]

    model = MoamaMolecularEncoder(
        num_layer=3,
        hidden_size=300,
        batch_size=5,
        epochs=5,
        verbose=True
    )
    print("\nMoAMa model initialized successfully")
    
    print("\n=== Testing MoAMa model self-supervised fitting ===")
    model.fit(molecules[:4])
    
    _test_save_load(model, MoamaMolecularEncoder, "moama_test.pt")

def test_supervised_encoder():
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]
    
    print("\nTest 1: Using predefined tasks (morgan and maccs, logp)")
    encoder1 = SupervisedMolecularEncoder(
        predefined_task=["morgan", "maccs", "logP"],
        epochs=2,
        verbose=True
    )
    encoder1.fit(molecules, y_train=None)
    encodings1 = encoder1.encode(molecules)
    print(f"Encoding shape: {encodings1.shape}")
    
    print("\nTest 2: Using custom tasks")
    n_samples = len(molecules)
    y_custom = np.zeros((n_samples, 2))
    y_custom[:, 0] = np.random.rand(n_samples)
    y_custom[:, 1] = np.random.randint(0, 2, n_samples)
    
    encoder2 = SupervisedMolecularEncoder(
        num_task=2,
        epochs=2,
        verbose=True
    )
    encoder2.fit(molecules, y_train=y_custom)
    encodings2 = encoder2.encode(molecules)
    print(f"Encoding shape: {encodings2.shape}")
    
    print("\nTest 3: Combining predefined and custom tasks")
    encoder3 = SupervisedMolecularEncoder(
        predefined_task=["morgan"],
        num_task=2,
        epochs=2,
        verbose=True
    )
    encoder3.fit(molecules, y_train=y_custom)
    encodings3 = encoder3.encode(molecules)
    print(f"Encoding shape: {encodings3.shape}")

def _test_save_load(model, model_class, save_path):
    """Helper function to test model saving and loading"""
    print("\n=== Testing model saving and loading ===")
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = model_class()
    new_model.load_from_local(save_path)
    print("Model loaded successfully")
    
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

def run_all_tests():
    """Run all encoder tests"""
    print("=== Starting All Encoder Tests ===\n")
    
    test_attrmask_encoder()
    test_contextpred_encoder()
    test_edgepred_encoder()
    test_moama_encoder()
    test_supervised_encoder()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    run_all_tests() 