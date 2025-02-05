import numpy as np
from torch_molecule import SupervisedMolecularEncoder

def test_supervised_encoder():
    # Test molecules (simple examples)
    molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]
    
    # Test case 1: Using only predefined tasks
    print("\nTest 1: Using predefined tasks (morgan and maccs, logp)")
    encoder1 = SupervisedMolecularEncoder(
        predefined_task=["morgan", "maccs", "logP" ],
        epochs=2,  # Small number for testing
        verbose=True
    )
    encoder1.fit(molecules, y_train=None)
    encodings1 = encoder1.encode(molecules)
    print(f"Encoding shape: {encodings1.shape}")
    
    # Test case 2: Using custom tasks
    print("\nTest 2: Using custom tasks")
    # Generate some dummy regression and classification targets
    n_samples = len(molecules)
    y_custom = np.zeros((n_samples, 2))
    # First column: regression task (random values between 0 and 1)
    y_custom[:, 0] = np.random.rand(n_samples)
    # Second column: binary classification task (0 or 1)
    y_custom[:, 1] = np.random.randint(0, 2, n_samples)
    
    encoder2 = SupervisedMolecularEncoder(
        num_task=2,  # Two custom tasks
        epochs=2,    # Small number for testing
        verbose=True
    )
    encoder2.fit(molecules, y_train=y_custom)
    encodings2 = encoder2.encode(molecules)
    print(f"Encoding shape: {encodings2.shape}")
    
    # Test case 3: Combining predefined and custom tasks
    print("\nTest 3: Combining predefined and custom tasks")
    encoder3 = SupervisedMolecularEncoder(
        predefined_task=["morgan"],
        num_task=2,  # 1024 (morgan) + 2 (custom)
        epochs=2,    # Small number for testing
        verbose=True
    )
    encoder3.fit(molecules, y_train=y_custom)
    encodings3 = encoder3.encode(molecules)
    print(f"Encoding shape: {encodings3.shape}")

if __name__ == "__main__":
    test_supervised_encoder()