import numpy as np
from torch_molecule import GRINMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

def test_grin_predictor():
    # Test data
    smiles_list = [
        'FC(F)(F)C(C1=CC2=C(C=C1)C(=O)N(C2=O)C1=CC=C(CC2=CC=C(*)C=C2)C=C1)(C1=CC=C2C(=O)N(*)C(=O)C2=C1)C(F)(F)F',
        'CCCCCCCCCCCCC1=C(*)SC(*)=C1',
        'CC(C)C1=C(O*)C=CC(=C1)C(C1C=CC=CC1C(O)=O)C1=CC(C(C)C)=C(OC2=CC=C(C=C2)C(=O)C2=CC=C(*)C=C2)C=C1',
        '*OC1=CC=C(C=C1)C1(OC(=O)C2=C1C=CC=C2)C1=CC=C(OC2=CC=C(C=C2)S(=O)(=O)C2=CC=C(*)C=C2)C=C1' 
    ]
    properties = np.array([19, 88.2, 22.8, 5.74])  # Regression
    print('smiles_list', len(smiles_list))
    print('properties', len(properties))
    # 1. Basic initialization test
    print("\n=== Testing model initialization ===")
    model = GRINMolecularPredictor(
        num_task=1,
        task_type="regression",
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=5,  # Small number for testing
        verbose="progress_bar",
    )
    print("Model initialized successfully")

    # 1.2. Basic fitting test
    print("\n=== Testing model fitting ===")
    model.fit(smiles_list, properties)
    print("Model fitting completed")

    # 1.3. Prediction test
    print("\n=== Testing model prediction ===")
    predictions = model.predict(smiles_list)
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new polymer repeat times 1: {predictions['prediction']}")

    # 2. Initialize with polymer train augmentation
    print("\n=== Testing model initialization with polymer train augmentation ===")
    model = GRINMolecularPredictor(
        num_task=1,
        task_type="regression",
        polymer_train_augmentation=3,
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=5,  # Small number for testing
        verbose="progress_bar",
    )
    print("Model initialized with polymer train augmentation successfully")

    # 2.1. Fitting test
    print("\n=== Testing model fitting with polymer train augmentation ===")
    model.fit(smiles_list, properties)
    print("Model fitting completed")

    # 2.2. Prediction test on different repeat times
    print("\n=== Testing model prediction with polymer train augmentation ===")
    predictions = model.predict(smiles_list, test_augmentation=3)
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new polymer repeat times 3: {predictions['prediction']}")
    predictions = model.predict(smiles_list, test_augmentation=5)
    print(f"Prediction for new polymer repeat times 5: {predictions['prediction']}")

    # 3. Auto-fitting test with custom parameters
    print("\n=== Testing model auto-fitting ===")
    search_parameters = {
        'num_layer': (2, 4),
        'hidden_size': (64, 256),
        'learning_rate': (1e-4, 1e-2),
        'l1_penalty': (1e-6, 1e-3),
        'epochs_to_penalize': (0, 100)
    }
    search_parameters = {
        'num_layer': ParameterSpec(
            param_type=ParameterType.INTEGER,
            value_range=(2, 4)
        ),
        'hidden_size': ParameterSpec(
            param_type=ParameterType.INTEGER,
            value_range=(64, 256)
        ),
        'learning_rate': ParameterSpec(
            param_type=ParameterType.LOG_FLOAT,
            value_range=(1e-4, 1e-2)
        ),
        'l1_penalty': ParameterSpec(
            param_type=ParameterType.LOG_FLOAT,
            value_range=(1e-6, 1e-3)
        ),
        'epochs_to_penalize': ParameterSpec(
            param_type=ParameterType.INTEGER,
            value_range=(0, 100)
        )
    }
    model_auto = GRINMolecularPredictor(
        num_task=1,
        task_type="regression",
        epochs=3,  # Small number for testing
        verbose="none"
    )
    
    model_auto.autofit(
        smiles_list,
        properties,
        search_parameters=search_parameters,
        n_trials=2  # Small number for testing
    )
    print("Model auto-fitting completed")

    # 4. Model saving and loading test
    print("\n=== Testing model saving and loading ===")
    save_path = "test_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = GRINMolecularPredictor(
        num_task=1,
        task_type="regression"
    )
    new_model.load_from_local(save_path)
    print("Model loaded successfully")

    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_grin_predictor()


