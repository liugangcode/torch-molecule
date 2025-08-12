import numpy as np
from torch_molecule import BFGNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

def test_bfgnn_predictor():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'  # Additional molecule
    ]
    properties = np.array([0, 0, 1, 1])  # Binary classification
    print('smiles_list', len(smiles_list))
    print('properties', len(properties))
    # 1. Basic initialization test
    print("\n=== Testing model initialization ===")
    model = BFGNNMolecularPredictor(
        num_task=1,
        task_type="classification",
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=100,  # Small number for testing
        patience=100,
        verbose=True,
        l1_penalty=1e-3
    )
    print("Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing model fitting ===")
    model.fit(smiles_list, properties)
    print("Model fitting completed")

    # 3. Prediction test
    print("\n=== Testing model prediction ===")
    predictions = model.predict(smiles_list)
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new molecule: {predictions['prediction']}")

    # 4. Auto-fitting test with custom parameters
    print("\n=== Testing model auto-fitting ===")
    search_parameters = {
        'num_layer': (2, 4),
        'hidden_size': (64, 256),
        'learning_rate': (1e-4, 1e-2),
        'l1_penalty': (1e-6, 1e-3)
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
        )
    }
    model_auto = BFGNNMolecularPredictor(
        num_task=1,
        task_type="classification",
        epochs=3,  # Small number for testing
        # verbose=True
        verbose=False
    )
    
    model_auto.autofit(
        smiles_list,
        properties,
        search_parameters=search_parameters,
        n_trials=2  # Small number for testing
    )
    print("Model auto-fitting completed")

    # 5. Model saving and loading test
    print("\n=== Testing model saving and loading ===")
    save_path = "test_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = BFGNNMolecularPredictor(
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
    test_bfgnn_predictor()


