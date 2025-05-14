import numpy as np
from torch_molecule import SMILESTransformerMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

def test_transformer_predictor():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'  # Additional molecule
    ]
    properties = np.array([0, 0, 1, 1])  # Binary classification

    # 1. Basic initialization test
    print("\n=== Testing model initialization ===")
    model = SMILESTransformerMolecularPredictor(
        task_type="regression",
        hidden_size=128,
        n_heads=4,
        num_layers=2,
        batch_size=2,
        epochs=2,
        device="cpu",
        verbose=True
    )
    print("Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing model fitting ===")
    model.fit(smiles_list[:3], properties[:3])
    print("Model fitting completed")

    # 3. Prediction test
    print("\n=== Testing model prediction ===")
    predictions = model.predict(smiles_list[3:])
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new molecule: {predictions['prediction']}")

    # 4. Auto-fitting test with custom parameters
    print("\n=== Testing model auto-fitting ===")

    search_parameters = {
        "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 128)),
        "n_heads": ParameterSpec(ParameterType.INTEGER, (2, 4)),
        "num_layers": ParameterSpec(ParameterType.INTEGER, (1, 3)),
        # Float-valued parameters
        "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
        "dropout": ParameterSpec(ParameterType.FLOAT, (0.0, 0.3))
    }
    model_auto = SMILESTransformerMolecularPredictor(
        num_task=1,
        task_type="regression",
        epochs=3,  # Small number for testing
        verbose=True
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
    save_path = "test_transformer_model.pt"
    model.save_to_local(save_path)
    print(f"Model saved to {save_path}")

    new_model = SMILESTransformerMolecularPredictor(
        task_type="regression",
        hidden_size=128,
        n_heads=4,
        num_layers=2,
        batch_size=2,
        epochs=2,
        device="cpu"
    )
    new_model.load_from_local(save_path)
    print("Model loaded successfully")

    # 6. Test model prediction after loading
    print("\n=== Testing model prediction after loading ===")
    predictions = new_model.predict(smiles_list[3:])
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new molecule: {predictions['prediction']}")

    # 7. Compare model performance with original LSTM
    print("\n=== Comparing Transformer with LSTM ===")
    # This section would be expanded with actual comparison metrics in a real-world scenario
    print("Performance comparison would be implemented here")

    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_transformer_predictor()


