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
    
    # ===== SINGLE TASK CLASSIFICATION TEST =====
    print("\n" + "="*50)
    print("TESTING SINGLE TASK CLASSIFICATION")
    print("="*50)
    
    properties_classification = np.array([0, 0, 1, 1])  # Binary classification

    # 1. Basic initialization test
    print("\n=== Testing model initialization ===")
    model = SMILESTransformerMolecularPredictor(
        task_type="regression",
        hidden_size=128,
        n_heads=4,
        num_layers=2,
        batch_size=2,
        epochs=200,
        patience=200,
        device="cpu",
        verbose=True,
        use_lr_scheduler=True,
    )
    print("Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing model fitting ===")
    model.fit(smiles_list[:3], properties_classification[:3])
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
        properties_classification,
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

    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

    # ===== MULTITASK REGRESSION TEST =====
    print("\n" + "="*50)
    print("TESTING MULTITASK REGRESSION (5 TASKS)")
    print("="*50)
    
    # Multitask regression with 5 tasks - each molecule has 5 property values
    properties_multitask = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Properties for molecule 1
        [0.2, 0.3, 0.4, 0.5, 0.6],  # Properties for molecule 2
        [0.8, 0.9, 1.0, 1.1, 1.2],  # Properties for molecule 3
        [0.9, 1.0, 1.1, 1.2, 1.3]   # Properties for molecule 4
    ])

    # 1. Multitask model initialization test
    print("\n=== Testing multitask model initialization ===")
    model_multitask = SMILESTransformerMolecularPredictor(
        num_task=5,  # 5 tasks for multitask regression
        task_type="regression",
        hidden_size=128,
        n_heads=4,
        num_layers=2,
        batch_size=2,
        epochs=2,
        device="cpu",
        verbose=True,
        use_lr_scheduler=True,
    )
    print("Multitask model initialized successfully")

    # 2. Multitask fitting test
    print("\n=== Testing multitask model fitting ===")
    model_multitask.fit(smiles_list[:3], properties_multitask[:3])
    print("Multitask model fitting completed")

    # 3. Multitask prediction test
    print("\n=== Testing multitask model prediction ===")
    predictions_multitask = model_multitask.predict(smiles_list[3:])
    print(f"Multitask prediction shape: {predictions_multitask['prediction'].shape}")
    print(f"Multitask prediction for new molecule: {predictions_multitask['prediction']}")

    # 4. Multitask auto-fitting test
    print("\n=== Testing multitask model auto-fitting ===")
    search_parameters_multitask = {
        "hidden_size": ParameterSpec(ParameterType.INTEGER, (64, 128)),
        "n_heads": ParameterSpec(ParameterType.INTEGER, (2, 4)),
        "num_layers": ParameterSpec(ParameterType.INTEGER, (1, 3)),
        # Float-valued parameters
        "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
        "dropout": ParameterSpec(ParameterType.FLOAT, (0.0, 0.3))
    }
    model_multitask_auto = SMILESTransformerMolecularPredictor(
        num_task=5,  # 5 tasks for multitask regression
        task_type="regression",
        epochs=3,  # Small number for testing
        verbose=True
    )
    
    model_multitask_auto.autofit(
        smiles_list,
        properties_multitask,
        search_parameters=search_parameters_multitask,
        n_trials=2  # Small number for testing
    )
    print("Multitask model auto-fitting completed")

    # 5. Multitask model saving and loading test
    print("\n=== Testing multitask model saving and loading ===")
    save_path_multitask = "test_transformer_multitask_model.pt"
    model_multitask.save_to_local(save_path_multitask)
    print(f"Multitask model saved to {save_path_multitask}")

    new_model_multitask = SMILESTransformerMolecularPredictor(
        num_task=5,  # 5 tasks for multitask regression
        task_type="regression",
        hidden_size=128,
        n_heads=4,
        num_layers=2,
        batch_size=2,
        epochs=2,
        device="cpu"
    )
    new_model_multitask.load_from_local(save_path_multitask)
    print("Multitask model loaded successfully")

    # 6. Test multitask model prediction after loading
    print("\n=== Testing multitask model prediction after loading ===")
    predictions_loaded = new_model_multitask.predict(smiles_list[3:])
    print(f"Loaded multitask prediction shape: {predictions_loaded['prediction'].shape}")
    print(f"Loaded multitask prediction for new molecule: {predictions_loaded['prediction']}")

    # Clean up multitask model
    if os.path.exists(save_path_multitask):
        os.remove(save_path_multitask)
        print(f"Cleaned up {save_path_multitask}")

    # 7. Compare single task vs multitask performance
    print("\n=== Comparing Single Task vs Multitask Performance ===")
    print("Single task prediction shape:", predictions['prediction'].shape)
    print("Multitask prediction shape:", predictions_multitask['prediction'].shape)
    print("Performance comparison would be implemented here with actual metrics")

if __name__ == "__main__":
    test_transformer_predictor()


