import os
import numpy as np
import warnings
from tqdm import tqdm

import torch
from torch_molecule import IRMMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

def test_irm_gnn_predictor():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'  # Additional molecule
    ]
    smiles_list = smiles_list * 100
    properties = np.array([0, 0, 1, 1] * 100)  # Binary classification
    print('smiles_list', len(smiles_list))
    print('properties', len(properties))

    # Add new test for manual environment assignment
    print("\n=== Testing manual environment assignment ===")
    environments = [0, 1, 0, 1] * 100  # Alternating environments
    model_manual_env = IRMMolecularPredictor(
        num_task=1,
        task_type="classification",
        num_layer=3,
        hidden_size=128,
        batch_size=10,
        epochs=2,
        verbose=True,
        IRM_environment=environments
    )
    model_manual_env.fit(smiles_list, properties)
    print("Manual environment assignment test passed")

    # Test invalid environment assignments
    print("\n=== Testing invalid environment assignments ===")
    try:
        invalid_environments = [0, 1, 0]  # Wrong length
        model_invalid_env = IRMMolecularPredictor(
            num_task=1,
            task_type="classification",
            IRM_environment=invalid_environments
        )
        model_invalid_env.fit(smiles_list, properties)
        raise AssertionError("Should have raised ValueError for wrong length environments")
    except ValueError as e:
        print("Correctly caught wrong length environment error")

    try:
        invalid_environments = ["a", "b", "c", "d"] * 100  # Wrong type
        model_invalid_env = IRMMolecularPredictor(
            num_task=1,
            task_type="classification",
            IRM_environment=invalid_environments
        )
        model_invalid_env.fit(smiles_list, properties)
        raise AssertionError("Should have raised ValueError for wrong type environments")
    except ValueError as e:
        print("Correctly caught wrong type environment error")

    # Test set_IRM_environment function
    print("\n=== Testing set_IRM_environment function ===")
    
    # Test setting environment after initialization
    model_env_setter = IRMMolecularPredictor(
        num_task=1,
        task_type="classification",
        num_layer=3,
        hidden_size=128
    )
    
    # Test valid environment assignments
    valid_environments = [0, 1, 0, 1] * 100
    model_env_setter.set_IRM_environment(valid_environments)
    print("Successfully set valid environment list")
    
    model_env_setter.set_IRM_environment("random")
    print("Successfully set random environment")
    
    # Test numpy array
    np_environments = np.array([0, 1, 0, 1] * 100)
    model_env_setter.set_IRM_environment(np_environments)
    print("Successfully set numpy array environment")
    
    # Test torch tensor
    torch_environments = torch.tensor([0, 1, 0, 1] * 100)
    model_env_setter.set_IRM_environment(torch_environments)
    print("Successfully set torch tensor environment")
    
    # Test invalid environment assignments
    try:
        model_env_setter.set_IRM_environment("invalid_string")
        raise AssertionError("Should have raised ValueError for invalid string")
    except ValueError as e:
        print("Correctly caught invalid string environment error")
    
    try:
        model_env_setter.set_IRM_environment([1.5, 2.5, 3.5, 4.5] * 100)  # Float values
        raise AssertionError("Should have raised ValueError for non-integer values")
    except ValueError as e:
        print("Correctly caught non-integer environment error")

    # 1. Basic initialization test
    print("\n=== Testing IRM model initialization ===")
    model = IRMMolecularPredictor(
        num_task=1,
        task_type="classification",
        num_layer=3,
        hidden_size=128,
        batch_size=10,
        patience=1000,
        epochs=10,  # Small number for testing
        verbose=True,
        IRM_environment="random",  # IRM-specific
        penalty_weight=100,        # IRM-specific
        penalty_anneal_iters=2    # IRM-specific
    )
    print("IRM Model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing IRM model fitting ===")
    model.fit(smiles_list, properties)
    print("IRM Model fitting completed")

    # 3. Prediction test
    print("\n=== Testing IRM model prediction ===")
    predictions = model.predict(smiles_list[3:])
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new molecule: {predictions['prediction']}")

    # 4. Auto-fitting test with custom parameters
    print("\n=== Testing IRM model auto-fitting ===")
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
        'penalty_weight': ParameterSpec(  # IRM-specific
            param_type=ParameterType.LOG_FLOAT,
            value_range=(1e-10, 1)
        ),
        'penalty_anneal_iters': ParameterSpec(  # IRM-specific
            param_type=ParameterType.INTEGER,
            value_range=(10, 100)
        ),
    }
    model_auto = IRMMolecularPredictor(
        num_task=1,
        task_type="classification",
        epochs=3,  # Small number for testing
        verbose=False
    )
    
    model_auto.autofit(
        smiles_list,
        properties,
        search_parameters=search_parameters,
        n_trials=2  # Small number for testing
    )
    print("IRM Model auto-fitting completed")

    # 5. Model saving and loading test
    print("\n=== Testing IRM model saving and loading ===")
    save_path = "irm_test_model.pt"
    model.save_to_local(save_path)
    print(f"IRM Model saved to {save_path}")

    new_model = IRMMolecularPredictor(
        num_task=1,
        task_type="classification"
    )
    new_model.load_from_local(save_path)
    print("IRM Model loaded successfully")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_irm_gnn_predictor()
