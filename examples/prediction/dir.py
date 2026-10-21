import numpy as np
from torch_molecule import DIRMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import os

def test_dir_predictor():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    properties = np.array([0, 0, 1, 1])  # Binary classification

    # 1. Basic initialization test
    print("\n=== Testing DIR model initialization ===")
    model = DIRMolecularPredictor(
        num_task=1,
        task_type="classification",
        causal_ratio=0.8,
        lw_invariant=1e-4,
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=5,
        verbose="progress_bar"
    )
    print("DIR model initialized successfully")

    # 2. Basic fitting test
    print("\n=== Testing DIR model fitting ===")
    model.fit(smiles_list[:3], properties[:3])
    print("DIR model fitting completed")

    # 3. Prediction test
    print("\n=== Testing DIR model prediction ===")
    predictions = model.predict(smiles_list[3:])
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new molecule: {predictions['prediction']}")

    # 4. Auto-fitting test with search space parameters
    print("\n=== Testing DIR model auto-fitting ===")
    
    # Define search parameters using ParameterSpec
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
        'drop_ratio': ParameterSpec(
            param_type=ParameterType.FLOAT,
            value_range=(0.1, 0.5)
        ),
        'causal_ratio': ParameterSpec(
            param_type=ParameterType.FLOAT,
            value_range=(0.2, 0.9)
        ),
        'lw_invariant': ParameterSpec(
            param_type=ParameterType.LOG_FLOAT,
            value_range=(1e-5, 1e-3)
        ),
        'gnn_type': ParameterSpec(
            param_type=ParameterType.CATEGORICAL,
            value_range=['gin-virtual', 'gcn']
        )
    }
    
    # Test different search configurations
    print("\n--- Testing with full search space ---")
    model_auto = DIRMolecularPredictor(
        num_task=1,
        task_type="classification",
        epochs=3,
        verbose="progress_bar"
    )
    
    model_auto.autofit(
        smiles_list,
        properties,
        search_parameters=search_parameters,
        n_trials=2
    )
    print("Full search space auto-fitting completed")
    
    # Test with subset of parameters
    print("\n--- Testing with partial search space ---")
    partial_search = {
        'num_layer': search_parameters['num_layer'],
        'learning_rate': search_parameters['learning_rate'],
        'causal_ratio': search_parameters['causal_ratio']
    }
    
    model_partial = DIRMolecularPredictor(
        num_task=1,
        task_type="classification",
        epochs=3,
        verbose="progress_bar"
    )
    
    model_partial.autofit(
        smiles_list,
        properties,
        search_parameters=partial_search,
        n_trials=2
    )
    print("Partial search space auto-fitting completed")
    
    # Test with default search space
    print("\n--- Testing with default search space ---")
    model_default = DIRMolecularPredictor(
        num_task=1,
        task_type="classification",
        epochs=3,
        verbose="progress_bar"
    )
    
    model_default.autofit(
        smiles_list,
        properties,
        n_trials=2
    )
    print("Default search space auto-fitting completed")

    # 5. Model saving and loading test
    print("\n=== Testing DIR model saving and loading ===")
    save_path = "test_dir_model.pt"
    model.save_to_local(save_path)
    print(f"DIR model saved to {save_path}")

    new_model = DIRMolecularPredictor(
        num_task=1,
        task_type="classification"
    )
    new_model.load_from_local(save_path)
    print("DIR model loaded successfully")

    # 6. Test invalid SMILES handling
    print("\n=== Testing invalid SMILES handling ===")
    try:
        invalid_smiles = ['CC', 'INVALID_SMILES']
        predictions = model.predict(invalid_smiles)
        print("Invalid SMILES handling test completed")
    except Exception as e:
        print(f"Invalid SMILES handled with error: {str(e)}")

    # 7. Test error handling for invalid search parameters
    print("\n=== Testing invalid search parameters handling ===")
    try:
        invalid_search = {
            'invalid_param': ParameterSpec(
                param_type=ParameterType.FLOAT,
                value_range=(0, 1)
            )
        }
        model_invalid = DIRMolecularPredictor(
            num_task=1,
            task_type="classification",
            verbose="progress_bar"
        )
        model_invalid.autofit(
            smiles_list,
            properties,
            search_parameters=invalid_search,
            n_trials=1
        )
    except ValueError as e:
        print(f"Invalid parameter handled correctly with error: {str(e)}")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

def test_dir_upload():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    properties = np.array([0, 0, 1, 1])  # Binary classification

    # 8. Test Hugging Face upload
    print("\n=== Testing Hugging Face Hub upload ===")
    # Initialize and fit a small model for testing upload
    model_for_upload = DIRMolecularPredictor(
        num_task=1,
        task_type="regression",
        causal_ratio=0.8,
        lw_invariant=1e-4,
        num_layer=2,  # Small model for quick testing
        hidden_size=64,   # Small embedding dimension
        batch_size=4,
        epochs=2,     # Few epochs for quick testing
        verbose="none",
        model_name='DIR_Invariant'
    )
    
    # Fit the model with sample data
    model_for_upload.autofit(smiles_list[:3], properties[:3])
    
    # Push to Hugging Face Hub
    # Note: HF_TOKEN should be set in environment variables
    repo_id = "liuganghuggingface/test-torch-molecule-ckpt-DIR-invariant"
    model_for_upload.save(
        repo_id=repo_id,
        commit_message="Upload DIR model for invariant rationale discovery",
        private=False
    )
    print("Successfully pushed model to Hugging Face Hub")
    
    # Test downloading and loading from Hub
    print("\n=== Testing model loading from Hugging Face Hub ===")        
    # Load model
    downloaded_model = DIRMolecularPredictor()
    downloaded_model.load_from_hf(repo_id=repo_id, path="./downloaded_model/DIR_Invariant.pt")
    
    # Test prediction with downloaded model
    test_pred = downloaded_model.predict(smiles_list[3:])
    print("Successfully loaded and tested model from Hugging Face Hub")
    print(f"Test prediction shape: {test_pred['prediction'].shape}")
    
    # Clean up downloaded files
    if os.path.exists("./downloaded_model"):
        import shutil
        shutil.rmtree("./downloaded_model")
        print("Cleaned up downloaded model files")

    # Previous cleanup code remains the same...
    if os.path.exists("test_dir_model.pt"):
        os.remove("test_dir_model.pt")
        print("Cleaned up test_dir_model.pt")

if __name__ == "__main__":
    test_dir_predictor()
    # test_dir_upload()