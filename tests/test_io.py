import numpy as np
import os
import shutil
from pathlib import Path
from torch_molecule import (
    GNNMolecularPredictor,
    GREAMolecularPredictor,
    SGIRMolecularPredictor,
    SupervisedMolecularEncoder
)
from unittest.mock import patch, MagicMock
import torch
from huggingface_hub import HfApi
import json

# Test data
SMILES_LIST = [
    'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
    'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
]
CLASSIFICATION_PROPERTIES = np.array([0, 1])
REGRESSION_PROPERTIES = np.array([1.2, 2.3])

def setup_test_directory():
    """Create and return test directory path."""
    test_dir = Path("test_model_checkpoints")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    return test_dir

def cleanup_test_directory(test_dir):
    """Clean up test directory."""
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_local_save_load():
    """Test local save and load functionality for all models."""
    print("\n=== Testing Local Save/Load ===")
    
    test_dir = setup_test_directory()
    
    models = {
        'gnn': (GNNMolecularPredictor(num_task=1, task_type="classification"), CLASSIFICATION_PROPERTIES),
        'grea': (GREAMolecularPredictor(num_task=1, task_type="classification"), CLASSIFICATION_PROPERTIES),
        'sgir': (SGIRMolecularPredictor(num_task=1, task_type="regression"), REGRESSION_PROPERTIES),
    }
    
    for model_name, (model, properties) in models.items():
        print(f"\nTesting {model_name.upper()} model:")
        
        # Test save_to_local and load_from_local
        save_path = test_dir / f"{model_name}_local.pt"
        try:
            # Fit and save model
            model.fit(SMILES_LIST, properties, X_unlbl=SMILES_LIST)
            model.save_to_local(str(save_path))
            print(f"✓ Successfully saved model to {save_path}")
            
            # Load model
            new_model = type(model)(num_task=1, task_type=model.task_type)
            new_model.load_from_local(str(save_path))
            print("✓ Successfully loaded model")
            
            # Verify loaded model works
            predictions = new_model.predict(SMILES_LIST)
            print("✓ Successfully made predictions with loaded model")
            
        except Exception as e:
            print(f"✗ Error in local save/load test: {str(e)}")
    
    cleanup_test_directory(test_dir)

def test_generic_save_load():
    """Test generic save and load methods."""
    print("\n=== Testing Generic Save/Load Interface ===")
    
    test_dir = setup_test_directory()
    
    # Test with GNN model as example
    model = GNNMolecularPredictor(num_task=1, task_type="classification")
    model.fit(SMILES_LIST, CLASSIFICATION_PROPERTIES)
    
    # Test local path
    local_path = test_dir / "generic_local.pt"
    try:
        # Test save method
        model.save(path=str(local_path))
        print("✓ Successfully saved model using generic save")
        
        # Test load method
        new_model = GNNMolecularPredictor(num_task=1, task_type="classification")
        new_model.load(path=str(local_path))
        print("✓ Successfully loaded model using generic load")
        
    except Exception as e:
        print(f"✗ Error in generic save/load test: {str(e)}")
    
    cleanup_test_directory(test_dir)

def test_huggingface_interface():
    """Test Hugging Face interface with mocked upload/download."""
    print("\n=== Testing Hugging Face Interface ===")
    
    test_dir = setup_test_directory()
    
    # Test with GREA model as example
    model = GREAMolecularPredictor(
        num_task=1,
        task_type="regression",
        gamma=0.8,
        num_layer=2,
        hidden_size=64,
        batch_size=4,
        epochs=2,
        verbose=False,
        model_name='GREA_TEST'
    )
    
    # Fit model with sample data
    model.fit(SMILES_LIST, REGRESSION_PROPERTIES)
    
    # Mock data for testing
    repo_id = "liuganghuggingface/test-torch-molecule"
    metrics = {
        "mse": 0.123,
        "r2": 0.456
    }

    # Test HF save interface
    with patch('huggingface_hub.HfApi') as mock_hf_api, \
         patch('torch.save') as mock_torch_save, \
         patch('json.dump') as mock_json_dump, \
         patch('builtins.open', create=True) as mock_open:
        
        # Mock HF API methods
        mock_api_instance = mock_hf_api.return_value
        mock_api_instance.create_repo = MagicMock()
        mock_api_instance.upload_file = MagicMock()
        
        try:
            # Test model upload
            model.save_to_hf(
                repo_id=repo_id,
                metrics=metrics,
                commit_message="Test upload",
                private=False
            )
            
            # Verify repo creation
            mock_api_instance.create_repo.assert_called_once_with(
                repo_id=repo_id,
                private=False,
                exist_ok=True
            )
            
            # Verify model state was saved
            mock_torch_save.assert_called()
            
            # Verify metadata and metrics were saved
            mock_json_dump.assert_called()
            
            print("✓ Successfully tested model upload to HF")
            
        except Exception as e:
            print(f"✗ Error in HF save test: {str(e)}")

    # Test HF load interface
    with patch('huggingface_hub.hf_hub_download') as mock_hf_download, \
         patch('torch.load') as mock_torch_load, \
         patch('json.load') as mock_json_load:
        
        # Mock the downloaded checkpoint
        mock_checkpoint = {
            "model_state_dict": MagicMock(),
            "hyperparameters": {
                "num_task": 1,
                "task_type": "regression",
                "gamma": 0.8,
                "num_layer": 2,
                "hidden_size": 64
            },
            "version": "1.0.0"
        }
        mock_torch_load.return_value = mock_checkpoint
        
        # Mock metadata and metrics
        
        try:
            # Test model download
            download_path = test_dir / "downloaded_model.pt"
            mock_hf_download.return_value = str(download_path)
            
            new_model = GREAMolecularPredictor(
                num_task=1,
                task_type="regression"
            )
            new_model.load_from_hf(repo_id=repo_id, path=str(download_path))
            
            # Verify download was attempted
            mock_hf_download.assert_called()
            
            # Verify model state was loaded
            mock_torch_load.assert_called()
            
            print("✓ Successfully tested model download from HF")
            
        except Exception as e:
            print(f"✗ Error in HF load test: {str(e)}")

    # Test error cases
    print("\n=== Testing HF Error Cases ===")
    
    # Test invalid repo_id
    with patch('huggingface_hub.HfApi') as mock_hf_api:
        mock_api_instance = mock_hf_api.return_value
        mock_api_instance.create_repo.side_effect = Exception("Invalid repo")
        
        # try:
        # model.save_to_hf(repo_id="invalid/repo")
        # print("✗ Should have raised error for invalid repo")
        # except Exception:
        #     print("✓ Correctly handled invalid repo_id")
    
    # Test missing token
    with patch('os.environ', {}):
        # try:
        # get HF_TOKEN from environment variable
        # token = os.getenv("HUGGINGFACE_TOKEN")  # If set as an environment variable   )
        model.save_to_hf(repo_id=repo_id, private=True)
        print("✗ Should have raised error for missing token")

        # except ValueError:
        #     print("✓ Correctly handled missing HF token for private repo")
    
    # Test invalid model state
    model.is_fitted_ = False
    try:
        model.save_to_hf(repo_id=repo_id)
        print("✗ Should have raised error for unfitted model")
    except ValueError:
        print("✓ Correctly handled unfitted model upload attempt")
    
    cleanup_test_directory(test_dir)

def test_supervised_encoder_save_load():
    """Test save/load functionality for SupervisedMolecularEncoder."""
    print("\n=== Testing Supervised Encoder Save/Load ===")
    
    test_dir = setup_test_directory()
    
    # Initialize encoder with predefined tasks
    encoder = SupervisedMolecularEncoder(
        predefined_task=["morgan", "maccs"],
        epochs=2,
        verbose=False
    )
    
    try:
        # Fit encoder
        encoder.fit(SMILES_LIST)
        
        # Test local save/load
        save_path = test_dir / "encoder_local.pt"
        encoder.save_to_local(str(save_path))
        print("✓ Successfully saved encoder")
        
        # Load encoder
        new_encoder = SupervisedMolecularEncoder(
            predefined_task=["morgan", "maccs"],
            epochs=2
        )
        new_encoder.load_from_local(str(save_path))
        print("✓ Successfully loaded encoder")
        
        # Verify loaded encoder works
        encodings = new_encoder.encode(SMILES_LIST)
        print("✓ Successfully generated encodings with loaded encoder")
        print(f"  Encoding shape: {encodings.shape}")
        
    except Exception as e:
        print(f"✗ Error in encoder save/load test: {str(e)}")
    
    cleanup_test_directory(test_dir)

def test_error_handling():
    """Test error handling in save/load operations."""
    print("\n=== Testing Error Handling ===")
    
    model = GNNMolecularPredictor(num_task=1, task_type="classification")
    
    # Test loading non-existent file
    try:
        model.load_from_local("non_existent_file.pt")
        print("✗ Should have raised FileNotFoundError")
    except FileNotFoundError:
        print("✓ Correctly handled non-existent file")
    
    # Test loading without fitting
    try:
        model.save_to_local("unfitted_model.pt")
        print("✗ Should have raised error for unfitted model")
    except Exception as e:
        print("✓ Correctly handled unfitted model save attempt")
    
    # Test generic save without path or repo_id
    try:
        model.save()
        print("✗ Should have raised ValueError")
    except ValueError:
        print("✓ Correctly handled missing path and repo_id")

if __name__ == "__main__":
    # Run all IO tests
    test_local_save_load()
    test_generic_save_load()
    test_huggingface_interface()
    test_supervised_encoder_save_load()
    test_error_handling()