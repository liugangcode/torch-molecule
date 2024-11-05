import pytest
import numpy as np
import torch
from torch_geometric.data import Data
import os
import tempfile

from torch_molecule import GNNMolecularPredictor  # Update with actual import path

# Sample SMILES for testing
SAMPLE_SMILES = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
    'CC',  # Testosterone
]

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = SAMPLE_SMILES
    y = np.array([[1], [0], [1], [0]], dtype=np.float32)
    return X, y

@pytest.fixture
def basic_predictor():
    """Create a basic GNNMolecularPredictor instance."""
    return GNNMolecularPredictor(
        num_tasks=1,
        task_type='classification',
        num_layer=3,
        emb_dim=64,
        batch_size=2,
        epochs=2,
        verbose=False
    )

def test_initialization(basic_predictor):
    """Test if the predictor initializes correctly."""
    assert basic_predictor.num_tasks == 1
    assert basic_predictor.task_type == 'classification'
    assert basic_predictor.num_layer == 3
    assert basic_predictor.emb_dim == 64
    assert isinstance(basic_predictor.model, torch.nn.Module)

def test_convert_to_pytorch_data(basic_predictor, sample_data):
    """Test conversion of SMILES to PyTorch Geometric data."""
    X, y = sample_data
    data_list = basic_predictor._convert_to_pytorch_data(X, y)
    
    assert len(data_list) == len(X)
    for data in data_list:
        assert isinstance(data, Data)
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'y')

def test_fit_and_predict(basic_predictor, sample_data):
    """Test model fitting and prediction."""
    X, y = sample_data
    X_train, y_train = X[:2], y[:2]
    X_test, y_test = X[2:], y[2:]

    # Test fitting
    basic_predictor.fit(X_train, y_train)
    assert basic_predictor.is_fitted_
    assert len(basic_predictor.fitting_loss) > 0

    # Test prediction
    predictions = basic_predictor.predict(X_test)
    assert isinstance(predictions, dict)
    assert 'prediction' in predictions
    assert predictions['prediction'].shape == (2, 1)  # One sample, one task

def test_save_load_model(basic_predictor, sample_data):
    """Test model saving and loading."""
    X, y = sample_data
    
    # Fit the model
    basic_predictor.fit(X[:2], y[:2])
    
    # Create temporary directory for model saving
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.pt")
        
        # Test saving
        basic_predictor.save_model(model_path)
        assert os.path.exists(model_path)
        
        # Create new predictor for loading
        new_predictor = GNNMolecularPredictor(
            num_tasks=1,
            task_type='classification',
            num_layer=3,
            emb_dim=64
        )
        
        # Test loading
        new_predictor.load_model(model_path)
        assert new_predictor.is_fitted_
        
        # Compare predictions
        pred1 = basic_predictor.predict(X[2:])
        pred2 = new_predictor.predict(X[2:])
        np.testing.assert_array_almost_equal(
            pred1['prediction'],
            pred2['prediction'],
            decimal=4  # Reduced precision requirement
        )

def test_auto_fit(basic_predictor, sample_data):
    """Test auto_fit functionality."""
    X, y = sample_data
    X_train, y_train = X[:2], y[:2]
    X_val, y_val = X[2:], y[2:]

    # Define minimal search space for testing
    search_parameters = {
        'num_layer': (2, 3),
        'emb_dim': (32, 64),
        'learning_rate': (0.001, 0.01)
    }

    # Test auto_fit with validation data
    basic_predictor.autofit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        search_parameters=search_parameters,
        n_trials=2
    )
    assert basic_predictor.is_fitted_
    
    # Test prediction after autofit
    predictions = basic_predictor.predict(X_val)
    assert isinstance(predictions, dict)
    assert 'prediction' in predictions
    assert predictions['prediction'].shape == (2, 1)

def test_invalid_inputs():
    """Test handling of invalid inputs."""
    with pytest.raises(ValueError):
        GNNMolecularPredictor(num_tasks=0)
    
    with pytest.raises(ValueError):
        GNNMolecularPredictor(task_type='invalid_type')
    
    predictor = GNNMolecularPredictor()
    with pytest.raises(ValueError):
        predictor.fit([], [])  # Empty input
        
    with pytest.raises(ValueError):
        predictor.fit(['CC'], [[1]], ['CC'], None)  # Mismatched validation data

def test_error_handling(basic_predictor):
    """Test error handling in various scenarios."""
    # Test prediction without fitting
    with pytest.raises(AttributeError):
        basic_predictor.predict(['CC'])
    
    # Test saving without fitting
    with pytest.raises(ValueError):
        basic_predictor.save_model('test.pt')
    
    # Test invalid file extension for saving
    basic_predictor.is_fitted_ = True  # Mock fitted state
    with pytest.raises(ValueError):
        basic_predictor.save_model('test.invalid')

def test_evaluation_metrics(sample_data):
    """Test different evaluation metrics."""
    X, y = sample_data
    
    # Test with different metrics
    for metric in ['accuracy', 'roc_auc']:
        predictor = GNNMolecularPredictor(
            evaluate_criterion=metric,
            epochs=2
        )
        predictor.fit(X[:2], y[:2])
        assert predictor.is_fitted_
        
        predictions = predictor.predict(X[2:])
        assert 'prediction' in predictions

if __name__ == '__main__':
    pytest.main([__file__])