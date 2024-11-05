import pytest
import numpy as np
import os
from torch_molecule import GNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec

@pytest.fixture
def test_data():
    """Fixture providing test SMILES strings and properties."""
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    properties = np.array([0, 1, 0, 1])  # Binary classification
    return smiles_list, properties

@pytest.fixture
def base_model():
    """Fixture providing a basic GNN model instance."""
    return GNNMolecularPredictor(
        num_tasks=1,
        task_type="classification",
        num_layer=3,
        emb_dim=128,
        batch_size=4,
        epochs=5,
        verbose=False
    )

def test_model_initialization(base_model):
    """Test basic model initialization."""
    assert isinstance(base_model, GNNMolecularPredictor)
    assert base_model.num_tasks == 1
    assert base_model.task_type == "classification"
    assert base_model.num_layer == 3
    assert base_model.emb_dim == 128

def test_model_fitting(base_model, test_data):
    """Test model fitting functionality."""
    smiles_list, properties = test_data
    # Fit on first 3 molecules
    base_model.fit(smiles_list[:3], properties[:3])
    assert hasattr(base_model, 'is_fitted_')
    assert base_model.is_fitted_
    assert len(base_model.fitting_loss) > 0

def test_model_prediction(base_model, test_data):
    """Test model prediction functionality."""
    smiles_list, properties = test_data
    # Fit on first 3 molecules
    base_model.fit(smiles_list[:3], properties[:3])
    
    # Predict on the 4th molecule
    predictions = base_model.predict(smiles_list[3:])
    
    assert 'prediction' in predictions
    assert isinstance(predictions['prediction'], np.ndarray)
    assert predictions['prediction'].shape == (1, 1)  # One prediction, one task
    assert predictions['prediction'].dtype == np.float32 or predictions['prediction'].dtype == np.float64

def test_model_autofit(test_data):
    """Test model auto-fitting functionality."""
    smiles_list, properties = test_data
    
    model_auto = GNNMolecularPredictor(
        num_tasks=1,
        task_type="classification",
        epochs=3,
        verbose=False
    )
    
    search_parameters = {
        'num_layer': ParameterSpec(
            param_type=ParameterType.INTEGER,
            value_range=(2, 4)
        ),
        'emb_dim': ParameterSpec(
            param_type=ParameterType.INTEGER,
            value_range=(64, 256)
        ),
        'learning_rate': ParameterSpec(
            param_type=ParameterType.LOG_FLOAT,
            value_range=(1e-4, 1e-2)
        ),
    }
    
    model_auto.autofit(
        smiles_list,
        properties,
        search_parameters=search_parameters,
        n_trials=2
    )
    
    assert hasattr(model_auto, 'is_fitted_')
    assert model_auto.is_fitted_
    assert len(model_auto.fitting_loss) > 0

@pytest.mark.skip(reason="save_model and load_model methods not implemented in provided code")
def test_model_save_load(base_model, test_data, tmp_path):
    """Test model saving and loading functionality."""
    smiles_list, properties = test_data
    save_path = tmp_path / "test_model.pt"
    
    # Fit and save model
    base_model.fit(smiles_list[:3], properties[:3])
    base_model.save_model(str(save_path))
    assert save_path.exists()
    
    # Load model and verify
    new_model = GNNMolecularPredictor(
        num_tasks=1,
        task_type="classification"
    )
    new_model.load_model(str(save_path))
    
    # Compare predictions
    original_pred = base_model.predict(smiles_list[3:])
    loaded_pred = new_model.predict(smiles_list[3:])
    np.testing.assert_array_almost_equal(
        original_pred['prediction'],
        loaded_pred['prediction']
    )

def test_invalid_inputs(base_model):
    """Test model behavior with invalid inputs."""
    base_model.is_fitted_ = True
    with pytest.raises(ValueError):
        base_model.fit([], [])  # Empty inputs
    
    with pytest.raises(ValueError):
        base_model.fit(['C'], [1, 2])  # Mismatched lengths
    
    with pytest.raises(ValueError):
        base_model.predict([])  # Empty prediction input

def test_model_validation_split(base_model, test_data):
    """Test model fitting with validation split."""
    smiles_list, properties = test_data
    base_model.fit(
        smiles_list[:2], properties[:2],  # Training data
        smiles_list[2:4], properties[2:4]  # Validation data
    )
    assert hasattr(base_model, 'is_fitted_')
    assert base_model.is_fitted_
    assert len(base_model.fitting_loss) > 0