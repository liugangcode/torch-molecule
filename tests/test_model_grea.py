import unittest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock
from typing import List

# from .grea_predictor import GREAMolecularPredictor
from torch_molecule import GREAMolecularPredictor

class TestGREAMolecularPredictor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_smiles = [
            "CC1=CC=CC=C1",  # toluene
            "CC(=O)O",       # acetic acid
            "CCO"            # ethanol
        ]
        
        # Default initialization parameters
        self.predictor = GREAMolecularPredictor(
            gamma=0.4,
            num_tasks=1,
            task_type="classification",
            num_layer=3,
            emb_dim=128,
            batch_size=2,
            epochs=2,
            device="cpu"
        )
        
        # Create a temporary directory for model saving/loading tests
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pt")

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary files
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_initialization(self):
        """Test if the predictor initializes with correct parameters."""
        self.assertEqual(self.predictor.gamma, 0.4)
        self.assertEqual(self.predictor.num_tasks, 1)
        self.assertEqual(self.predictor.task_type, "classification")
        self.assertEqual(self.predictor.num_layer, 3)
        self.assertEqual(self.predictor.emb_dim, 128)
        self.assertEqual(str(self.predictor.device), "cpu")
        self.assertIsNotNone(self.predictor.model)

    def test_get_param_names(self):
        """Test if all required parameters are included in get_param_names."""
        param_names = GREAMolecularPredictor._get_param_names()
        required_params = [
            "gamma", "num_tasks", "task_type", "num_layer", "emb_dim",
            "gnn_type", "drop_ratio", "norm_layer", "batch_size",
            "learning_rate", "weight_decay"
        ]
        for param in required_params:
            self.assertIn(param, param_names)

    @patch('torch_geometric.loader.DataLoader')
    def test_predict(self, mock_dataloader):
        """Test prediction functionality with mock data."""
        # Mock the model's forward pass
        mock_output = {
            "prediction": torch.tensor([[0.8], [0.2], [0.6]]),
            "variance": torch.tensor([[0.1], [0.1], [0.1]])
        }
        self.predictor.model = MagicMock()
        self.predictor.model.eval = MagicMock()
        self.predictor.model.return_value = mock_output
        
        # Mock the data conversion and loading
        self.predictor._validate_inputs = MagicMock(return_value=(self.test_smiles, None))
        self.predictor._convert_to_pytorch_data = MagicMock()
        mock_dataloader.return_value = [MagicMock()]
        self.predictor.is_fitted_ = True

        # Make predictions
        predictions = self.predictor.predict(self.test_smiles)

        # Verify predictions structure and types
        self.assertIn("prediction", predictions)
        self.assertIn("variance", predictions)
        self.assertIsInstance(predictions["prediction"], np.ndarray)
        self.assertIsInstance(predictions["variance"], np.ndarray)

    def test_save_model_not_fitted(self):
        """Test that saving an unfitted model raises an error."""
        self.predictor.is_fitted_ = False
        with self.assertRaises(ValueError):
            self.predictor.save_model(self.model_path)

    def test_save_model_invalid_path(self):
        """Test that saving with invalid file extension raises an error."""
        self.predictor.is_fitted_ = True
        invalid_path = os.path.join(self.temp_dir, "model.invalid")
        with self.assertRaises(ValueError):
            self.predictor.save_model(invalid_path)

    @patch('torch.save')
    def test_save_model_success(self, mock_save):
        """Test successful model saving."""
        self.predictor.is_fitted_ = True
        self.predictor.fitting_epoch = 10
        self.predictor.fitting_loss = 0.5
        self.predictor.fitting_loss_mean = 0.4
        
        self.predictor.save_model(self.model_path)
        
        # Verify torch.save was called with correct arguments
        mock_save.assert_called_once()
        save_dict = mock_save.call_args[0][0]
        
        required_keys = {
            "model_state_dict",
            "hyperparameters",
            "fitting_epoch",
            "fitting_loss",
            "fitting_loss_mean",
            "model_name",
            "date_saved",
            "version"
        }
        for key in required_keys:
            self.assertIn(key, save_dict)

if __name__ == '__main__':
    unittest.main()