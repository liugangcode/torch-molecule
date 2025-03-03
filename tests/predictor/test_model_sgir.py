import unittest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import patch, MagicMock
from typing import List

from torch_molecule import SGIRMolecularPredictor

class TestSGIRMolecularPredictor(unittest.TestCase):
    def setUp(self):
        self.test_smiles = [
            "CC1=CC=CC=C1",  # toluene
            "CC(=O)O",       # acetic acid
            "CCO"            # ethanol
        ]
        self.test_values = [1.5, 2.7, 3.2]  # Example regression values
        
        self.predictor = SGIRMolecularPredictor(
            num_anchor=10,
            warmup_epoch=20,
            labeling_interval=5,
            augmentation_interval=5,
            top_quantile=0.1,
            label_logscale=False,
            lw_aug=1.0,
            gamma=0.4,
            num_task=1,
            task_type="regression",
            num_layer=3,
            hidden_size=128,
            batch_size=2,
            epochs=2,
            device="cpu"
        )
        
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.pt")

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_initialization(self):
        self.assertEqual(self.predictor.task_type, "regression")
        # Test invalid task type
        with self.assertRaises(ValueError):
            SGIRMolecularPredictor(task_type="classification")
        # Test invalid num_task
        with self.assertRaises(ValueError):
            SGIRMolecularPredictor(num_task=2)

    def test_get_param_names(self):
        param_names = SGIRMolecularPredictor._get_param_names()
        required_params = [
            "num_anchor", "warmup_epoch", "labeling_interval",
            "augmentation_interval", "top_quantile", "label_logscale", "lw_aug"
        ]
        for param in required_params:
            self.assertIn(param, param_names)
        
        # Verify inheritance of parent parameters
        parent_params = ["num_layer", "hidden_size", "batch_size", "epochs"]
        for param in parent_params:
            self.assertIn(param, param_names)

    def test_get_default_search_space(self):
        search_space = self.predictor._get_default_search_space()
        # Test SGIR-specific parameters
        self.assertIn("num_anchor", search_space)
        self.assertIn("labeling_interval", search_space)
        self.assertIn("augmentation_interval", search_space)
        self.assertIn("top_quantile", search_space)
        self.assertIn("lw_aug", search_space)

    @patch('torch_geometric.loader.DataLoader')
    def test_fit_validation_error(self, mock_dataloader):
        X_train = self.test_smiles[:2]
        y_train = np.array(self.test_values[:2])
        
        # Test validation input mismatch error
        with self.assertRaises(ValueError):
            self.predictor.fit(X_train, y_train, X_val=["CCO"], y_val=None)

    @patch('torch_geometric.loader.DataLoader')
    def test_train_epoch_with_augmentation(self, mock_dataloader):
        train_loader = MagicMock()
        optimizer = MagicMock()
        augmented_dataset = {
            'representations': torch.randn(4, 128),
            'labels': torch.randn(4, 1)
        }
        
        self.predictor.model = MagicMock()
        self.predictor.model.compute_loss = MagicMock(return_value=torch.tensor(0.5))
        self.predictor.loss_criterion = MagicMock(return_value=torch.tensor(0.3))
        
        losses = self.predictor._train_epoch(train_loader, augmented_dataset, optimizer)
        self.assertIsInstance(losses, list)

if __name__ == '__main__':
    unittest.main()