import numpy as np
import pandas as pd
import torch
from torch_molecule import RPGNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import os


def train_rpgnn_predictor():
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    properties = np.array([0, 0, 1, 1])  # Binary classification
    unlabeled_smiles = [
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C'
    ]
    num_node_feature = 9
    
    print("\n=== rpgnn model initialization ===")
    model = RPGNNMolecularPredictor(
        num_task=1,
        task_type="regression",
        num_layer=3,
        num_node_feature=num_node_feature,
        fixed_size=10,
        hidden_size=300,
        batch_size=128,
        epochs=2,
        verbose=True
    )
    print("RPGNN model initialized successfully")   
    
    # 2. Basic fitting test    
    print("\n=== Testing RPGNN model fitting ===")
    model.fit(smiles_list[:3], properties[:3])
    print("RPGNN model fitting completed")
    
    # 3. Prediction test
    print("\n=== Testing RPGNN model prediction ===")
    predictions = model.predict(smiles_list[3:])
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction for new molecule: {predictions['prediction']}")

    # 4. Auto-fitting test with search space parameters
    print("\n=== Testing RPGNN model auto-fitting ===")
    # Test auto-fitting
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
        )
    }
    
    model_auto = RPGNNMolecularPredictor(
        num_task=1,
        task_type="regression",
        epochs=3,
        num_node_feature=num_node_feature,
        fixed_size=10,
        verbose=True
    )
    
    model_auto.autofit(
        smiles_list,
        properties,
        X_unlbl=unlabeled_smiles,
        search_parameters=search_parameters,
        n_trials=2
    )

    # Test save/load
    save_path = "test_rpgnn_model.pt"
    model.save(save_path)
    
    new_model = RPGNNMolecularPredictor(
        num_task=1,
        task_type="regression"
    )
    new_model.load(save_path)

    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    train_rpgnn_predictor()