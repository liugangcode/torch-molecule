import numpy as np
import os
import shutil
from torch_molecule import (
    GNNMolecularPredictor,
    GREAMolecularPredictor,
    SGIRMolecularPredictor,
    SupervisedMolecularEncoder
)
from torch_molecule.utils.search import ParameterType, ParameterSpec

# Common test data
SMILES_LIST = [
    'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
    'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
    'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
    'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
]
CLASSIFICATION_PROPERTIES = np.array([0, 0, 1, 1])
REGRESSION_PROPERTIES = np.array([1.2, 2.3, 3.4, 4.5])

# Common search parameters
BASE_SEARCH_PARAMETERS = {
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

def test_gnn_predictor():
    print("\n=== Testing GNN Model ===")
    
    # Initialize model
    model = GNNMolecularPredictor(
        num_task=1,
        task_type="classification",
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=5,
        verbose=True
    )

    # Test basic fitting and prediction
    print(model.num_task)
    model.fit(SMILES_LIST[:3], CLASSIFICATION_PROPERTIES[:3])
    predictions = model.predict(SMILES_LIST[3:])
    print(f"GNN Prediction shape: {predictions['prediction'].shape}")

    # Test autofit
    model_auto = GNNMolecularPredictor(
        num_task=1,
        task_type="classification",
        epochs=3,
        verbose=False
    )
    model_auto.autofit(
        SMILES_LIST,
        CLASSIFICATION_PROPERTIES,
        search_parameters=BASE_SEARCH_PARAMETERS,
        n_trials=2
    )

def test_grea_predictor():
    print("\n=== Testing GREA Model ===")
    
    # Initialize model
    model = GREAMolecularPredictor(
        num_task=1,
        task_type="classification",
        gamma=0.8,
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=5,
        verbose=True
    )

    # Test basic fitting and prediction
    model.fit(SMILES_LIST[:3], CLASSIFICATION_PROPERTIES[:3])
    predictions = model.predict(SMILES_LIST[3:])
    print(f"GREA Prediction shape: {predictions['prediction'].shape}")

    # Test Hugging Face integration
    model_hf = GREAMolecularPredictor(
        num_task=1,
        task_type="regression",
        gamma=0.8,
        num_layer=2,
        hidden_size=64,
        batch_size=4,
        epochs=2,
        verbose=False,
        model_name='GREA_O2'
    )
    
    model_hf.autofit(SMILES_LIST[:3], REGRESSION_PROPERTIES[:3])
    
    # Uncomment to test HF upload (requires HF_TOKEN)
    # repo_id = "your-huggingface-repo-id"
    # model_hf.save(
    #     repo_id=repo_id,
    #     commit_message="Upload GREA model for testing",
    #     private=False
    # )

def test_sgir_predictor():
    print("\n=== Testing SGIR Model ===")
    
    unlabeled_smiles = SMILES_LIST[2:]  # Using last 2 SMILES as unlabeled data
    
    # Initialize model
    model = SGIRMolecularPredictor(
        num_task=1,
        task_type="regression",
        gamma=0.8,
        num_layer=3,
        hidden_size=128,
        batch_size=4,
        epochs=20,
        verbose=True,
        warmup_epoch=5,
        labeling_interval=1,
        augmentation_interval=1
    )

    # Test fitting with unlabeled data
    model.fit(SMILES_LIST[:2], REGRESSION_PROPERTIES[:2], X_unlbl=unlabeled_smiles)
    predictions = model.predict(SMILES_LIST[2:])
    print(f"SGIR Prediction shape: {predictions['prediction'].shape}")

def test_supervised_encoder():
    print("\n=== Testing Supervised Encoder ===")
    
    simple_molecules = [
        "CC(=O)O",  # Acetic acid
        "CCO",      # Ethanol
        "CCCC",     # Butane
        "c1ccccc1", # Benzene
        "CCN",      # Ethylamine
    ]
    
    # Test with predefined tasks
    encoder = SupervisedMolecularEncoder(
        predefined_task=["morgan", "maccs", "logP"],
        epochs=2,
        verbose=True
    )
    encoder.fit(simple_molecules)
    encodings = encoder.encode(simple_molecules)
    print(f"Supervised Encoder shape: {encodings.shape}")

    # Test with custom tasks
    y_custom = np.random.rand(len(simple_molecules), 2)  # 2 custom tasks
    encoder_custom = SupervisedMolecularEncoder(
        num_task=2,
        epochs=2,
        verbose=True
    )
    encoder_custom.fit(simple_molecules, y_train=y_custom)
    encodings_custom = encoder_custom.encode(simple_molecules)
    print(f"Custom Supervised Encoder shape: {encodings_custom.shape}")

def test_model_save_load():
    print("\n=== Testing Model Save/Load ===")
    
    # Test saving and loading for each model type
    models = {
        'gnn': GNNMolecularPredictor(num_task=1, task_type="classification"),
        'grea': GREAMolecularPredictor(num_task=1, task_type="classification"),
        'sgir': SGIRMolecularPredictor(num_task=1, task_type="regression")
    }
    
    for model_name, model in models.items():
        save_path = f"test_{model_name}_model.pt"
        
        print(model_name, model)
        # Fit with minimal data
        if model_name == 'sgir':
            model.fit(SMILES_LIST[:4],  REGRESSION_PROPERTIES[:4], X_unlbl=SMILES_LIST[:4])
        else:
            model.fit(SMILES_LIST[:4], CLASSIFICATION_PROPERTIES[:4])
        
        # Test save/load
        model.save_to_local(save_path)
        
        new_model = type(model)(num_task=1, 
                              task_type="classification" if model_name != 'sgir' else "regression")
        new_model.load_from_local(save_path)
        
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    # Run all tests
    test_gnn_predictor()
    test_grea_predictor()
    test_sgir_predictor()
    test_supervised_encoder()
    test_model_save_load()