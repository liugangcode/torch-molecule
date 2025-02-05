import numpy as np
from torch_molecule import SGIRMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import os

def test_sgir_predictor():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    properties = np.array([1.2, 2.3, 3.4, 4.5])  # Regression values
    unlabeled_smiles = [
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F',
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C'
    ]

    # Initialize model
    print("\n=== Testing SGIR model initialization ===")
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
    print("\n=== Testing SGIR model fitting with unlabeled data ===")
    model.fit(smiles_list[:3], properties[:3], X_unlbl=unlabeled_smiles)

    # Test prediction
    print("\n=== Testing SGIR model prediction ===")
    predictions = model.predict(smiles_list[3:])
    print(f"Prediction shape: {predictions['prediction'].shape}")
    print(f"Prediction value: {predictions['prediction']}")

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
    
    model_auto = SGIRMolecularPredictor(
        num_task=1,
        task_type="regression",
        epochs=3,
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
    save_path = "test_sgir_model.pt"
    model.save_model(save_path)
    
    new_model = SGIRMolecularPredictor(
        num_task=1,
        task_type="regression"
    )
    new_model.load_model(save_path)

    # # Test invalid SMILES
    # try:
    #     invalid_smiles = ['CC', 'INVALID_SMILES']
    #     predictions = model.predict(invalid_smiles)
    # except Exception as e:
    #     print(f"Invalid SMILES handled with error: {str(e)}")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)

if __name__ == "__main__":
    test_sgir_predictor()