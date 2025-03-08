import os
import numpy as np
import warnings
from tqdm import tqdm
from joblib import delayed

import torch
from torch_molecule.generator.graphga.modeling_graph_ga import GraphGAMolecularGenerator
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# Define the minimum value for score adjustment
MINIMUM = 1e-3

def test_graph_ga_generator():
    # Test data
    smiles_list = [
        'CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C',
        'CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C',
        'C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F',
        'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
    ]
    smiles_list = smiles_list * 25  # Create 100 molecules for training
    
    # Create synthetic properties for conditional generation
    # Let's use molecular weight and logP as example properties
    properties = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = rdMolDescriptors.CalcExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            properties.append([mw, logp])
        else:
            properties.append([np.nan, np.nan])
    
    properties = np.array(properties)
    
    print("\n=== Testing GraphGA model initialization ===")
    # 1. Initialize model for unconditional generation
    model_uncond = GraphGAMolecularGenerator(
        num_task=0,  # 0 for unconditional generation
        population_size=50,
        offspring_size=25,
        mutation_rate=0.01,
        n_jobs=1,  # Use 1 for easier debugging
        iteration=3,
        verbose=True
    )
    print("GraphGA Model (unconditional) initialized successfully")
    
    # 2. Initialize model for conditional generation
    model_cond = GraphGAMolecularGenerator(
        num_task=2,  # 2 properties: MW and logP
        population_size=50,
        offspring_size=25,
        mutation_rate=0.01,
        n_jobs=1,  # Use 1 for easier debugging
        iteration=3,
        verbose=True
    )
    print("GraphGA Model (conditional) initialized successfully")
    
    # 3. Test fitting
    print("\n=== Testing GraphGA model fitting (unconditional) ===")
    model_uncond.fit(smiles_list)
    print("GraphGA Model (unconditional) fitting completed")
    
    print("\n=== Testing GraphGA model fitting (conditional) ===")
    model_cond.fit(smiles_list, properties)
    print("GraphGA Model (conditional) fitting completed")
    
    # 4. Test unconditional generation
    print("\n=== Testing GraphGA model unconditional generation ===")
    generated_smiles_uncond = model_uncond.generate(num_samples=5)
    print(f"Generated {len(generated_smiles_uncond)} molecules unconditionally")
    print("Example generated SMILES:", generated_smiles_uncond[:2])

    
    # 5. Test conditional generation
    print("\n=== Testing GraphGA model conditional generation ===")
    # Create target properties (MW around 400, logP around 3)
    target_properties = np.array([[400, 3.0], [450, 4.0]])
    generated_smiles_cond = model_cond.generate(labels=target_properties)
    print(f"Generated {len(generated_smiles_cond)} molecules conditionally")
    print("Example generated SMILES:", generated_smiles_cond)
    
    # Verify properties of generated molecules
    print("\nProperties of conditionally generated molecules:")
    for i, smiles in enumerate(generated_smiles_cond):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Chem.Descriptors.MolWt(mol)
            logp = Chem.Descriptors.MolLogP(mol)
            print(f"Target: MW={target_properties[i][0]}, logP={target_properties[i][1]}")
            print(f"Actual: MW={mw:.2f}, logP={logp:.2f}")

    
    # 6. Test model saving and loading
    print("\n=== Testing GraphGA model saving and loading ===")
    save_path = "graph_ga_test_model.pkl"
    try:
        model_cond.save_to_local(save_path)
        print(f"GraphGA Model saved to {save_path}")
        
        new_model = GraphGAMolecularGenerator(num_task=2)
        new_model.load_from_local(save_path)
        print("GraphGA Model loaded successfully")
        
        # Test generation with loaded model
        generated_smiles = new_model.generate(labels=target_properties[:1])
        print("Generated molecules with loaded model:", len(generated_smiles))
    except Exception as e:
        print(f"Error in saving/loading: {e}")
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Cleaned up {save_path}")

if __name__ == "__main__":
    test_graph_ga_generator()