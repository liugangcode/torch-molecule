import os
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random
from rdkit import Chem

# Import GraphGA model
from torch_molecule import GraphGAMolecularGenerator
# Import TDC Oracle
from tdc import Oracle

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default=None, help='Specify a single oracle name to process')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)

# Oracle names
oracle_names = [
    "Albuterol_Similarity", "Amlodipine_MPO", "Celecoxib_Rediscovery", 
    "Deco_Hop", "DRD2", "Fexofenadine_MPO", "GSK3B", "Isomers_C7H8N2O2",
    "Isomers_C7H8N2O3", "Isomers_C9H10N2O2PF2Cl", "JNK3", "Median 1", 
    "Median 2", "Mestranol_Similarity", "Osimertinib_MPO", "Perindopril_MPO",
    "QED", "Ranolazine_MPO", "Scaffold_Hop", "Sitagliptin_MPO", 
    "Thiothixene_Rediscovery", "Troglitazone_Rediscovery", "Valsartan_Smarts",
    "Zaleplon_MPO"
]

if args.task:
    if args.task == 'Median_1':
        args.task = 'Median 1'
    elif args.task == 'Median_2':
        args.task = 'Median 2'
    if args.task not in oracle_names:
        raise ValueError(f"Task '{args.task}' is not a valid oracle name.")
    oracle_names = [args.task]

# Wrapper for oracle that takes target label into account
class TargetOracle:
    def __init__(self, oracle):
        self.oracle = oracle
    
    def __call__(self, smiles_list, target_value):
        # Convert rdkit.Chem.rdchem.Mol to smiles if smiles_list is a list of rdkit.Chem.rdchem.Mol
        if isinstance(smiles_list[0], Chem.rdchem.Mol):
            smiles_list = [Chem.MolToSmiles(mol) for mol in smiles_list]
        raw_scores = self.oracle(smiles_list)
        # Calculate the difference between target and actual
        scores = [abs(float(target_value) - float(score)) for score in raw_scores]
        return scores
    
# placeholder for training smiles
train_smiles = pd.read_csv('../../data/molecule100.csv')['smiles'].tolist()
print(f"Loaded {len(train_smiles)} training SMILES")
# remove non-valid smiles
train_smiles = [s for s in train_smiles if Chem.MolFromSmiles(s) is not None]
print(f"Removed {len(train_smiles) - len(train_smiles)} non-valid SMILES")

for idx, name in enumerate(oracle_names, 1):
    print(f"Processing {name} [{idx}/{len(oracle_names)}] ({idx / len(oracle_names) * 100:.1f}%)")
    oracle = Oracle(name=name)
    
    # Create output directory
    output_dir = os.path.join('generated', name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Target labels to generate molecules for
    sample_size = 1000
    target_labels = [1] * sample_size

    print(f"Generating molecules for sample_size")
    
    # Create wrapped oracle
    wrapped_oracle = TargetOracle(oracle)
    
    # Initialize GraphGA model
    model = GraphGAMolecularGenerator(
        num_task=1,  # 1 for single property optimization
        population_size=100,
        offspring_size=50,
        mutation_rate=0.01,
        n_jobs=10,
        iteration=10,
        verbose="progress_bar"
    )
    
    # Fit the model
    print("Fitting GraphGA model...")
    model.fit(train_smiles, oracle=wrapped_oracle)
    
    # Generate molecules
    print(f"Generating molecules with target score")
    generated_smiles = model.generate(labels=target_labels)
    
    # Evaluate generated molecules
    results = []
    for smiles in tqdm(generated_smiles, desc="Evaluating"):
        score = oracle([smiles])[0]
        results.append({'smiles': smiles, 'score': score})

    # save to generated/{task_name}.csv
    output_file = os.path.join(output_dir, f'{name}.csv')
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Saved {len(results)} molecules to {output_file}")
    
    # Also add to positive.csv if score > 0.75
    pos_results = [r for r in results if r['score'] > 0.75]
    if pos_results:
        pos_file = os.path.join(output_dir, 'positive.csv')
        if os.path.exists(pos_file):
            df_pos = pd.read_csv(pos_file)
            df_new = pd.DataFrame(pos_results)
            df_combined = pd.concat([df_pos, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['smiles']).reset_index(drop=True)
            df_combined.to_csv(pos_file, index=False)
        else:
            pd.DataFrame(pos_results).to_csv(pos_file, index=False)
        print(f"Added {len(pos_results)} molecules to positive.csv")