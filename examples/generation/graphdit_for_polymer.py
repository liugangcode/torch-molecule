import pandas as pd
import numpy as np
from torch_molecule import GraphDITMolecularGenerator
import matplotlib.pyplot as plt
import os
from rdkit import Chem

# Load training data
path_to_data = ''
train_data = pd.read_csv(f'{path_to_data}/train.csv')
test_data = pd.read_csv(f'{path_to_data}/test.csv')

# Extract SMILES and properties
train_smiles_list = train_data['smiles'].tolist()
test_smiles_list = test_data['smiles'].tolist()

# Get property names (all columns except 'smiles')
property_names = [col for col in train_data.columns if col != 'smiles']

# Convert properties to numpy arrays, replacing empty strings with NaN and then with None
train_property_array = train_data[property_names].replace('', np.nan).to_numpy()
print('train_smiles_list', train_smiles_list[0])
print('train_property_array', train_property_array[0], train_property_array.shape)

test_property_array = test_data[property_names].replace('', np.nan).to_numpy()
print('GraphDITMolecularGenerator', GraphDITMolecularGenerator)

# Get the number of heavy atoms in the training/test set and filter molecules
max_node = 0
atom_counts = []
filtered_train_smiles = []
filtered_train_properties = []

for i, smiles in enumerate(train_smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Count only heavy atoms (non-hydrogen atoms)
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        atom_counts.append(num_heavy_atoms)
        max_node = max(max_node, num_heavy_atoms)
        
        # Only keep molecules with 50 or fewer heavy atoms
        if num_heavy_atoms <= 50:
            filtered_train_smiles.append(smiles)
            filtered_train_properties.append(train_property_array[i])

# Convert filtered properties to numpy array
filtered_train_properties = np.array(filtered_train_properties)

print('max_heavy_atoms', max_node)
print(f'Original dataset size: {len(train_smiles_list)}, Filtered dataset size: {len(filtered_train_smiles)}')

# Plot histogram of heavy atom counts
plt.figure(figsize=(10, 6))
plt.hist(atom_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Number of Heavy Atoms', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Molecule Sizes (Number of Heavy Atoms)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save the histogram
plt.savefig('output/atom_hist.png', dpi=300)
plt.close()
print("Atom histogram saved to output/atom_hist.png")

# Update training data to use filtered molecules
train_smiles_list = filtered_train_smiles
train_property_array = filtered_train_properties

model_cond = GraphDITMolecularGenerator(
    max_node=50,
    task_type=['regression'] * len(property_names),
    y_dim=len(property_names),
    batch_size=1024,
    drop_condition=0.1,
    verbose=True,
    epochs=10000,
)

model_cond.fit(train_smiles_list, train_property_array)
model_cond.save('output/graphdit_plym.pt')
print('model_cond saved to output/graphdit_plym.pt', model_cond)

# generate with retry logic for invalid molecules
max_retries = 10
generated_smiles_list = model_cond.generate(test_property_array)

# Function to check if SMILES is valid
def is_valid_smiles(smiles):
    if smiles is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Retry generation for invalid molecules
for retry in range(max_retries - 1):  # Already did first generation
    # Find indices of invalid SMILES
    invalid_indices = [i for i, smiles in enumerate(generated_smiles_list) if not is_valid_smiles(smiles)]
    
    if not invalid_indices:
        print(f"All SMILES valid after {retry + 1} attempts")
        break
        
    print(f"Retry {retry + 1}: Regenerating {len(invalid_indices)} invalid molecules")
    
    # Extract properties for invalid molecules
    invalid_properties = test_property_array[invalid_indices]
    
    # Regenerate only for invalid molecules
    new_smiles = model_cond.generate(invalid_properties)
    
    # Replace invalid molecules with new generations
    for idx, new_idx in enumerate(invalid_indices):
        generated_smiles_list[new_idx] = new_smiles[idx]
    
    if retry == max_retries - 2:  # Last iteration
        print(f"Reached maximum retries ({max_retries}). {len(invalid_indices)} molecules still invalid.")

print('test_property_array', test_property_array.shape)
print('generated_smiles_list', generated_smiles_list[0])

# Save results to CSV
results_df = pd.DataFrame()
results_df['generated'] = generated_smiles_list
results_df['reference'] = test_smiles_list

# Add property columns
for i, prop_name in enumerate(property_names):
    results_df[prop_name] = test_property_array[:, i]
    
# Replace None/NaN values with empty strings for CSV output
results_df = results_df.replace({np.nan: None})

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save to CSV
output_path = 'output/graphdit_generation_plym.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
