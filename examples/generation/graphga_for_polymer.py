import pandas as pd
import numpy as np
from torch_molecule import GraphGAMolecularGenerator
import os

# Load training data
path_to_data = pd.read_csv('../../data/polymer100.csv')
train_data = path_to_data
test_data = path_to_data

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

model_cond = GraphGAMolecularGenerator(
    num_task=len(property_names),  # Number of properties from the dataset
    population_size=20,
    offspring_size=10,
    n_jobs=1,
    iteration=5,
    verbose="progress_bar"
)
print("GraphGA Model (conditional) initialized successfully")
model_cond.fit(train_smiles_list, train_property_array)

# test_property_array = test_property_array[:100]
# test_smiles_list = test_smiles_list[:100]
print('test_property_array', test_property_array.shape)
generated_smiles_list = model_cond.generate(test_property_array)
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
output_path = 'output/graphga_generation_plym.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
