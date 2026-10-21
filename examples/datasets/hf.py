import os
import tempfile
import shutil
from torch_molecule.datasets import load_qm9, load_chembl2k, load_broad6k, load_toxcast, load_admet, load_zinc250k
import numpy as np
import csv
import gzip

def load_dataset(dataset_name="qm9"):
    if dataset_name == "qm9":
        return load_qm9
    elif dataset_name == "chembl2k":
        return load_chembl2k
    elif dataset_name == "broad6k":
        return load_broad6k
    elif dataset_name == "toxcast":
        return load_toxcast
    elif dataset_name == "admet":
        return load_admet
    elif dataset_name == "zinc250k":
        return load_zinc250k
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def test_download_and_cleanup(dataset_name="qm9"):
    """
    Test QM9 dataset download, print results, and cleanup local files.
    """
    print("=" * 60)
    print(f"Testing {dataset_name} Dataset Download and Loading")
    print("=" * 60)
    
    # Create a temporary directory for testing
    if False:
        temp_dir = tempfile.mkdtemp()
        test_csv_path = temp_dir
    else:
        test_csv_path = "torchmol_data"

    load_func = load_dataset(dataset_name)
    
    try:
        print(f"\n1. Testing download to temporary location: {test_csv_path}")
        print("-" * 40)
        
        # Test with default target columns
        result = load_func(
            local_dir=test_csv_path,
            return_local_data_path=True,
        )
        molecular_dataset, local_data_path = result
        
        # Print results
        print(f"\nResults:")
        print(f"- Number of molecules: {len(molecular_dataset.data)}")
        print(f"- Property array shape: {molecular_dataset.target.shape if molecular_dataset.target is not None else 'None'}")
        print(f"- File exists: {os.path.exists(local_data_path)}")
        print(f"- File size: {os.path.getsize(local_data_path) if os.path.exists(local_data_path) else 0} bytes")
        
        print(f"\nFirst 5 SMILES:")
        for i, smiles in enumerate(molecular_dataset.data[:5]):
            print(f"  {i+1}. {smiles}")
        
        print(f"\nFirst 5 property values (gap):")
        if molecular_dataset.target is not None:
            for i, prop in enumerate(molecular_dataset.target[:5]):
                print(f"  {i+1}. {prop[0]:.6f}")
        else:
            print("  No property values available (target is None)")
        
        print(f"\nProperty statistics:")
        # Calculate statistics excluding NaN values
        if molecular_dataset.target is not None:
            non_null_mask = ~np.isnan(molecular_dataset.target)
            non_null_values = molecular_dataset.target[non_null_mask]
            
            print(f"  Total values: {molecular_dataset.target.size}")
            print(f"  Non-null values: {non_null_values.size}")
            print(f"  Null values: {molecular_dataset.target.size - non_null_values.size}")
            print(f"  Non-null percentage: {(non_null_values.size / molecular_dataset.target.size * 100):.2f}%")
            
            if non_null_values.size > 0:
                print(f"  Min (non-null): {non_null_values.min():.6f}")
                print(f"  Max (non-null): {non_null_values.max():.6f}")
                print(f"  Mean (non-null): {non_null_values.mean():.6f}")
                print(f"  Std (non-null): {non_null_values.std():.6f}")
            else:
                print("  No non-null values found")
        else:
            print("  No property statistics available (target is None)")
        
        # Test loading from existing file (should not download again)
        print(f"\n2. Testing loading from existing file")
        print("-" * 40)
        
        result2 = load_func(
            local_dir=test_csv_path,
            return_local_data_path=True,
        )
        molecular_dataset2, local_data_path2 = result2
        
        print(f"Second load results:")
        print(f"- Same number of molecules: {len(molecular_dataset2.data) == len(molecular_dataset.data)}")
        if molecular_dataset.target is not None and molecular_dataset2.target is not None:
            print(f"- Same property shape: {molecular_dataset2.target.shape == molecular_dataset.target.shape}")
        elif molecular_dataset.target is None and molecular_dataset2.target is None:
            print(f"- Same property shape: True (both are None)")
        else:
            print(f"- Same property shape: False (different None status)")
        print(f"- Local data path: {local_data_path2}")
        
        # Test with multiple target columns (if available)
        print(f"\n3. Testing with multiple target columns")
        print("-" * 40)
        # check if .gz file
        if local_data_path.endswith('.gz'):
            with gzip.open(local_data_path, 'rt', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                available_cols = list(reader.fieldnames) # type: ignore
            print(f"Available columns: {available_cols}")
        else:
            with open(local_data_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                available_cols = list(reader.fieldnames) # type: ignore
            print(f"Available columns: {available_cols}")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    
    finally:
        # Cleanup: Delete the temporary directory and all files
        print(f"\n4. Cleaning up temporary files")
        print("-" * 40)
        
        if os.path.exists(test_csv_path):
            file_count = len(os.listdir(test_csv_path))
            shutil.rmtree(test_csv_path)
            print(f"- Deleted temporary directory: {test_csv_path}")
            print(f"- Removed {file_count} file(s)")
        
        print(f"- Cleanup completed successfully")
    
    print("\n" + "=" * 60)
    print(f"{dataset_name} Dataset Test Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # test_download_and_cleanup(dataset_name="qm9")
    # test_download_and_cleanup(dataset_name="chembl2k")
    # test_download_and_cleanup(dataset_name="broad6k")
    # test_download_and_cleanup(dataset_name="toxcast")
    # test_download_and_cleanup(dataset_name="admet")
    test_download_and_cleanup(dataset_name="zinc250k")