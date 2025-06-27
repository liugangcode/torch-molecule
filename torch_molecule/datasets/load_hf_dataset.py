import os
import csv
import gzip
import numpy as np
from huggingface_hub import hf_hub_download
from typing import List, Tuple, Optional

from .constant import TOXCAST_TASKS

def _load_from_hf(
    repo_id: str,
    filename: str,
    local_dir: str,
    target_cols: List[str],
    SMILES_col: str = "smiles"
) -> Tuple[List[str], np.ndarray, str]:
    """
    Load dataset from Hugging Face Hub.
    
    Parameters
    ----------
    repo_id : str
        Hugging Face repository ID
    filename : str
        Name of the file to download
    local_dir : str
        Path where the data should be saved
    target_cols : List[str]
        List of target column names
    SMILES_col : str, optional
        Name of the SMILES column, by default "smiles"
    
    Returns
    -------
    Tuple[List[str], np.ndarray, str]
        - smiles_list: List of SMILES strings
        - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
        - local_dir: Path where the data is saved
    """
    if os.path.exists(local_dir):
        print(f"Found existing file at {local_dir}")
    else:
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        
        print(f"Downloading dataset from Hugging Face Hub...")
        # Download the dataset from Hugging Face Hub
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(local_dir),
            repo_type="dataset"
        )
        
        # If the downloaded file has a different name, rename it to match local_dir
        if downloaded_file != local_dir:
            os.rename(downloaded_file, local_dir)
        
        print(f"Dataset downloaded and saved to {local_dir}")
    
    # Determine if file is compressed based on extension
    is_compressed = local_dir.endswith('.gz')
    
    # Read CSV file (handle both compressed and uncompressed)
    if is_compressed:
        with gzip.open(local_dir, 'rt', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Get column names from the first row
            columns = reader.fieldnames
            if columns is None:
                raise ValueError("CSV file appears to be empty or malformed")
            
            # Check if SMILES column exists
            if SMILES_col not in columns:
                raise ValueError(f"SMILES column '{SMILES_col}' not found in dataset. Available columns: {list(columns)}")
            
            # Check if target columns exist
            missing_cols = [col for col in target_cols if col not in columns]
            if missing_cols:
                raise ValueError(f"Target columns {missing_cols} not found in dataset. Available columns: {list(columns)}")
            
            # Read all rows
            rows = list(reader)
    else:
        with open(local_dir, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Get column names from the first row
            columns = reader.fieldnames
            if columns is None:
                raise ValueError("CSV file appears to be empty or malformed")
            
            # Check if SMILES column exists
            if SMILES_col not in columns:
                raise ValueError(f"SMILES column '{SMILES_col}' not found in dataset. Available columns: {list(columns)}")
            
            # Check if target columns exist
            missing_cols = [col for col in target_cols if col not in columns]
            if missing_cols:
                raise ValueError(f"Target columns {missing_cols} not found in dataset. Available columns: {list(columns)}")
            
            # Read all rows
            rows = list(reader)
    
    # Extract SMILES strings
    smiles_list = [row[SMILES_col] for row in rows]
    
    # Extract target properties
    property_data = []
    for row in rows:
        row_properties = []
        for col in target_cols:
            try:
                # Convert to float, handle potential missing values
                value_str = row[col].strip()
                if value_str == '' or value_str.lower() in ['nan', 'na', 'null', 'none']:
                    value = np.nan
                else:
                    value = float(value_str)
                row_properties.append(value)
            except (ValueError, KeyError):
                # Handle cases where conversion fails - keep as NaN
                row_properties.append(np.nan)
        property_data.append(row_properties)
    
    # Convert to numpy array
    property_numpy = np.array(property_data)
        
    return smiles_list, property_numpy, local_dir


def load_qm9(
    local_dir: str = "torchmol_data",
    target_cols: List[str] = ["gap"],
    return_local_data_path: bool = False,
):
    """
    Load QM9 dataset from Hugging Face Hub.

    Source: http://quantum-machine.org/datasets/
    
    Parameters
    ----------
    local_dir : str, optional
        Path where the data should be saved, by default "torchmol_data"
    target_cols : List[str], optional
        List of target column names, by default ["gap"]
    return_local_data_path : bool, optional
        Whether to return the local data path, by default False
    
    Returns
    -------
    Tuple[List[str], np.ndarray] or Tuple[List[str], np.ndarray, str]
        If return_local_data_path is False:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
        If return_local_data_path is True:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
            - local_data_path: Path where the data is saved
    """
    smiles_list, property_numpy, local_data_path = _load_from_hf(
        repo_id="liuganghuggingface/QM9",
        filename="qm9.csv",
        local_dir=f"{local_dir}/qm9.csv",
        target_cols=target_cols,
        SMILES_col="smiles"
    )
    if return_local_data_path:
        return smiles_list, property_numpy, local_data_path
    else:
        return smiles_list, property_numpy

def load_chembl2k(
    local_dir: str = "torchmol_data",
    target_cols: List[str] = ["ABCB1", "ABL1", "ADRA1D", "ADRA2B", "ADRB2", "CA12", "CA2", "CA9", "CACNA1H", "CDK2", "CHRM1", "CHRM3", "CHRM4", "CNR1", "CYP1A2", "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4", "DRD2", "DRD3", "DRD4", "EGFR", "ESR1", "FLT1", "HRH1", "HTR1A", "HTR2A", "HTR2B", "HTR2C", "HTR6", "KCNH2", "KDR", "LCK", "MCL1", "OPRK1", "PPARG", "PTGS1", "SIGMAR1", "SLC6A2", "SLC6A4"],
    return_local_data_path: bool = False,
):
    """
    Load ChEMBL2K dataset from Hugging Face Hub.

    ChEMBL2K is introduced from "Learning Molecular Representation in a Cell" (ICLR 2025)
    
    Parameters
    ----------
    local_dir : str, optional
        Path where the data should be saved, by default "torchmol_data"
    target_cols : List[str], optional
        List of target column names, by default ["ABCB1", "ABL1", "ADRA1D", "ADRA2B", "ADRB2", "CA12", "CA2", "CA9", "CACNA1H", "CDK2", "CHRM1", "CHRM3", "CHRM4", "CNR1", "CYP1A2", "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4", "DRD2", "DRD3", "DRD4", "EGFR", "ESR1", "FLT1", "HRH1", "HTR1A", "HTR2A", "HTR2B", "HTR2C", "HTR6", "KCNH2", "KDR", "LCK", "MCL1", "OPRK1", "PPARG", "PTGS1", "SIGMAR1", "SLC6A2", "SLC6A4"]
    return_local_data_path : bool, optional
        Whether to return the local data path, by default False
    
    Returns
    -------
    Tuple[List[str], np.ndarray] or Tuple[List[str], np.ndarray, str]
        If return_local_data_path is False:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
        If return_local_data_path is True:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
            - local_data_path: Path where the data is saved
    """
    smiles_list, property_numpy, local_data_path = _load_from_hf(
        repo_id="liuganghuggingface/InfoAlign-Data",
        filename="chembl2k_raw/assays.csv.gz",
        local_dir=f"{local_dir}/cheml2k.csv.gz",
        target_cols=target_cols,
        SMILES_col="smiles"
    )
    if return_local_data_path:
        return smiles_list, property_numpy, local_data_path
    else:
        return smiles_list, property_numpy


def load_broad6k(
    local_dir: str = "torchmol_data",
    target_cols: List[str] = ["220_692", "221_693", "231_703", "233_706", "234_707", "235_708", "238_712", "239_713", "240_714", "241_715", "242_716", "243_717", "244_718", "246_720", "247_721", "248_722", "249_724", "251_727", "260_738", "264_742", "265_743", "267_745", "268_746", "269_747", "270_748", "274_752", "275_754", "276_755", "277_756", "278_757", "279_758", "280_759"],
    return_local_data_path: bool = False,
):
    """
    Load Broad6K dataset from Hugging Face Hub. 

    Broad6K is introduced from "Learning Molecular Representation in a Cell" (ICLR 2025)

    TODO: replace the column names with the original names
    
    Parameters
    ----------
    local_dir : str, optional
        Path where the data should be saved, by default "torchmol_data"
    target_cols : List[str], optional
        List of target column names, by default all tasks
    return_local_data_path : bool, optional
        Whether to return the local data path, by default False
    
    Returns
    -------
    Tuple[List[str], np.ndarray] or Tuple[List[str], np.ndarray, str]
        If return_local_data_path is False:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
        If return_local_data_path is True:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
            - local_data_path: Path where the data is saved
    """
    smiles_list, property_numpy, local_data_path = _load_from_hf(
        repo_id="liuganghuggingface/InfoAlign-Data",
        filename="broad6k_raw/assays.csv.gz",
        local_dir=f"{local_dir}/broad6k.csv.gz",
        target_cols=target_cols,
        SMILES_col="smiles"
    )
    if return_local_data_path:
        return smiles_list, property_numpy, local_data_path
    else:
        return smiles_list, property_numpy


def load_toxcast(
    local_dir: str = "torchmol_data",
    target_cols: List[str] = TOXCAST_TASKS,
    return_local_data_path: bool = False,
):
    """
    Load ToxCast dataset from Hugging Face Hub.

    Source: https://www.epa.gov/comptox-tools/exploring-toxcast-data
    
    Parameters
    ----------
    local_dir : str, optional
        Path where the data should be saved, by default "torchmol_data"
    target_cols : List[str], optional
        List of target column names, by default all tasks
    return_local_data_path : bool, optional
        Whether to return the local data path, by default False
    
    Returns
    -------
    Tuple[List[str], np.ndarray] or Tuple[List[str], np.ndarray, str]
        If return_local_data_path is False:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
        If return_local_data_path is True:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
            - local_data_path: Path where the data is saved
    """
    smiles_list, property_numpy, local_data_path = _load_from_hf(
        repo_id="liuganghuggingface/toxcast",
        filename="toxcast_data.csv.gz",
        local_dir=f"{local_dir}/toxcast.csv.gz",
        target_cols=target_cols,
        SMILES_col="smiles"
    )
    if return_local_data_path:
        return smiles_list, property_numpy, local_data_path
    else:
        return smiles_list, property_numpy



def load_admet(
    local_dir: str = "torchmol_data",
    target_cols: List[str] = ["AMES","BBB_Martins","Bioavailability_Ma","CYP1A2_Veith","CYP2C19_Veith","CYP2C9_Substrate_CarbonMangels","CYP2C9_Veith","CYP2D6_Substrate_CarbonMangels","CYP2D6_Veith","CYP3A4_Substrate_CarbonMangels","CYP3A4_Veith","Caco2_Wang","Carcinogens_Lagunin","Clearance_Hepatocyte_AZ","Clearance_Microsome_AZ","ClinTox","DILI","HIA_Hou","Half_Life_Obach","HydrationFreeEnergy_FreeSolv","LD50_Zhu","Lipophilicity_AstraZeneca","NR-AR-LBD","NR-AR","NR-AhR","NR-Aromatase","NR-ER-LBD","NR-ER","NR-PPAR-gamma","PAMPA_NCATS","PPBR_AZ","Pgp_Broccatelli","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53","Skin_Reaction","Solubility_AqSolDB","VDss_Lombardo","hERG"],
    return_local_data_path: bool = False,
):
    """
    Load ADMET dataset from Hugging Face Hub.
    
    Source: https://github.com/swansonk14/admet_ai/blob/main/admet_ai/resources/data/admet.csv
    
    Parameters
    ----------
    local_dir : str, optional
        Path where the data should be saved, by default "torchmol_data"
    target_cols : List[str], optional
        List of target column names, by default all tasks
    return_local_data_path : bool, optional
        Whether to return the local data path, by default False
    
    Returns
    -------
    Tuple[List[str], np.ndarray] or Tuple[List[str], np.ndarray, str]
        If return_local_data_path is False:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
        If return_local_data_path is True:
            - smiles_list: List of SMILES strings
            - property_numpy: 2D numpy array with properties (rows=molecules, cols=targets)
            - local_data_path: Path where the data is saved
    """
    smiles_list, property_numpy, local_data_path = _load_from_hf(
        repo_id="liuganghuggingface/admet",
        filename="admet.csv",
        local_dir=f"{local_dir}/admet.csv",
        target_cols=target_cols,
        SMILES_col="smiles"
    )
    if return_local_data_path:
        return smiles_list, property_numpy, local_data_path
    else:
        return smiles_list, property_numpy

