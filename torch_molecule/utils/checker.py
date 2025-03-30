import warnings
import numpy as np
from rdkit import Chem
from typing import Optional, Union, List, Tuple

class MolecularInputChecker:
    """
    Class for validating input data used in molecular models.
    """

    @staticmethod
    def validate_smiles(
        smiles: str, 
        idx: int
    ) -> Tuple[bool, Optional[str], Optional[Chem.Mol]]:
        """Validate a single SMILES string at a given index.

        Parameters
        ----------
        smiles : str
            The SMILES string to validate
        idx : int
            The index of the SMILES string in the original list

        Returns
        -------
        Tuple[bool, Optional[str], Optional[Chem.Mol]]
            A tuple containing:
            
            - A boolean indicating whether the SMILES string is valid
            - A string describing the error if the SMILES is invalid, or None if valid
            - The RDKit Mol object if valid, or None if invalid
        """
        if not smiles or not smiles.strip():
            return False, f"Empty SMILES at index {idx}", None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, f"Invalid SMILES structure at index {idx}: {smiles}", None
            return True, None, mol
        except Exception as e:
            return False, f"RDKit error at index {idx}: {str(e)}", None

    @staticmethod
    def validate_inputs(
        X: List[str],
        y: Optional[Union[List, np.ndarray]] = None,
        num_task: int = 0,
        num_pretask: int = 0,
        return_rdkit_mol: bool = True
    ) -> Tuple[Union[List[str], List["Chem.Mol"]], Optional[np.ndarray]]:
        """Validate a list of SMILES strings, and optionally validate a target array.

        Parameters
        ----------
        X : List[str]
            List of SMILES strings
        y : Optional[Union[List, np.ndarray]], optional
            Optional target values, by default None
        num_task : int, optional
            Total number of tasks; used to check dimensions of y, by default 0
        num_pretask : int, optional
            Number of (pseudo)-tasks that are predefined in the modeling; 
            used to check dimensions of y. Preliminarily used in supervised pretraining,
            by default 0
        return_rdkit_mol : bool, optional
            If True, convert SMILES to RDKit Mol objects, by default True

        Returns
        -------
        Tuple[Union[List[str], List["Chem.Mol"]], Optional[np.ndarray]]
            A tuple containing:
            
            - The original or converted SMILES (RDKit Mol objects if return_rdkit_mol=True)
            - The target array as a numpy array, or None if y was not provided

        Raises
        ------
        ValueError
            If SMILES or target dimensions are invalid
        """
        if not isinstance(X, list):
            raise ValueError("X must be a list of SMILES strings.")

        if not all(isinstance(s, str) for s in X):
            raise ValueError("All elements in X must be strings.")

        invalid_smiles = []
        rdkit_mols = []
        for i, smiles in enumerate(X):
            is_valid, error_msg, mol = MolecularInputChecker.validate_smiles(smiles, i)
            if not is_valid:
                invalid_smiles.append(error_msg)
            else:
                rdkit_mols.append(mol)

        if invalid_smiles:
            raise ValueError("Invalid SMILES found:\n" + "\n".join(invalid_smiles))

        if y is not None:
            try:
                y = np.asarray(y, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Could not convert y to numpy array: {str(e)}")

            if len(y.shape) == 1:
                if num_task - num_pretask != 1:
                    raise ValueError(
                        f"1D target array provided but num_task is {num_task - num_pretask}. "
                        "For multiple tasks, y must be 2D."
                    )
                y = y.reshape(-1, 1)

            if len(y.shape) != 2:
                raise ValueError(
                    "y must be 1D (single task) or 2D (multiple tasks). "
                    f"Got shape {y.shape}."
                )

            if y.shape[0] != len(X):
                raise ValueError(
                    f"Number of samples in y ({y.shape[0]}) must match length of X ({len(X)})."
                )

            if y.shape[1] != num_task - num_pretask:
                raise ValueError(
                    f"Second dimension of y ({y.shape[1]}) must match num_task ({num_task - num_pretask})."
                )

            inf_mask = np.isinf(y)
            if np.any(inf_mask):
                inf_indices = np.where(inf_mask)
                warnings.warn(
                    f"Infinite values found in y at indices: {list(zip(*inf_indices))}. "
                    "Converting to NaN.",
                    RuntimeWarning,
                )
                y = y.astype(float)
                y[inf_mask] = np.nan

            # nan_mask = np.isnan(y)
            # if np.any(nan_mask):
            #     nan_counts = np.sum(nan_mask, axis=0)
            #     nan_percentages = (nan_counts / len(X)) * 100
            #     task_warnings = []
            #     for task_idx, (count, percentage) in enumerate(zip(nan_counts, nan_percentages)):
            #         if count > 0:
            #             task_warnings.append(f"Task {task_idx}: {count} NaNs ({percentage:.1f}%)")

            #     warnings.warn(
            #         "NaN values present in y:\n"
            #         + "\n".join(task_warnings)
            #         + "\nSamples with NaN will be ignored or cause issues unless handled.",
            #         RuntimeWarning,
            #     )

        return rdkit_mols if return_rdkit_mol else X, y
