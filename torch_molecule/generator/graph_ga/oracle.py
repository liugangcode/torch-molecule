import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from typing import List, Any
from ...utils.graph.features import getmorganfingerprint


class Oracle:
    """The default Oracle class for scoring molecules in GraphGA.
    
    This class wraps predictive models (like RandomForestRegressor) to score molecules
    based on their properties. It handles conversion of SMILES to fingerprints.
    
    Parameters
    ----------
    models : List[Any], optional
        List of trained models that implement a predict method.
        If None, RandomForestRegressors will be created when fit is called.
    num_task : int, default=1
        Number of properties to predict.
    """
    
    def __init__(self, models=None, num_task=1):
        self.models = models if models is not None else [RandomForestRegressor() for _ in range(num_task)]
        self.num_task = num_task if models is None else len(models)
        
    def _convert_to_fingerprint(self, molecules):
        """Convert SMILES or RDKit molecules to fingerprints."""
        if isinstance(molecules[0], str):
            return np.array([getmorganfingerprint(Chem.MolFromSmiles(mol)) for mol in molecules])
        else:
            return np.array([getmorganfingerprint(mol) for mol in molecules])
    
    def fit(self, X_train, y_train):
        """Fit the underlying models with training data.
        
        Parameters
        ----------
        X_train : List[str] or List[RDKit.Mol]
            Training molecules as SMILES strings or RDKit Mol objects.
        y_train : np.ndarray
            Training labels with shape (n_samples, num_task).
        
        Returns
        -------
        self : Oracle
            Fitted oracle.
        """
        X_train_fp = self._convert_to_fingerprint(X_train)
        
        for i in range(self.num_task):
            nan_mask = ~np.isnan(y_train[:, i])
            y_train_ = y_train[:, i][nan_mask]
            X_train_fp_ = X_train_fp[nan_mask]
            self.models[i].fit(X_train_fp_, y_train_)
            
        return self
    
    def __call__(self, molecules, target_values):
        """Score molecules based on their predicted properties.
        
        Parameters
        ----------
        molecules : List[str] or List[RDKit.Mol]
            Molecules to score as SMILES strings or RDKit Mol objects.
        target_values : np.ndarray,
            Scores will be based on distance to these targets.
            
        Returns
        -------
        List[float]
            Scores for each molecule.
        """
        fps = self._convert_to_fingerprint(molecules)
        scores_list = []
        
        for i, fp in enumerate(fps):
            if self.num_task == 1:
                score = self.models[0].predict([fp])[0]
                scores_list.append(float(score))
            else:
                mol_scores = []
                for idx in range(self.num_task):
                    pred = self.models[idx].predict([fp])[0]
                    
                    if target_values is not None and not np.isnan(target_values[0][idx]):
                        # Lower score for values closer to target
                        target = target_values[0][idx]
                        dist = abs(float(pred) - target) / (abs(target) + 1e-8)
                        mol_scores.append(dist)
                        
                score = np.nanmean(mol_scores)  # Lower is better when using distances
                scores_list.append(float(score))

        return scores_list