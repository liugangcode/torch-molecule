import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Type
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ...base import BaseMolecularGenerator
from ...utils import graph_from_smiles, graph_to_smiles

import random
import joblib
from rdkit import Chem

@dataclass
class GraphGAMolecularGenerator(BaseMolecularGenerator):
    """This predictor implements the Graph Genetic Algorithm for molecular generation.
    """

    # GA parameters
    population_size: int = 10
    offspring_size: int = 10
    mutation_rate: float = 0.01
    n_jobs: int = -1  # Number of parallel jobs
    patience: int = 5
    
    # Other parameters
    verbose: bool = False
    model_name: str = "GraphGAMolecularGenerator"
    model_class: Type[Transformer] = field(default=Transformer, init=False)

    # Non-init fields
    
    def __post_init__(self):
        """Initialize the model after dataclass initialization."""
        super().__post_init__()
        self.mol_buffer = {}
        self.is_fitted_ = True  # GA doesn't require traditional fitting

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return [
            "population_size", "offspring_size", "mutation_rate",
            "n_jobs", "patience", "verbose"
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        params = ["max_node", "hidden_size", "num_layer", "num_head", "mlp_ratio", 
                 "dropout", "drop_condition", "X_dim", "E_dim", "y_dim", "task_type"]
        
        if checkpoint is not None:
            if "hyperparameters" not in checkpoint:
                raise ValueError("Checkpoint missing 'hyperparameters' key")
            return {k: checkpoint["hyperparameters"][k] for k in params}
        
        return {k: getattr(self, k) for k in params}
        
    def _convert_to_pytorch_data(self, X, y=None):
        """Convert numpy arrays to PyTorch Geometric data format.
        """
        raise NotImplementedError("GraphGA does not support converting to PyTorch Geometric data format")
    

    def _initialize_model(
        self,
        model_class: Type[torch.nn.Module],
        checkpoint: Optional[Dict] = None
    ) -> torch.nn.Module:
        raise NotImplementedError("GraphGA does not support initializing the model")
        """Initialize the model with parameters or a checkpoint."""
        model_params = self._get_model_params(checkpoint)
        self.model = model_class(**model_params)
        self.model = self.model.to(self.device)
        
        if checkpoint is not None:
            self._setup_diffusion_params(checkpoint)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        return self.model

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]] = None,
    ) -> "GraphGAMolecularGenerator":
        """Store training data statistics for generation."""
        if y_train is not None:
            self.y_train = np.array(y_train)
        self.X_train = X_train
        self.is_fitted_ = True
        return self

    def _make_mating_pool(self, population_mol: List[Mol], population_scores, offspring_size: int):
        """Create mating pool based on molecule scores."""
        population_scores = [s + MINIMUM for s in population_scores]
        sum_scores = sum(population_scores)
        population_probs = [p / sum_scores for p in population_scores]
        mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
        return mating_pool

    def _reproduce(self, mating_pool, mutation_rate):
        """Create new molecule through crossover and mutation."""
        parent_a = random.choice(mating_pool)
        parent_b = random.choice(mating_pool)
        new_child = co.crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
        return new_child

    @torch.no_grad()
    def generate(
        self, 
        labels: Optional[Union[List[List], np.ndarray, torch.Tensor]] = None,
        num_samples: int = 32
    ) -> List[str]:
        """Generate molecules using genetic algorithm optimization."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before generating")
            
        # Initialize population from training data
        population_idx = np.random.choice(len(self.X_train), self.population_size)
        population_smiles = [self.X_train[i] for i in population_idx]
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        
        # Setup parallel processing
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        for generation in range(self.patience):
            # Score current population
            population_scores = [1.0] * len(population_mol)  # Replace with actual scoring
            
            # Create mating pool and new offspring
            mating_pool = self._make_mating_pool(population_mol, population_scores, self.offspring_size)
            offspring_mol = pool(
                delayed(self._reproduce)(mating_pool, self.mutation_rate) 
                for _ in range(self.offspring_size)
            )
            
            # Combine populations and select best
            population_mol += [m for m in offspring_mol if m is not None]
            population_scores = [1.0] * len(population_mol)  # Replace with actual scoring
            
            # Select top molecules for next generation
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
            population_tuples = population_tuples[:self.population_size]
            
            population_mol = [t[1] for t in population_tuples]
            
        # Convert final population to SMILES
        return [Chem.MolToSmiles(mol) for mol in population_mol[:num_samples]]
    
