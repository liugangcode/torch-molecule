import numpy as np
import random
import joblib
from joblib import delayed
from rdkit import Chem
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor
from warnings import warn


from .crossover import crossover
from .mutate import mutate

from ...base import BaseMolecularGenerator
from ...utils import graph_from_smiles, graph_to_smiles
from ...utils.graph.features import getmorganfingerprint

MINIMUM = 1e-10

@dataclass
class GraphGAMolecularGenerator(BaseMolecularGenerator):
    """This predictor implements the Graph Genetic Algorithm for molecular generation.
    Paper: A Graph-Based Genetic Algorithm and Its Application to the Multiobjective Evolution of Median Molecules 
    Link: https://pubs.acs.org/doi/full/10.1021/ci034290p
    Reference code: https://github.com/wenhao-gao/mol_opt/blob/main/main/graph_ga/run.py

    Parameters
    ----------
    num_task : int, default=0
        Number of properties to condition on. Set to 0 for unconditional generation.
    population_size : int, default=100
        Size of the population in each iteration.
    offspring_size : int, default=50
        Number of offspring molecules to generate in each iteration.
    mutation_rate : float, default=0.0067
        Probability of mutation occurring during reproduction.
    n_jobs : int, default=1
        Number of parallel jobs to run. -1 means using all processors.
    iteration : int, default=5
        Number of generations to run the genetic algorithm.
    """

    # GA parameters
    num_task: int = 0
    population_size: int = 100
    offspring_size: int = 50
    mutation_rate: float = 0.0067
    n_jobs: int = 1
    iteration: int = 5
    
    # Other parameters
    verbose: bool = False
    model_name: str = "GraphGAMolecularGenerator"
    model_class = None
    
    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    def _get_param_names() -> List[str]:
        return [
            "num_task", "population_size", "offspring_size", "mutation_rate",
            "n_jobs", "iteration", "verbose"
        ]
    
    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        raise NotImplementedError("GraphGA does not support getting model parameters")

    def save_to_local(self, path: str):
        joblib.dump(self.oracles, path)
        if self.verbose:
            print(f"Saved oracles to {path}")

    def load_from_local(self, path: str):
        self.oracles = joblib.load(path)
        if self.verbose:
            print(f"Loaded oracles from {path}")
    
    def _setup_optimizers(self):
        raise NotImplementedError("GraphGA does not support setting up optimizers")
    
    def _train_epoch(self, train_loader, optimizer):
        raise NotImplementedError("GraphGA does not support training epochs")
    
    def push_to_huggingface(self, repo_id: str, task_id: str = "default"):
        raise NotImplementedError("GraphGA does not support pushing to huggingface")
    
    def load_from_huggingface(self, repo_id: str, task_id: str = "default"):
        raise NotImplementedError("GraphGA does not support loading from huggingface")
    
    def _convert_to_fingerprint(self, X_train: List[str]) -> List[np.ndarray]:
        """Convert SMILES to fingerprint."""
        if isinstance(X_train[0], str):
            return np.array([getmorganfingerprint(Chem.MolFromSmiles(mol)) for mol in X_train])
        else:
            return np.array([getmorganfingerprint(mol) for mol in X_train])

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]] = None,
        oracles: Optional[List[Callable]] = None
    ) -> "GraphGAMolecularGenerator":
        X_train, y_train = self._validate_inputs(X_train, y_train, num_task=self.num_task, return_rdkit_mol=False)
        if y_train is not None:
            self.y_train = np.array(y_train)
            if oracles is None:
                warn("No oracles provided but y_train is provided, using default oracles (RandomForestRegressor)", UserWarning)
                self.oracles = [RandomForestRegressor() for _ in range(self.num_task)]
                for i in range(self.num_task):
                    X_train_fp = self._convert_to_fingerprint(X_train)
                    y_train_ = y_train[:, i]
                    y_train_ = y_train_[~np.isnan(y_train_)]
                    X_train_fp = X_train_fp[~np.isnan(y_train_)]
                    self.oracles[i].fit(X_train_fp, y_train_)
            else:
                self.oracles = oracles

        self.X_train = X_train
        self.is_fitted_ = True
        return self

    def _make_mating_pool(self, population_mol, population_scores, offspring_size: int):
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
        new_child = crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mutate(new_child, mutation_rate)
        return new_child

    def _sanitize_molecules(self, population_mol):
        """Sanitize molecules by removing duplicates and invalid molecules."""
        new_mol_list = []
        smiles_set = set()
        for mol in population_mol:
            if mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles is not None and smiles not in smiles_set:
                        smiles_set.add(smiles)
                        new_mol_list.append(mol)
                except ValueError:
                    pass
        return new_mol_list
    
    def _get_score(self, mol_list, label):
        label = label[0]
        assert self.num_task == len(label)
        scores_list = []
        for mol in mol_list:
            if mol is None:
                raise ValueError("Please remove invalid molecules from the population before scoring")

            if self.num_task == 1:
                fp = self._convert_to_fingerprint([mol])[0]
                score = self.oracles[0].predict([fp])[0]
                scores_list.append(float(score))
            else:
                mol_scores = []
                fp = self._convert_to_fingerprint([mol])[0]
                for idx, target in enumerate(label):
                    if not np.isnan(target):
                        pred = self.oracles[idx].predict([fp])[0]
                        # Lower score for values closer to target
                        dist = abs(float(pred) - target) / (abs(target) + 1e-8)
                        mol_scores.append(dist)  
                score = np.nanmean(mol_scores)
                scores_list.append(float(score))
        
        return scores_list

    def generate(
        self, 
        labels: Optional[Union[List[List], np.ndarray]] = None,
        num_samples: int = 32
    ) -> List[str]:
        """Generate molecules using genetic algorithm optimization."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before generating")
        
        all_generated_mols = []
        
        if labels is not None:
            labels = np.array(labels)
            if labels.shape[1] != len(self.oracles):
                raise ValueError(f"The second dimension of labels (number of tasks) must equal to the number of tasks in oracles ({len(self.oracles)})")
            
            label_iterator = range(labels.shape[0])
            if self.verbose:
                label_iterator = tqdm(label_iterator, desc="Generating molecules for labels")
                
            for i in label_iterator:
                label = labels[i:i+1]  # Keep as 2D array
                
                # Initialize population based on similarity to target label
                population_mol = self._initialize_population_for_label(label)
                
                # Run GA for this specific label
                generated_mol = self._run_generation(population_mol, label)
                all_generated_mols.append(Chem.MolToSmiles(generated_mol))
        else:
            sample_iterator = range(num_samples)
            if self.verbose:
                sample_iterator = tqdm(sample_iterator, desc="Generating molecules")
                
            for _ in sample_iterator:
                population_idx = np.random.choice(len(self.X_train), min(self.population_size, len(self.X_train)))
                population_smiles = [self.X_train[i] for i in population_idx]
                population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
                
                # Run GA for this sample
                generated_mol = self._run_generation(population_mol, None)
                all_generated_mols.append(Chem.MolToSmiles(generated_mol))
        
        return all_generated_mols
    
    def _initialize_population_for_label(self, label):
        """Initialize population based on similarity to target label."""
        similarities = []
        
        for i in range(len(self.X_train)):
            if hasattr(self, 'y_train'):
                sample_label = self.y_train[i]
                similarity = -np.nansum((sample_label - label[0])**2)
                similarities.append((i, similarity))
        
        if similarities:
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in similarities[:self.population_size]]
        else:
            top_indices = np.random.choice(len(self.X_train), min(self.population_size, len(self.X_train)))
            
        population_smiles = [self.X_train[i] for i in top_indices]
        return [Chem.MolFromSmiles(s) for s in population_smiles]
    
    def _run_generation(self, population_mol, label):
        """Run the genetic algorithm for a specific population and label."""
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        for generation in range(self.iteration):
            if label is not None:
                population_scores = self._get_score(population_mol, label)
            else:
                population_scores = [1.0] * len(population_mol)  # For unconditional generation
            
            mating_pool = self._make_mating_pool(population_mol, population_scores, self.offspring_size)
            offspring_mol = pool(
                delayed(self._reproduce)(mating_pool, self.mutation_rate) 
                for _ in range(self.offspring_size)
            )
            population_mol += offspring_mol
            population_mol = self._sanitize_molecules(population_mol)

            # Re-score the expanded population
            if label is not None:
                population_scores = self._get_score(population_mol, label)
            else:
                population_scores = [1.0] * len(population_mol)
            
            # Select top molecules for next generation
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=False) # lower score is better
            population_tuples = population_tuples[:self.population_size]
            
            population_mol = [t[1] for t in population_tuples]
        
        # Return the best molecule
        return population_mol[0]
    
