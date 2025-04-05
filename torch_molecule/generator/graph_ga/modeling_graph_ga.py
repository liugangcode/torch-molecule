import numpy as np
import random
import joblib
from joblib import delayed
from rdkit import Chem
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
from warnings import warn

from .crossover import crossover
from .mutate import mutate
from .oracle import Oracle

from ...base import BaseMolecularGenerator

@dataclass
class GraphGAMolecularGenerator(BaseMolecularGenerator):
    """This generator implements the Graph Genetic Algorithm for molecular generation.
    
    References
    ----------
    - A Graph-Based Genetic Algorithm and Its Application to the Multiobjective Evolution of 
    Median Molecules. Journal of Chemical Information and Computer Sciences. https://pubs.acs.org/doi/10.1021/ci034290p
    - Implementation: https://github.com/wenhao-gao/mol_opt

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
        Number of iterations for each target label (or random sample) to run the genetic algorithm.
    verbose : bool, default=False
        Whether to display progress bars and logs.
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
        joblib.dump(self.oracle, path)
        if self.verbose:
            print(f"Saved oracle to {path}")

    def load_from_local(self):
        raise NotImplementedError(
            "GraphGA does not support loading from local. "
            "If you want to load the oracles saved through save_to_local, "
            "you need to manually load the oracle from the path with joblib.load(path) "
            "and pass it to the fit function."
        )
    
    def save_to_hf(self, repo_id: str, task_id: str = "default"):
        raise NotImplementedError("GraphGA does not support pushing to huggingface")
    
    def load_from_hf(self, repo_id: str, task_id: str = "default"):
        raise NotImplementedError("GraphGA does not support loading from huggingface")
    
    def _setup_optimizers(self):
        raise NotImplementedError("GraphGA does not support setting up optimizers")
    
    def _train_epoch(self, train_loader, optimizer):
        raise NotImplementedError("GraphGA does not support training epochs")

    def fit(
        self,
        X_train: List[str],
        y_train: Optional[Union[List, np.ndarray]] = None,
        oracle: Optional[List[Callable]] = None
    ) -> "GraphGAMolecularGenerator":
        """Fit the model to the training data.

        Parameters
        ----------
        X_train : List[str]
            Training data, which will be used as the initial population.
        y_train : Optional[Union[List, np.ndarray]]
            Training labels for conditional generation (num_task is not 0).
        oracle : Optional[Callable]
            Oracle used to score the generated molecules. If not provided, default oracles based on 
            ``sklearn.ensemble.RandomForestRegressor`` are trained on the X_train and y_train.
            
            For a customized oracle, it should be a Callable object, i.e., ``oracle(X, y)``.
            Please properly wrap your oracle to take two inputs:
              - a list of ``rdkit.Chem.rdchem.Mol`` objects and 
              - a (1, num_task) numpy array of target values that all the molecules in the list target to achieve. Take care of NaN values if any.
            
            Scores for different tasks should be aggregated, i.e., mean or sum. The return should be a list of scores (float).
            Smaller scores mean closer to the target goal.
            
            Oracles are not needed for unconditional generation.

        Returns
        -------
        self : GraphGAMolecularGenerator
            Fitted model.
        """
        self.y_train = None
        if oracle is not None:
            self.oracle = oracle
        else:
            X_train, y_train = self._validate_inputs(X_train, y_train, num_task=self.num_task, return_rdkit_mol=False)
            if y_train is not None:
                warn("No oracles provided but y_train is provided, using default oracles (RandomForestRegressor)", UserWarning)
                self.oracle = Oracle(num_task=self.num_task)
                self.oracle.fit(X_train, y_train)
                self.y_train = y_train
            else:
                assert self.num_task == 0, "No oracles or y_train provided but num_task is not 0"
                self.oracle = None
                    
        self.X_train = X_train
        self.is_fitted_ = True
        return self

    def _make_mating_pool(self, population_mol, population_scores, offspring_size: int):
        """Create mating pool where smaller scores have higher selection probabilities."""
        max_score = max(population_scores)
        # Invert scores so that smaller scores become larger probabilities
        inverted_scores = [max_score - s + 1e-6 for s in population_scores]  # Add small constant to avoid zeros
        sum_scores = sum(inverted_scores)
        population_probs = [p / sum_scores for p in inverted_scores]
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
        if label is None:
            return [1.0] * len(mol_list)  # For unconditional generation
        return self.oracle(mol_list, label)

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
            try:
                labels = np.array(labels).reshape(-1, self.num_task)
            except:
                raise ValueError(f"labels must be convertible to a numpy array with shape (-1, {self.num_task})")
            
            # Prepare all inputs for parallel processing
            parallel_inputs = []
            for i in range(labels.shape[0]):
                label = labels[i:i+1]  # Keep as 2D array
                
                # Initialize population based on similarity to target label
                if self.y_train is not None:
                    population_mol = self._initialize_population_for_label(label)
                else:
                    population_idx = np.random.choice(len(self.X_train), min(self.population_size, len(self.X_train)))
                    population_smiles = [self.X_train[idx] for idx in population_idx]
                    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
                
                parallel_inputs.append((population_mol, label))
            
            # Run GA for all labels in parallel with tqdm progress bar
            if self.verbose:
                results = joblib.Parallel(n_jobs=self.n_jobs)(
                    delayed(self._run_generation)(pop_mol, lbl) 
                    for pop_mol, lbl in tqdm(parallel_inputs, desc="Generating molecules")
                )
            else:
                results = joblib.Parallel(n_jobs=self.n_jobs)(
                    delayed(self._run_generation)(pop_mol, lbl) 
                    for pop_mol, lbl in parallel_inputs
                )
            
            # Convert results to SMILES in the original order
            all_generated_mols = [Chem.MolToSmiles(mol) for mol in results]
        else:
            # Prepare all inputs for parallel processing
            parallel_inputs = []
            for _ in range(num_samples):
                population_idx = np.random.choice(len(self.X_train), min(self.population_size, len(self.X_train)))
                population_smiles = [self.X_train[idx] for idx in population_idx]
                population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
                parallel_inputs.append((population_mol, None))
            
            # Run GA for all samples in parallel with tqdm progress bar
            if self.verbose:
                results = joblib.Parallel(n_jobs=self.n_jobs)(
                    delayed(self._run_generation)(pop_mol, lbl) 
                    for pop_mol, lbl in tqdm(parallel_inputs, desc="Generating molecules", total=num_samples)
                )
            else:
                results = joblib.Parallel(n_jobs=self.n_jobs)(
                    delayed(self._run_generation)(pop_mol, lbl) 
                    for pop_mol, lbl in parallel_inputs
                )
            
            # Convert results to SMILES
            all_generated_mols = [Chem.MolToSmiles(mol) for mol in results]
        
        return all_generated_mols
    
    def _initialize_population_for_label(self, label):
        """Initialize population based on similarity to target label."""
        similarities = []
        
        for i in range(len(self.X_train)):
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
        for generation_idx in range(self.iteration):
            if label is not None:
                population_scores = self._get_score(population_mol, label)
            else:
                population_scores = [1.0] * len(population_mol)  # For unconditional generation
            
            mating_pool = self._make_mating_pool(population_mol, population_scores, self.offspring_size)
            
            # Create offspring sequentially (parallelization is at the higher level now)
            offspring_mol = []
            for _ in range(self.offspring_size):
                offspring = self._reproduce(mating_pool, self.mutation_rate)
                offspring_mol.append(offspring)
            
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
    
