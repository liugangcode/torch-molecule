# from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import random
from typing import List

import joblib
import numpy as np
from joblib import delayed
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')

import main_llamole.graph_ga.crossover as co, main_llamole.graph_ga.mutate as mu
from main_llamole.optimizer import BaseOptimizer

import time 

MINIMUM = 1e-10

def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs 
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args, reference_smiles, targets, train_smiles, task_type, active_atoms, num_mol_each_target_dict):
        super().__init__(args, reference_smiles, targets, train_smiles, task_type, active_atoms, num_mol_each_target_dict)
        self.model_name = "graph_ga"

    def _optimize(self, all_predictor_list, seed_smiles, seed_targets, all_target_np, config):

        # self.oracle.assign_evaluator(oracle)
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        # pool = joblib.Parallel(n_jobs=3)
        
        # print('all_target_np', all_target_np.shape)
        # print('seed_targets', seed_targets.shape)
        retrieved_idx = np.abs(all_target_np[:, np.newaxis, :] - seed_targets[np.newaxis, :, :])
        retrieved_idx = np.nanmean(retrieved_idx, axis=-1)
        retrieved_idx = np.argsort(retrieved_idx, axis=1)
        print('retrieved_idx', retrieved_idx[:100], retrieved_idx.shape)

        # retrieved_idx = retrieved_idx[:, :10]
        print('retrieved_idx', retrieved_idx.shape)
        if self.oracle.task_label in ['esol', 'lipo']:
            retrieved_idx = [np.random.choice(row, 5, replace=False) for row in retrieved_idx]
            offspring_size = 5
            population_size = 5
            mutation_rate = config["mutation_rate"]
        elif self.oracle.task_label in ['bace_b', 'bbbp_b', 'hiv_b']:
            retrieved_idx = [np.random.choice(row, 10, replace=False) for row in retrieved_idx]
            offspring_size = 10
            population_size = 10
            mutation_rate = config["mutation_rate"]
        elif self.oracle.task_label in ['thermal']:
            retrieved_idx = [np.random.choice(row, 5, replace=False) for row in retrieved_idx]
            offspring_size = 5
            population_size = 5
            mutation_rate = config["mutation_rate"]
        elif self.oracle.task_label in ['TPSA-BertzCT-NumRings']:
            retrieved_idx = [np.random.choice(row, 3, replace=False) for row in retrieved_idx]
            offspring_size = 3
            population_size = 3
            # mutation_rate = config["mutation_rate"]
            mutation_rate = 0.01
        else:
            # retrieved_idx = [np.random.choice(row, 10, replace=False) for row in retrieved_idx]
            # offspring_size = 10
            # population_size = 10
            # mutation_rate = config["mutation_rate"]
            retrieved_idx = [np.random.choice(row, 3, replace=False) for row in retrieved_idx]
            offspring_size = 3
            population_size = 3
            mutation_rate = 0.067

        retrieved_idx = np.array(retrieved_idx)
        starting_population_all = np.array(seed_smiles)[retrieved_idx]
        interval = len(self.oracle.mol_num) // 10
        for idx_target, starting_population in enumerate(starting_population_all):
            # if idx_target not in [0,1,9999,10000]:
            #     continue
            start_time = time.time()
            if idx_target % (interval+1) == 0:
                print('optimizing target: ', idx_target)
            current_target = all_target_np[idx_target]
            population_smiles = starting_population.tolist()
            population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
            
            mol_pop_list = [Chem.MolToSmiles(mol) for mol in population_mol]
            population_scores = self.oracle(mol_pop_list, idx_target, current_target, all_predictor_list)

            patience = 0
            opt_count = 0
            while True:
                time_cur = time.time()
                opt_count += 1
                print(f'optimizing target {idx_target} for {opt_count}th iteration')
                if len(list(self.mol_buffer[idx_target].items())) > 0:
                    self.sort_buffer(idx_target)
                    old_score = np.mean([item[1][0] for item in list(self.mol_buffer[idx_target].items())[:self.oracle.mol_num[idx_target]]])
                else:
                    old_score = 0

                # new_population
                # print('111', time.time()-time_cur)
                mating_pool = make_mating_pool(population_mol, population_scores, population_size)
                # offspring_mol = pool(delayed(reproduce)(mating_pool, config["mutation_rate"]) for _ in range(config["offspring_size"]))
                # print('222', time.time()-time_cur)
                offspring_mol = pool(delayed(reproduce)(mating_pool, mutation_rate) for _ in range(offspring_size))


                # add new_population
                # print('333', time.time()-time_cur)
                population_mol += offspring_mol
                population_mol = self.sanitize(population_mol)
                # offspring_mol = self.sanitize(offspring_mol)
                # if len(offspring_mol) > 0:
                #     population_mol = offspring_mol

                # stats
                # print('444', time.time()-time_cur)
                old_scores = population_scores
                # population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
                mol_pop_list = [Chem.MolToSmiles(mol) for mol in population_mol]

                # print('555', time.time()-time_cur)
                # big model make this step very slow
                population_scores = self.oracle(mol_pop_list, idx_target, current_target, all_predictor_list)

                # print('666', time.time()-time_cur)

                population_tuples = list(zip(population_scores, population_mol))
                # population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
                population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
                population_mol = [t[1] for t in population_tuples]
                population_scores = [t[0] for t in population_tuples]

                ### early stopping
                if opt_count >= 5:
                    print('single target optimization achieved 5 iterations, breaking')
                    break
                
                if len(list(self.mol_buffer[idx_target].items())) > 0:
                    self.sort_buffer(idx_target)
                    new_score = np.mean([item[1][0] for item in list(self.mol_buffer[idx_target].items())[:self.oracle.mol_num[idx_target]]])
                    if (new_score - old_score) < 1e-3:
                        patience += 1
                        if patience >= self.args.patience:
                            if idx_target % interval == 0:
                                self.log_intermediate(idx_target)
                                end_time = time.time()
                                print(f'convergence criteria met, abort for {idx_target} ({end_time-start_time}s) ...... ')
                                break
                    else:
                        patience = 0

                    old_score = new_score
                
                if self.oracle.last_log[idx_target] >=  self.oracle.mol_num[idx_target]:
                    print('777', time.time()-time_cur)
                    self.sort_buffer(idx_target)
                    if idx_target % (interval+1) == 0:
                        self.log_intermediate(idx_target)
                        end_time = time.time()
                        print(f'taking  {end_time-start_time} s')
                    break
