import os
import numpy as np
import warnings
import datetime
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Type
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from .model import GNN
from ...utils import graph_from_smiles
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)

@dataclass
class IRMMolecularPredictor(GNNMolecularPredictor):
    """This predictor implements a Invariant Risk Minimization model with the GNN.
    
    The full name of IRM is Invariant Risk Minimization.

    References
    ----------
    - Invariant Risk Minimization.
      https://arxiv.org/abs/1907.02893

    - Reference Code: https://github.com/facebookresearch/InvariantRiskMinimization
    
    Parameters
    ----------
    IRM_environment : Union[torch.Tensor, np.ndarray, List, str], default="random"
        Environment assignments for IRM. Can be a list of integers (one per sample),
        or "random" to assign environments randomly.
    scale : float, default=1.0
        Scaling factor for the IRM penalty term.
    penalty_weight : float, default=1.0
        Weight of the IRM penalty in the loss function.
    penalty_anneal_iters : int, default=100
        Number of iterations for annealing the penalty weight.
    """
    
    IRM_environment: Union[torch.Tensor, np.ndarray, List, str] = "random"
    scale: float = 1.0
    penalty_weight: float = 1.0
    penalty_anneal_iters: int = 100

    # Other Non-init fields
    model_name: str = "IRMMolecularPredictor"
    model_class: Type[GNN] = field(default=GNN, init=False)
    
    def __post_init__(self):
        super().__post_init__()

    @staticmethod
    def _get_param_names() -> List[str]:
        return GNNMolecularPredictor._get_param_names() + [
            "IRM_environment",
            "scale",
            "penalty_weight",
            "penalty_anneal_iters",
        ]

    def _get_default_search_space(self):
        search_space = super()._get_default_search_space().copy()
        search_space["penalty_weight"] = ParameterSpec(ParameterType.LOG_FLOAT, (1e-10, 1))
        search_space["penalty_anneal_iters"] = ParameterSpec(ParameterType.INTEGER, (10, 100))
        return search_space

    def _get_model_params(self, checkpoint: Optional[Dict] = None) -> Dict[str, Any]:
        base_params = super()._get_model_params(checkpoint)
        return base_params
    
    def set_IRM_environment(self, environment: Union[torch.Tensor, np.ndarray, List, str]):
        if isinstance(environment, str):
            if environment != "random":
                raise ValueError("IRM_environment must be 'random' if specified with a string")
            self.IRM_environment = environment
        else:
            if isinstance(environment, np.ndarray) or isinstance(environment, torch.Tensor):
                self.IRM_environment = environment.reshape(-1).tolist()
            else:
                self.IRM_environment = environment
            
            if not all(isinstance(item, int) for item in self.IRM_environment):
                raise ValueError("IRM_environment must be a list of integers")

    def _convert_to_pytorch_data(self, X, y=None):
        """Convert numpy arrays to PyTorch Geometric data format.
        """
        if self.verbose:
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        else:
            iterator = enumerate(X)

        pyg_graph_list = []
        for idx, smiles_or_mol in iterator:
            if y is not None:
                properties = y[idx]
            else:
                properties = None
            graph = graph_from_smiles(smiles_or_mol, properties, self.augmented_feature)
            g = Data()
            g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            if graph["y"] is not None:
                g.y = torch.from_numpy(graph["y"])
                del graph["y"]
   
            if graph["morgan"] is not None:
                g.morgan = torch.tensor(graph["morgan"], dtype=torch.int8).view(1, -1)
                del graph["morgan"]
            
            if graph["maccs"] is not None:
                g.maccs = torch.tensor(graph["maccs"], dtype=torch.int8).view(1, -1)
                del graph["maccs"]
    
            if self.IRM_environment == "random":
                g.environment = torch.randint(0, 2, (1,)).view(1, 1)
            elif len(X) != len(self.IRM_environment):
                raise ValueError("IRM_environment must has the same length as the input, which is {}".format(len(X)))
            else:
                if isinstance(self.IRM_environment[idx], int):
                    g.environment = torch.tensor(self.IRM_environment[idx], dtype=torch.int64).view(1, 1)
                else:
                    raise ValueError("IRM_environment must be a list of integers")
            pyg_graph_list.append(g)

        return pyg_graph_list

    def _train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        losses = []
        losses_erm = []
        penalties = []

        iterator = (
            tqdm(train_loader, desc="Training", leave=False)
            if self.verbose
            else train_loader
        )

        for step, batch in enumerate(iterator):
            batch = batch.to(self.device)
            optimizer.zero_grad()

            if epoch >= self.penalty_anneal_iters:
                penalty_weight = self.penalty_weight
            else:
                penalty_weight = 1.0
            loss, loss_erm, penalty = self.model.compute_loss(batch, self.loss_criterion, self.scale, penalty_weight)
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            optimizer.step()
            losses.append(loss.item())
            losses_erm.append(loss_erm.item())
            penalties.append(penalty.item())

            if self.verbose:
                iterator.set_postfix({"Epoch": epoch, "Total Loss": f"{loss.item():.4f}", "ERM Loss": f"{loss_erm.item():.4f}", "IRM Penalty": f"{penalty.item():.4f}"})

        return losses