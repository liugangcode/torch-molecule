import numpy as np
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, List, Type, Callable, Literal

import torch
from torch_geometric.data import Data

from .model import GNN
from ...utils import graph_from_smiles
from ..gnn.modeling_gnn import GNNMolecularPredictor
from ...utils.search import (
    ParameterSpec,
    ParameterType,
)
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
    num_task : int, default=1
        Number of prediction tasks.
    task_type : str, default="regression"
        Type of prediction task, either "regression" or "classification".
    num_layer : int, default=5
        Number of GNN layers.
    hidden_size : int, default=300
        Dimension of hidden node features.
    gnn_type : str, default="gin-virtual"
        Type of GNN architecture to use. One of ["gin-virtual", "gcn-virtual", "gin", "gcn"].
    drop_ratio : float, default=0.5
        Dropout probability.
    norm_layer : str, default="batch_norm"
        Type of normalization layer to use. One of ["batch_norm", "layer_norm", "instance_norm", "graph_norm", "size_norm", "pair_norm"].
    graph_pooling : str, default="sum"
        Method for aggregating node features to graph-level representations. One of ["sum", "mean", "max"].
    augmented_feature : list or None, default=None
        Additional molecular fingerprints to use as features. It will be concatenated with the graph representation after pooling.
        Examples like ["morgan", "maccs"] or None.
    batch_size : int, default=128
        Number of samples per batch for training.
    epochs : int, default=500
        Maximum number of training epochs.
    loss_criterion : callable, optional
        Loss function for training.
    evaluate_criterion : str or callable, optional
        Metric for model evaluation.
    evaluate_higher_better : bool, optional
        Whether higher values of the evaluation metric are better.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    grad_clip_value : float, optional
        Maximum norm of gradients for gradient clipping.
    weight_decay : float, default=0.0
        L2 regularization strength.
    patience : int, default=50
        Number of epochs to wait for improvement before early stopping.
    use_lr_scheduler : bool, default=False
        Whether to use learning rate scheduler.
    scheduler_factor : float, default=0.5
        Factor by which to reduce learning rate when plateau is reached.
    scheduler_patience : int, default=5
        Number of epochs with no improvement after which learning rate will be reduced.
    verbose : str, default="none"
        Whether to display progress info. Options are: "none", "progress_bar", "print_statement". If any other, "none" is automatically chosen.
    device : torch.device or str, optional
        Device to run the model on.
    """
    def __init__(
        self,
        # IRM specific parameters
        IRM_environment: Union[torch.Tensor, np.ndarray, List, str] = "random",
        scale: float = 1.0,
        penalty_weight: float = 1.0,
        penalty_anneal_iters: int = 100,
        # Core model parameters
        num_task: int = 1,
        task_type: str = "regression",
        # GNN architecture parameters
        num_layer: int = 5,
        hidden_size: int = 300,
        gnn_type: str = "gin-virtual",
        drop_ratio: float = 0.5,
        norm_layer: str = "batch_norm",
        graph_pooling: str = "sum",
        augmented_feature: Optional[list[Literal["morgan", "maccs"]]] = None,
        # Training parameters
        batch_size: int = 128,
        epochs: int = 500,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        grad_clip_value: Optional[float] = None,
        patience: int = 50,
        # Learning rate scheduler parameters
        use_lr_scheduler: bool = False,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
        # Loss and evaluation parameters
        loss_criterion: Optional[Callable] = None,
        evaluate_criterion: Optional[Union[str, Callable]] = None,
        evaluate_higher_better: Optional[bool] = None,
        # General parameters
        verbose: str = "none",
        device: Optional[Union[torch.device, str]] = None,
        model_name: str = "IRMMolecularPredictor",
    ):
        super().__init__(
            num_task=num_task,
            task_type=task_type,
            num_layer=num_layer,
            hidden_size=hidden_size,
            gnn_type=gnn_type,
            drop_ratio=drop_ratio,
            norm_layer=norm_layer,
            graph_pooling=graph_pooling,
            augmented_feature=augmented_feature,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip_value=grad_clip_value,
            patience=patience,
            use_lr_scheduler=use_lr_scheduler,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            loss_criterion=loss_criterion,
            evaluate_criterion=evaluate_criterion,
            evaluate_higher_better=evaluate_higher_better,
            verbose=verbose,
            device=device,
            model_name=model_name
        )
        
        self.IRM_environment = IRM_environment
        self.scale = scale
        self.penalty_weight = penalty_weight
        self.penalty_anneal_iters = penalty_anneal_iters
        self.model_class = GNN

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
        if self.verbose == "progress_bar":
            iterator = tqdm(enumerate(X), desc="Converting molecules to graphs", total=len(X))
        elif self.verbose == "print_statement":
            iterator = enumerate(X)
            print("Converting molecules to graphs: preparing data for training...")
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

    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        self.model.train()
        losses = []
        losses_erm = []
        penalties = []

        for batch_idx, batch in enumerate(train_loader):
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

            if global_pbar is not None:
                global_pbar.update(1)
                global_pbar.set_postfix({
                    "Epoch": f"{epoch+1}/{self.epochs}",
                    "Batch": f"{batch_idx+1}/{len(train_loader)}",
                    "Loss": f"{loss.item():.4f}",
                    "ERM Loss": f"{loss_erm.item():.4f}",
                    "IRM Penalty": f"{penalty.item():.4f}"
                })

        return losses