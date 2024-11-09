from enum import Enum
from typing import Dict, Any, Union, List, Tuple, NamedTuple

class ParameterType(Enum):
    """Enum defining types of hyperparameters for optimization.

    Each type corresponds to a specific Optuna suggest method and parameter behavior.
    """

    CATEGORICAL = "categorical"  # Uses suggest_categorical for discrete choices
    INTEGER = "integer"  # Uses suggest_int for whole numbers
    FLOAT = "float"  # Uses suggest_float for continuous values
    LOG_FLOAT = "log_float"  # Uses suggest_float with log=True for exponential scale

class ParameterSpec(NamedTuple):
    """Specification for a hyperparameter including its type and valid range/options.

    Attributes
    ----------
    param_type : ParameterType
        The type of parameter (categorical, integer, float, log_float)
    value_range : Union[Tuple[Any, Any], List[Any]]
        For numerical parameters: (min_value, max_value)
        For categorical parameters: list of possible values
    """

    param_type: ParameterType
    value_range: Union[Tuple[Any, Any], List[Any]]


# Dictionary mapping parameter names to their types and ranges
DEFAULT_GNN_SEARCH_SPACES: Dict[str, ParameterSpec] = {
    # Model architecture parameters
    "gnn_type": ParameterSpec(
        ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]
    ),
    "norm_layer": ParameterSpec(
        ParameterType.CATEGORICAL,
        [
            "batch_norm",
            "layer_norm",
            "instance_norm",
            "graph_norm",
            "size_norm",
            "pair_norm",
        ],
    ),
    "graph_pooling": ParameterSpec(ParameterType.CATEGORICAL, ["mean", "sum", "max"]),
    "augmented_feature": ParameterSpec(ParameterType.CATEGORICAL, ["maccs,morgan", "maccs", "morgan", None]),
    # Integer-valued parameters
    "num_layer": ParameterSpec(ParameterType.INTEGER, (2, 8)),
    "emb_dim": ParameterSpec(ParameterType.INTEGER, (64, 512)),
    # Float-valued parameters with linear scale
    "drop_ratio": ParameterSpec(ParameterType.FLOAT, (0.0, 0.75)),
    "scheduler_factor": ParameterSpec(ParameterType.FLOAT, (0.1, 0.5)),
    # Float-valued parameters with log scale
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-5, 1e-2)),
    "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-8, 1e-3)),
}

def suggest_parameter(trial: Any, param_name: str, param_spec: ParameterSpec) -> Any:
    """Suggest a parameter value using the appropriate Optuna suggest method.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial object
    param_name : str
        Name of the parameter
    param_spec : ParameterSpec
        Specification of the parameter type and range

    Returns
    -------
    Any
        The suggested parameter value

    Raises
    ------
    ValueError
        If the parameter type is not recognized
    """
    if param_spec.param_type == ParameterType.CATEGORICAL:
        return trial.suggest_categorical(param_name, param_spec.value_range)

    elif param_spec.param_type == ParameterType.INTEGER:
        min_val, max_val = param_spec.value_range
        return trial.suggest_int(param_name, min_val, max_val)

    elif param_spec.param_type == ParameterType.FLOAT:
        min_val, max_val = param_spec.value_range
        return trial.suggest_float(param_name, min_val, max_val)

    elif param_spec.param_type == ParameterType.LOG_FLOAT:
        min_val, max_val = param_spec.value_range
        return trial.suggest_float(param_name, min_val, max_val, log=True)

    else:
        raise ValueError(f"Unknown parameter type: {param_spec.param_type}")
    
def parse_list_params(params_str):
    if params_str is None:
        return None
    return params_str.split(',')