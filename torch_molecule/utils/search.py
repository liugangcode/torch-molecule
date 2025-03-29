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
    """Specification for a hyperparameter including its type and valid range/options."""

    param_type: ParameterType
    value_range: Union[Tuple[Any, Any], List[Any]]

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