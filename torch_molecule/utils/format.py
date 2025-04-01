import torch
import numpy as np
import types
import inspect

def serialize_config(obj):
    """Helper function to make config JSON serializable.
    
    Handles special cases like lambda functions, torch modules, and numpy arrays.
    
    Parameters
    ----------
    obj : Any
        The object to serialize
        
    Returns
    -------
    Any
        JSON serializable representation of the object
    """
    # Handle None
    if obj is None:
        return None
        
    # Handle lambda and regular functions
    if isinstance(obj, (types.LambdaType, types.FunctionType, types.MethodType)):
        # For lambda functions, try to get the source code
        try:
            if isinstance(obj, types.LambdaType):
                source = inspect.getsource(obj).strip()
                return {"_type": "lambda", "source": source}
            else:
                return {"_type": "function", "name": obj.__name__}
        except (IOError, TypeError):
            # Fallback for built-in functions or when source isn't available
            return {"_type": "function", "name": str(obj)}
    
    # Handle PyTorch modules and optimizers
    elif isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
        return {
            "_type": "torch_class",
            "class_name": obj.__class__.__name__,
            "module": obj.__class__.__module__
        }
        
    # Handle PyTorch tensors
    elif isinstance(obj, torch.Tensor):
        # If it's a single number wrapped in a torch tensor, just return the number
        if obj.numel() == 1:
            return obj.item()
        elif obj.numel() < 1000:
            return {
                "_type": "torch_tensor",
                "data": obj.detach().cpu().tolist(),
                "shape": list(obj.shape)
            }
        return {
            "_type": "torch_tensor",
            "shape": list(obj.shape)
        }
        
    # Handle numpy arrays
    elif isinstance(obj, (np.ndarray, np.generic)):
        # If it's a single number wrapped in a numpy array, just return the number
        if obj.size == 1:
            return obj.item()
        elif obj.size < 1000:
            return {
                "_type": "numpy_array",
                "data": obj.tolist(),
                "shape": list(obj.shape)
            }
        return {
            "_type": "numpy_array",
            "shape": list(obj.shape)
        }
        
    # Handle sets and frozensets
    elif isinstance(obj, (set, frozenset)):
        return {
            "_type": "set",
            "data": list(obj)
        }
        
    # Handle custom objects with __dict__
    elif hasattr(obj, '__dict__'):
        return {
            "_type": "custom_class",
            "class_name": obj.__class__.__name__,
            "module": obj.__class__.__module__
        }
        
    # Handle basic types that are JSON serializable
    elif isinstance(obj, (str, int, float, bool)):
        return obj
        
    # Handle any other types by converting to string
    return {
        "_type": "unknown",
        "repr": str(obj)
    }
        

def sanitize_config(config_dict):
    """Recursively sanitize config dictionary for JSON serialization.
    
    Handles nested structures and special cases.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary to sanitize
        
    Returns
    -------
    dict
        Sanitized configuration dictionary that is JSON serializable
    """
    if not isinstance(config_dict, dict):
        return serialize_config(config_dict)
        
    clean_dict = {}
    for key, value in config_dict.items():
        # Skip private attributes and callable objects stored as attributes
        if isinstance(key, str) and (key.startswith('_') or callable(value)):
            continue
            
        # Handle nested dictionaries
        if isinstance(value, dict):
            clean_dict[key] = sanitize_config(value)
            
        # Handle lists and tuples
        elif isinstance(value, (list, tuple)):
            clean_dict[key] = [sanitize_config(v) for v in value]
            
        # Handle all other types
        else:
            clean_dict[key] = serialize_config(value)
            
    return clean_dict