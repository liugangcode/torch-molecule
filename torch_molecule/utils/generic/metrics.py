import warnings
from typing import Union, Optional
import numpy as np
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error as sk_mae
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.metrics import r2_score as sk_r2_score

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                   1 / (1 + np.exp(-x)),
                   np.exp(x) / (1 + np.exp(x)))

def roc_auc_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    average: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Calculate ROC AUC scores for multi-task binary classification, handling NaN values.

    For each task dimension, computes AUC score using only the non-NaN samples.
    Tasks with insufficient valid samples or unique labels are masked in the output.

    Parameters
    ----------
    y_true : Union[np.ndarray, list]
        True binary labels. Shape should be (n_samples, n_tasks)
    y_pred : Union[np.ndarray, list]
        Predicted probabilities. Shape should be (n_samples, n_tasks)
    average : bool, default=True
        If True, return the average ROC AUC score across all valid tasks.
        If False, return individual scores for each task (NaN for invalid tasks).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights for each instance. Shape should be (n_samples,)

    Returns
    -------
    Union[float, np.ndarray]
        If average=True, returns mean ROC AUC score across all valid tasks.
        If average=False, returns array of ROC AUC scores with NaN for invalid tasks.

    Raises
    ------
    ValueError
        If input shapes don't match or no valid tasks are found
    TypeError
        If inputs are not of correct type

    Examples
    --------
    >>> y_true = np.array([[0, 1, np.nan], [1, 0, 1], [1, np.nan, 0], [0, 0, 1]])
    >>> y_pred = np.array([[0.1, 0.8, 0.7], [0.9, 0.2, 0.8], [0.8, 0.7, 0.3], [0.2, 0.1, 0.9]])
    >>> score = roc_auc_score(y_true, y_pred)
    >>> print(f"Average ROC AUC across valid tasks: {score:.3f}")
    """
    # Convert inputs to numpy arrays if needed
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {str(e)}")

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got y_true.ndim={y_true.ndim}, y_pred.ndim={y_pred.ndim}"
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Sample weight length {sample_weight.shape[0]} != number of samples {y_true.shape[0]}"
            )

    n_tasks = y_true.shape[1]
    auc_scores = np.full(n_tasks, np.nan)  # Initialize with NaN
    valid_task_mask = np.zeros(n_tasks, dtype=bool)

    # Calculate AUC for each task
    for task_idx in range(n_tasks):
        # Get valid sample mask for this task
        valid_samples = ~np.isnan(y_true[:, task_idx])
        
        if not np.any(valid_samples):
            continue  # Skip if no valid samples
            
        task_true = y_true[valid_samples, task_idx]
        task_pred = y_pred[valid_samples, task_idx]
        
        # Get task-specific sample weights if provided
        task_weights = None
        if sample_weight is not None:
            task_weights = sample_weight[valid_samples]
        
        # Check for valid binary labels
        unique_labels = np.unique(task_true)
        if len(unique_labels) < 2:
            continue  # Skip if not enough unique labels
            
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                auc_scores[task_idx] = sk_roc_auc_score(
                    task_true,
                    task_pred,
                    sample_weight=task_weights
                )
            valid_task_mask[task_idx] = True
        except Exception:
            continue  # Skip if AUC calculation fails

    # Check if any valid tasks remain
    if not np.any(valid_task_mask):
        raise ValueError("No valid tasks found for AUC calculation")

    # Return results based on averaging preference
    if average:
        return float(np.nanmean(auc_scores))
    else:
        return auc_scores
    
def accuracy_score(y_true, logits, avergae=None, thresholds=None, task_weights=None, task_types=None):
    """
    Calculate accuracy for multiple tasks from prediction logits.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        Ground truth labels with shape (n_samples, n_tasks)
    logits : numpy.ndarray
        Prediction logits with shape (n_samples, n_tasks)
    task_types : list or None, optional
        List of task types ('binary' or 'multiclass') for each task
        If None, assumes all tasks are binary
    thresholds : numpy.ndarray or None, optional
        Classification thresholds for binary tasks with shape (n_tasks,)
        If None, uses 0.5 for all binary tasks
    task_weights : numpy.ndarray or None, optional
        Weights for each task with shape (n_tasks,)
        If None, all tasks are weighted equally
        
    Returns:
    --------
    dict
        A dictionary containing:
        - 'task_accuracies': Accuracy for each individual task
        - 'weighted_accuracy': Overall weighted accuracy across all tasks
        - 'macro_accuracy': Simple average of all task accuracies
        - 'predictions': Binary predictions after applying activation and thresholds
    
    Raises:
    -------
    ValueError
        If input shapes don't match or dimensions are incorrect
    """
    # Convert inputs to numpy arrays if needed
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {str(e)}")

    # Input validation
    if y_true.shape != logits.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != logits {logits.shape}")
    
    if len(y_true.shape) != 2:
        raise ValueError(f"Expected 2D arrays, got shape {y_true.shape}")
    
    n_samples, n_tasks = y_true.shape
    
    # Set default task types if none provided
    if task_types is None:
        task_types = ['binary'] * n_tasks
    
    if len(task_types) != n_tasks:
        raise ValueError(f"Task types length {len(task_types)} != number of tasks {n_tasks}")
    
    # Set default thresholds if none provided
    if thresholds is None:
        thresholds = np.array([0.5] * n_tasks)
    else:
        thresholds = np.array(thresholds)
        if len(thresholds) != n_tasks:
            raise ValueError(f"Thresholds length {len(thresholds)} != number of tasks {n_tasks}")
    
    # Set default weights if none provided
    if task_weights is None:
        task_weights = np.ones(n_tasks) / n_tasks
    else:
        if len(task_weights) != n_tasks:
            raise ValueError(f"Task weights length {len(task_weights)} != number of tasks {n_tasks}")
        # Normalize weights to sum to 1
        task_weights = np.array(task_weights) / np.sum(task_weights)
    
    # Initialize predictions array
    y_pred = np.zeros_like(logits)
    
    # Process each task
    for task_idx in range(n_tasks):
        task_type = task_types[task_idx]
        task_logits = logits[:, task_idx]
        
        if task_type == 'binary':
            # Apply sigmoid activation
            probabilities = 1 / (1 + np.exp(-task_logits))
            # Apply threshold
            y_pred[:, task_idx] = (probabilities >= thresholds[task_idx]).astype(int)
        elif task_type == 'multiclass':
            # For multiclass, assume logits are already proper shape and just take argmax
            y_pred[:, task_idx] = np.argmax(task_logits, axis=-1)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    # Calculate accuracy for each task
    task_accuracies = np.mean(y_true == y_pred, axis=0)
    
    # Calculate weighted average accuracy
    weighted_accuracy = np.sum(task_accuracies * task_weights)
    
    # Calculate macro accuracy (simple average)
    if avergae:
        return float(np.mean(task_accuracies))
    else:
        return weighted_accuracy


def mean_absolute_error(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    average: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Calculate Mean Absolute Error for multi-task regression, handling NaN values.

    Parameters
    ----------
    y_true : Union[np.ndarray, list]
        Ground truth values. Shape should be (n_samples, n_tasks)
    y_pred : Union[np.ndarray, list]
        Predicted values. Shape should be (n_samples, n_tasks)
    average : bool, default=True
        If True, return the average MAE across all valid tasks.
        If False, return individual MAE for each task (NaN for invalid tasks).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights. Shape should be (n_samples,)

    Returns
    -------
    Union[float, np.ndarray]
        If average=True, returns mean MAE across all valid tasks.
        If average=False, returns array of MAE scores with NaN for invalid tasks.
    """
    # Convert inputs to numpy arrays if needed
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {str(e)}")

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got y_true.ndim={y_true.ndim}, y_pred.ndim={y_pred.ndim}"
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Sample weight length {sample_weight.shape[0]} != number of samples {y_true.shape[0]}"
            )

    n_tasks = y_true.shape[1]
    mae_scores = np.full(n_tasks, np.nan)
    valid_task_mask = np.zeros(n_tasks, dtype=bool)

    # Calculate MAE for each task
    for task_idx in range(n_tasks):
        # Get valid sample mask for this task
        valid_samples = ~np.isnan(y_true[:, task_idx]) & ~np.isnan(y_pred[:, task_idx])
        
        if not np.any(valid_samples):
            continue  # Skip if no valid samples
            
        task_true = y_true[valid_samples, task_idx]
        task_pred = y_pred[valid_samples, task_idx]
        
        # Get task-specific sample weights if provided
        task_weights = None
        if sample_weight is not None:
            task_weights = sample_weight[valid_samples]
        
        try:
            mae_scores[task_idx] = sk_mae(
                task_true,
                task_pred,
                sample_weight=task_weights
            )
            valid_task_mask[task_idx] = True
        except Exception:
            continue

    # Check if any valid tasks remain
    if not np.any(valid_task_mask):
        raise ValueError("No valid tasks found for MAE calculation")

    # Return results based on averaging preference
    if average:
        return float(np.nanmean(mae_scores))
    else:
        return mae_scores

def root_mean_squared_error(y_true, y_pred, average, sample_weight):
    return mean_squared_error(y_true, y_pred, average, sample_weight, squared=False)

def mean_squared_error(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    average: bool = True,
    sample_weight: Optional[np.ndarray] = None,
    squared: bool = True
) -> Union[float, np.ndarray]:
    """Calculate Mean Squared Error for multi-task regression, handling NaN values.

    Parameters
    ----------
    y_true : Union[np.ndarray, list]
        Ground truth values. Shape should be (n_samples, n_tasks)
    y_pred : Union[np.ndarray, list]
        Predicted values. Shape should be (n_samples, n_tasks)
    average : bool, default=True
        If True, return the average MSE across all valid tasks.
        If False, return individual MSE for each task (NaN for invalid tasks).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights. Shape should be (n_samples,)
    squared : bool, default=True
        If True, returns MSE value.
        If False, returns RMSE value.

    Returns
    -------
    Union[float, np.ndarray]
        If average=True, returns mean MSE/RMSE across all valid tasks.
        If average=False, returns array of MSE/RMSE scores with NaN for invalid tasks.
    """
    # Convert inputs to numpy arrays if needed
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {str(e)}")

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got y_true.ndim={y_true.ndim}, y_pred.ndim={y_pred.ndim}"
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Sample weight length {sample_weight.shape[0]} != number of samples {y_true.shape[0]}"
            )

    n_tasks = y_true.shape[1]
    mse_scores = np.full(n_tasks, np.nan)
    valid_task_mask = np.zeros(n_tasks, dtype=bool)

    # Calculate MSE for each task
    for task_idx in range(n_tasks):
        # Get valid sample mask for this task
        valid_samples = ~np.isnan(y_true[:, task_idx]) & ~np.isnan(y_pred[:, task_idx])
        
        if not np.any(valid_samples):
            continue  # Skip if no valid samples
            
        task_true = y_true[valid_samples, task_idx]
        task_pred = y_pred[valid_samples, task_idx]
        
        # Get task-specific sample weights if provided
        task_weights = None
        if sample_weight is not None:
            task_weights = sample_weight[valid_samples]
        
        try:
            mse_scores[task_idx] = sk_mse(
                task_true,
                task_pred,
                sample_weight=task_weights,
            )
            valid_task_mask[task_idx] = True
        except Exception:
            continue

    # Check if any valid tasks remain
    if not np.any(valid_task_mask):
        raise ValueError("No valid tasks found for MSE calculation")

    # Convert to RMSE if requested
    if not squared:
        mse_scores = np.sqrt(mse_scores)

    # Return results based on averaging preference
    if average:
        return float(np.nanmean(mse_scores))
    else:
        return mse_scores

def r2_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    average: bool = True,
    sample_weight: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """Calculate R² Score for multi-task regression, handling NaN values.

    Parameters
    ----------
    y_true : Union[np.ndarray, list]
        Ground truth values. Shape should be (n_samples, n_tasks)
    y_pred : Union[np.ndarray, list]
        Predicted values. Shape should be (n_samples, n_tasks)
    average : bool, default=True
        If True, return the average R² across all valid tasks.
        If False, return individual R² for each task (NaN for invalid tasks).
    sample_weight : Optional[np.ndarray], default=None
        Sample weights. Shape should be (n_samples,)

    Returns
    -------
    Union[float, np.ndarray]
        If average=True, returns mean R² across all valid tasks.
        If average=False, returns array of R² scores with NaN for invalid tasks.
    """
    # Convert inputs to numpy arrays
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Could not convert inputs to numpy arrays: {str(e)}")

    # Validate input shapes
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )

    if y_true.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got y_true.ndim={y_true.ndim}, y_pred.ndim={y_pred.ndim}"
        )

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Sample weight length {sample_weight.shape[0]} != number of samples {y_true.shape[0]}"
            )

    n_tasks = y_true.shape[1]
    r2_scores = np.full(n_tasks, np.nan)
    valid_task_mask = np.zeros(n_tasks, dtype=bool)

    # Calculate R² for each task
    for task_idx in range(n_tasks):
        # Get valid sample mask for this task
        valid_samples = ~np.isnan(y_true[:, task_idx]) & ~np.isnan(y_pred[:, task_idx])
        
        if not np.any(valid_samples):
            continue  # Skip if no valid samples
            
        task_true = y_true[valid_samples, task_idx]
        task_pred = y_pred[valid_samples, task_idx]
        
        # Get task-specific sample weights if provided
        task_weights = None
        if sample_weight is not None:
            task_weights = sample_weight[valid_samples]
        
        try:
            r2_scores[task_idx] = sk_r2_score(
                task_true,
                task_pred,
                sample_weight=task_weights
            )
            valid_task_mask[task_idx] = True
        except Exception:
            continue

    # Check if any valid tasks remain
    if not np.any(valid_task_mask):
        raise ValueError("No valid tasks found for R² calculation")

    # Return results based on averaging preference
    if average:
        return float(np.nanmean(r2_scores))
    else:
        return r2_scores
    