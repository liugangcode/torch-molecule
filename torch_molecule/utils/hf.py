import os
import json
import datetime
from typing import Dict, Tuple, Optional, List, Union
from huggingface_hub import hf_hub_download

def merge_task_configs(
    task_id: str,
    existing_config: Dict,
    new_task_config: Dict,
    num_params: int
) -> Dict:
    """Merge task-specific configuration and maintain version history.
    
    Parameters
    ----------
    task_id : str
        Task identifier (e.g., 'O2', 'N2')
    existing_config : Dict
        Existing configuration dictionary
    new_task_config : Dict
        New task configuration to merge
    num_params : int
        Number of model parameters
        
    Returns
    -------
    Dict
        Updated configuration with task history
    """
    merged = existing_config.copy()
    tasks_config = merged.get("tasks", {})
    task_config = tasks_config.get(task_id, {
        "versions": [],
        "current_version": "0.0.0",
        "date_added": datetime.datetime.now().isoformat(),
    })
    
    # Update task version history
    current_version = task_config["current_version"]
    version_parts = current_version.split('.')
    version_parts[-1] = str(int(version_parts[-1]) + 1)
    new_version = '.'.join(version_parts)
    
    # Archive current version if it exists
    if task_config["versions"]:
        task_config["versions"].append({
            "version": current_version,
            "date": datetime.datetime.now().isoformat(),
            "config": task_config.copy()
        })
    
    # Update task config
    task_config.update({
        "current_version": new_version,
        "last_updated": datetime.datetime.now().isoformat(),
        "num_parameters": num_params,
        **new_task_config
    })
    
    # Update tasks configuration
    tasks_config[task_id] = task_config
    merged["tasks"] = tasks_config
    merged["last_updated"] = datetime.datetime.now().isoformat()
    
    return merged

def get_existing_repo_data(repo_id: str, token: Optional[str] = None) -> Tuple[bool, Dict, str]:
    """Get existing repository data from HuggingFace Hub.
    
    Parameters
    ----------
    repo_id : str
        Repository ID
    token : Optional[str]
        HuggingFace token
        
    Returns
    -------
    Tuple[bool, Dict, str]
        Tuple containing (repo_exists, existing_config, existing_readme)
    """
    repo_exists = False
    existing_config = {}
    existing_readme = ""
    
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            token=token
        )
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
        repo_exists = True
        
        try:
            readme_path = hf_hub_download(
                repo_id=repo_id,
                filename="README.md",
                token=token
            )
            with open(readme_path, 'r') as f:
                existing_readme = f.read()
        except Exception:
            pass
            
    except Exception:
        pass
        
    return repo_exists, existing_config, existing_readme

def create_model_card(
    model_class: str,
    model_name: str,
    tasks_config: Dict,
    model_config: Dict,
    repo_id: str,
    existing_readme: str = ""
) -> str:
    """Create a model card for multiple tasks.
    
    Parameters
    ----------
    model_class : str
        Class name of the model
    model_name : str
        Name of the model
    tasks_config : Dict
        Configuration for all tasks
    model_config : Dict
        General model configuration
    repo_id : str
        Repository ID
    existing_readme : str
        Existing README content
        
    Returns
    -------
    str
        Generated model card content
    """
    # Create tasks summary table
    tasks_table = "| Task | Version | Last Updated | Parameters | Metrics |\n"
    tasks_table += "|------|---------|--------------|------------|----------|\n"
    
    for task_id, task_info in sorted(tasks_config.items()):
        metrics = task_info.get("metrics", {})
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        tasks_table += (
            f"| {task_id} | {task_info['current_version']} | "
            f"{task_info['last_updated'][:10]} | "
            f"{task_info['num_parameters']:,} | {metrics_str} |\n"
        )

# - molecular-property-prediction
    readme_content = f"""---
tags:
- torch_molecule
library_name: torch_molecule
---

# {model_class} Model

## Model Description
- **Model Type**: {model_class}
- **Framework**: torch_molecule
- **Last Updated**: {model_config.get('last_updated', datetime.datetime.now().isoformat())[:10]}

## Task Summary
{tasks_table}

## Usage

```python
from torch_molecule import {model_class}

# Load model for specific task
model = {model_class}()
model.load(
    "local_model_dir/{model_name}.pt",
    repo="{repo_id}"
)

# For predictor: Make predictions
# predictions = model.predict(smiles_list)
# For generator: Make generations
# generations = model.generate(n_samples)
# For encoder: Make encodings
# encodings = model.encode(smiles_list)
```

## Tasks Details

"""
    # Add detailed task sections (latest version only, excluding fitting_loss)
    for task_id, task_info in sorted(tasks_config.items()):
        # Filter out fitting_loss from config
        task_config = task_info.get('config', {}).copy()
        if 'fitting_loss' in task_config:
            del task_config['fitting_loss']
        
        readme_content += f"""
### {task_id} Task
- **Current Version**: {task_info['current_version']}
- **Last Updated**: {task_info['last_updated'][:10]}
- **Parameters**: {task_info['num_parameters']:,}
- **Configuration**:
```python
{json.dumps(task_config, indent=2)}
```
"""

    # Preserve custom sections from existing README
    # if existing_readme:
    #     try:
    #         custom_sections = ""
    #         split_points = ["## Tasks Details", "## Usage"]
    #         for split_point in split_points:
    #             if split_point in existing_readme:
    #                 parts = existing_readme.split(split_point)
    #                 if len(parts) > 1 and "##" in parts[1]:
    #                     custom_sections = parts[1].split("##", 1)[1]
    #                     custom_sections = "##" + custom_sections
    #                     break
            
    #         if custom_sections:
    #             readme_content += f"\n{custom_sections}"
    #     except Exception as e:
    #         print(f"Warning: Error preserving custom README sections: {e}")

    return readme_content