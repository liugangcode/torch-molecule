import os
import json
import datetime
import tempfile
import torch
import warnings
import numpy as np

# Import typing dependencies if needed
from typing import Optional, Dict, Any

HF_METADATA = {
    "tags": [
        "torch_molecule",
        "molecular-property-prediction",
    ],
    "library_name": "torch_molecule", 
}
from ..utils.format import sanitize_config

class LocalCheckpointManager:
    """Handles saving and loading of models to and from local paths."""

    @staticmethod
    def save_model_to_local(model_instance, path: str) -> None:
        """Save model weights and configuration to a local file."""
        if not model_instance.is_fitted_:
            raise ValueError("Model must be fitted before saving.")

        if not path.endswith((".pt", ".pth")):
            raise ValueError("Save path should end with '.pt' or '.pth'.")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        model_name = os.path.splitext(os.path.basename(path))[0]

        save_dict = {
            "model_state_dict": model_instance.model.state_dict(),
            "hyperparameters": model_instance.get_params(),
            "model_name": model_name,
            "date_saved": datetime.datetime.now().isoformat(),
            "version": getattr(model_instance, "__version__", "1.0.0"),
        }
        torch.save(save_dict, path)

    @staticmethod
    def load_model_from_local(model_instance, path: str) -> None:
        """Load model weights and configuration from a local file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file found at '{path}'.")

        try:
            checkpoint = torch.load(path, map_location=model_instance.device, weights_only=False)
        except Exception as e:
            raise ValueError(f"Error loading model from {path}: {str(e)}")

        verbose = model_instance.get_params().get("verbose", False)

        required_keys = {"model_state_dict", "hyperparameters", "model_name"}
        if not all(key in checkpoint for key in required_keys):
            missing_keys = required_keys - set(checkpoint.keys())
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

        parameter_status = []
        for key, new_value in checkpoint["hyperparameters"].items():
            if hasattr(model_instance, key):
                old_value = getattr(model_instance, key)
                is_changed = (old_value != new_value)
                parameter_status.append({
                    "Parameter": key,
                    "Old Value": old_value,
                    "New Value": new_value,
                    "Status": "Changed" if is_changed else "Unchanged",
                })
                if is_changed:
                    setattr(model_instance, key, new_value)

        if parameter_status and verbose:
            print("\nHyperparameter Status:")
            print("-" * 80)
            print(f"{'Parameter':<20} {'Old Value':<20} {'New Value':<20} {'Status':<10}")
            print("-" * 80)
            # Sort so that changed parameters appear first
            parameter_status.sort(key=lambda x: (x["Status"] != "Changed", x["Parameter"]))
            for param in parameter_status:
                color = "\033[91m" if param["Status"] == "Changed" else "\033[92m"
                print(
                    f"{param['Parameter']:<20} "
                    f"{str(param['Old Value']):<20} "
                    f"{str(param['New Value']):<20} "
                    f"{color}{param['Status']}\033[0m"
                )
            print("-" * 80)
            changes_count = sum(1 for p in parameter_status if p["Status"] == "Changed")
            print(
                f"\nSummary: {changes_count} parameters changed, "
                f"{len(parameter_status) - changes_count} unchanged"
            )

        # Reinitialize
        model_instance._initialize_model(model_instance.model_class, checkpoint)
        model_instance.model_name = checkpoint["model_name"]
        model_instance.is_fitted_ = True
        model_instance.model = model_instance.model.to(model_instance.device)
        print(f"Model successfully loaded from local path: {path}")

class HuggingFaceCheckpointManager:
    """Handles saving and loading of models to and from the Hugging Face Hub."""

    @staticmethod
    def load_model_from_hf(model_instance, repo_id: str, path: str, config_filename: str = "config.json") -> None:
        """Load model from Hugging Face Hub, saving locally to `path` first."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub package is required to load from Hugging Face Hub. "
                "Install it with: pip install huggingface_hub"
            )

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            model_name = os.path.splitext(os.path.basename(path))[0]

            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_name}.pt",
                local_dir=os.path.dirname(path),
            )

            hf_hub_download(
                repo_id=repo_id,
                filename=config_filename,
                local_dir=os.path.dirname(path),
            )

            checkpoint = torch.load(downloaded_path, map_location=model_instance.device, weights_only=False)

            required_keys = {"model_state_dict", "hyperparameters", "model_name"}
            if not all(key in checkpoint for key in required_keys):
                missing_keys = required_keys - set(checkpoint.keys())
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            parameter_status = []
            for key, new_value in checkpoint["hyperparameters"].items():
                if hasattr(model_instance, key):
                    old_value = getattr(model_instance, key)
                    is_changed = (old_value != new_value)
                    parameter_status.append({
                        "Parameter": key,
                        "Old Value": old_value,
                        "New Value": new_value,
                        "Status": "Changed" if is_changed else "Unchanged",
                    })
                    if is_changed:
                        setattr(model_instance, key, new_value)

            if parameter_status:
                print("\nHyperparameter Status:")
                print("-" * 80)
                print(f"{'Parameter':<20} {'Old Value':<20} {'New Value':<20} {'Status':<10}")
                print("-" * 80)
                parameter_status.sort(key=lambda x: (x["Status"] != "Changed", x["Parameter"]))
                for param in parameter_status:
                    color = "\033[91m" if param["Status"] == "Changed" else "\033[92m"
                    print(
                        f"{param['Parameter']:<20} "
                        f"{str(param['Old Value']):<20} "
                        f"{str(param['New Value']):<20} "
                        f"{color}{param['Status']}\033[0m"
                    )
                print("-" * 80)
                changes_count = sum(1 for p in parameter_status if p["Status"] == "Changed")
                print(
                    f"\nSummary: {changes_count} parameters changed, "
                    f"{len(parameter_status) - changes_count} unchanged"
                )

            model_instance._initialize_model(model_instance.model_class, checkpoint)
            model_instance.model_name = checkpoint["model_name"]
            model_instance.is_fitted_ = True
            model_instance.model = model_instance.model.to(model_instance.device)
            print(f"Model successfully loaded from repository {repo_id}")

        except Exception as e:
            if os.path.exists(path):
                os.remove(path)  # Clean up partial downloads
            raise RuntimeError(f"Failed to download or load model from repository '{repo_id}': {str(e)}")

    @staticmethod
    def push_to_huggingface(
        model_instance,
        repo_id: str,
        task_id: str = "default",
        metadata_dict: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        commit_message: str = "Update model",
        token: Optional[str] = None,
        private: bool = False,
        config_filename: str = "config.json",
    ) -> None:
        """Push a task-specific model checkpoint to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi, create_repo, metadata_update
            from .hf import merge_task_configs, get_existing_repo_data, create_model_card
        except ImportError:
            raise ImportError(
                "huggingface_hub package is required to push to Hugging Face Hub. "
                "Install it with: pip install huggingface_hub"
            )

        if not isinstance(repo_id, str) or "/" not in repo_id:
            raise ValueError("repo_id must be in format '<username>/<model_name>'.")
        if not task_id:
            raise ValueError("task_id must be provided.")
        if not model_instance.is_fitted_:
            raise ValueError("Model must be fitted before pushing to Hugging Face Hub.")

        try:
            api = HfApi(token=token)
            repo_exists, existing_config, existing_readme = get_existing_repo_data(repo_id, token)
            create_repo(repo_id, private=private, token=token, exist_ok=True)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, f"{model_instance.model_name}.pt")
                # Use the local manager to save a copy
                LocalCheckpointManager.save_model_to_local(model_instance, local_path)

                # Prepare task-specific config
                task_config = {
                    **({"task_type": model_instance.task_type} if hasattr(model_instance, "task_type") else {}),
                    "config": sanitize_config(model_instance.get_params(deep=True)),
                    "metrics": metrics or {},
                }

                # Merge with any existing config
                base_config = (
                    {
                        "model_name": model_instance.model_name,
                        "framework": "torch_molecule",
                        "date_created": existing_config.get(
                            "date_created", datetime.datetime.now().isoformat()
                        ),
                    }
                    if not repo_exists
                    else existing_config
                )

                num_params = sum(p.numel() for p in model_instance.model.parameters())
                final_config = merge_task_configs(
                    task_id=task_id,
                    existing_config=base_config,
                    new_task_config=task_config,
                    num_params=num_params,
                )

                # Save config file
                config_path = os.path.join(tmp_dir, config_filename)
                with open(config_path, "w") as f:
                    json.dump(final_config, f, indent=2)

                # Create model card
                readme_content = create_model_card(
                    model_class=model_instance.__class__.__name__,
                    model_name=model_instance.model_name,
                    tasks_config=final_config.get("tasks", {}),
                    model_config=final_config,
                    repo_id=repo_id,
                    existing_readme=existing_readme,
                )
                readme_path = os.path.join(tmp_dir, "README.md")
                with open(readme_path, "w") as f:
                    f.write(readme_content)

                # Upload everything
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmp_dir,
                    commit_message=f"{commit_message} - Task: {task_id}",
                )

                # Update or add repository metadata
                metadata_dict = HF_METADATA if metadata_dict is None else metadata_dict
                metadata_update(repo_id=repo_id, metadata=metadata_dict, token=token, overwrite=True)

                # Print summary
                task_info = final_config["tasks"][task_id]
                print(f"Successfully pushed model for task {task_id} to {repo_id}")
                print(f"Task version: {task_info['current_version']}")
                if metrics:
                    print("Metrics:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value:.4f}")
        except Exception as e:
            raise RuntimeError(f"Failed to push to Hugging Face Hub: {str(e)}")
