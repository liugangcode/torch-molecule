"""Local save/load helpers for HF pretrained generators."""

import json
import os
from typing import Any, Dict, Optional

METADATA_FILENAME = "hf_generator_metadata.json"


def build_metadata(
    repo_id: str,
    family: str,
    max_length: int,
    revision: Optional[str],
    trust_remote_code: bool,
    tokenizer_repo_id: Optional[str],
    generate_max_length: int,
    model_name: str,
) -> Dict[str, Any]:
    return {
        "repo_id": repo_id,
        "family": family,
        "max_length": max_length,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "tokenizer_repo_id": tokenizer_repo_id,
        "generate_max_length": generate_max_length,
        "model_name": model_name,
    }


def save_metadata(path: str, metadata: Dict[str, Any]) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, METADATA_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_metadata(path: str) -> Dict[str, Any]:
    metadata_path = os.path.join(path, METADATA_FILENAME)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing {METADATA_FILENAME} in '{path}'. Expected a directory saved by "
            "HFPretrainedMolecularGenerator.save_to_local()."
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
