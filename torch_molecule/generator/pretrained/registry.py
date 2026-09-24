"""Family registry for Hugging Face pretrained molecular generators."""

from typing import Dict

KNOWN_REPOS: Dict[str, str] = {
    "chandar-lab/NovoMolGen_32M_SMILES_BPE": "novomolgen",
    "zjunlp/MolGen-large": "molgen",
    "zjunlp/MolGen-large-opt": "molgen",
    "fairydance/molexar-10m-base": "molexar",
    "fairydance/molexar-10m-omni": "molexar",
    "datamol-io/safe-gpt": "safe_gpt",
}

FAMILY_PREFIXES: Dict[str, str] = {
    "chandar-lab/NovoMolGen": "novomolgen",
    "zjunlp/MolGen": "molgen",
    "fairydance/molexar": "molexar",
    "datamol-io/safe": "safe_gpt",
}

MOLEXAR_OMNI_REPOS = frozenset(
    {
        "fairydance/molexar-10m-omni",
    }
)

CAUSAL_LM_FAMILIES = frozenset({"novomolgen", "safe_gpt", "causal_lm"})
SEQ2SEQ_FAMILIES = frozenset({"molgen"})
MOLEXAR_FAMILIES = frozenset({"molexar"})
SAFE_GPT_FAMILIES = frozenset({"safe_gpt"})


def resolve_family(repo_id: str) -> str:
    """Map a Hugging Face repo id to a generator family."""
    if repo_id in KNOWN_REPOS:
        return KNOWN_REPOS[repo_id]

    for prefix, family in FAMILY_PREFIXES.items():
        if repo_id.startswith(prefix):
            return family

    return "causal_lm"
