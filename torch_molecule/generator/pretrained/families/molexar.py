"""Molexar Fragment-SELFIES generation."""

from typing import Any, Dict, List, Optional

SINGLE_FRAGMENT_TASKS = frozenset({"motif_extension", "scaffold_decoration"})
TWO_FRAGMENT_TASKS = frozenset({"linker_design", "scaffold_morphing"})
FRAGMENT_CONSTRAINED_TASKS = SINGLE_FRAGMENT_TASKS | TWO_FRAGMENT_TASKS | frozenset({"superstructure"})

PROPERTY_KEYS = (
    "mol_hac",
    "mol_hbdc",
    "mol_hbac",
    "mol_rotbc",
    "mol_wt",
    "mol_logp",
    "mol_tpsa",
    "mol_qed",
    "mol_sas",
)


def _require_molexar():
    try:
        from molexar.inference import MolexarInference  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'molexar' package is required for Molexar generation. "
            "Install it with `pip install git+https://github.com/fairydance/Molexar.git`."
        ) from exc


def resolve_start_string(
    *,
    start_string: Optional[str] = None,
    start_smiles: Optional[str] = None,
    start_fragment_selfies: Optional[str] = None,
    generation_task: Optional[str] = None,
) -> Optional[str]:
    """Resolve the Fragment-SELFIES prefix placed after ``<MOL>``."""
    if start_string is not None:
        return start_string
    if start_fragment_selfies is not None:
        return start_fragment_selfies
    if start_smiles is None:
        return None

    task = generation_task or "motif_extension"
    if task == "de_novo":
        raise ValueError("de_novo generation does not accept start_smiles")

    from molexar.data.converter import smiles_fragment_to_fragment_selfies

    if task in SINGLE_FRAGMENT_TASKS | {"superstructure"}:
        encoded = smiles_fragment_to_fragment_selfies(start_smiles, randomized=True)
        return f"{encoded}[Attach:0]"

    if task in TWO_FRAGMENT_TASKS:
        fragments = [fragment.strip() for fragment in start_smiles.split(".") if fragment.strip()]
        if len(fragments) != 2:
            raise ValueError(
                "linker_design and scaffold_morphing require exactly two "
                "dot-separated SMILES fragments"
            )
        encoded_fragments = [
            smiles_fragment_to_fragment_selfies(fragment, randomized=True) for fragment in fragments
        ]
        return "".join(encoded_fragments) + "[Attach:0]"

    raise ValueError(
        f"Unsupported generation_task '{task}'. Supported tasks: de_novo, "
        "motif_extension, scaffold_decoration, linker_design, scaffold_morphing, superstructure"
    )


def extract_conditions(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract Molexar condition kwargs into a conditions dictionary."""
    conditions = dict(kwargs.pop("conditions", {}) or {})
    for key in PROPERTY_KEYS:
        if key in kwargs:
            conditions[key] = kwargs.pop(key)
    for key in ("mol_pharma_fp", "prot_seq_esm_emb", "prot_poc_gvp_emb"):
        if key in kwargs:
            conditions[key] = kwargs.pop(key)
    return conditions


def generate_molexar(
    engine: Any,
    n_samples: int,
    *,
    start_string: Optional[str] = None,
    start_smiles: Optional[str] = None,
    start_fragment_selfies: Optional[str] = None,
    generation_task: Optional[str] = None,
    conditions: Optional[Dict[str, Any]] = None,
    max_new_tokens: Optional[int] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    batch_size: int = 100,
    **kwargs: Any,
) -> List[str]:
    """Generate Fragment-SELFIES strings with a Molexar inference engine."""
    _require_molexar()

    merged_conditions = dict(conditions or {})

    resolved_start = resolve_start_string(
        start_string=start_string,
        start_smiles=start_smiles,
        start_fragment_selfies=start_fragment_selfies,
        generation_task=generation_task,
    )

    return engine.generate(
        conditions=merged_conditions,
        start_string=resolved_start,
        max_new_tokens=max_new_tokens,
        num_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        batch_size=batch_size,
        **kwargs,
    )
