"""SAFE conversion and tokenization for GenMol."""

import json
import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from rdkit import Chem


def _require_safe():
    try:
        import safe
    except ImportError as exc:
        raise ImportError(
            "GenMol requires its optional dependencies. "
            'Install them with `pip install "torch-molecule[genmol]"`. '
        ) from exc
    return safe


def encode_smiles(smiles: str, fragment: bool = False) -> str:
    """Convert a molecule or an attachment-bearing fragment to SAFE."""
    safe = _require_safe()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}.")
    converter = (
        safe.SAFEConverter(slicer=None) if fragment else safe.SAFEConverter(ignore_stereo=True)
    )
    try:
        encoded = converter.encoder(smiles, allow_empty=True)
    except (safe.SAFEEncodeError, safe.SAFEFragmentationError, ValueError) as exc:
        raise ValueError(f"SMILES cannot be converted to SAFE: {smiles!r}.") from exc
    if not encoded:
        raise ValueError(f"SMILES cannot be converted to SAFE: {smiles!r}.")
    return encoded


def decode_safe(value: str, fix: bool = True) -> Optional[str]:
    """Decode SAFE, optionally discarding fragments that cannot be repaired."""
    safe = _require_safe()
    if not value:
        return None
    if fix:
        value = ".".join(
            fragment
            for fragment in value.split(".")
            if safe.decode(fragment, ignore_errors=True) is not None
        )
    decoded = safe.decode(value, canonical=True, ignore_errors=True)
    return decoded or None


def encode_scaffold(scaffold: str) -> str:
    """Prepare a SAFE prefix for native GenMol fragment completion."""
    safe = _require_safe()
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid scaffold SMILES: {scaffold!r}.")
    if "*" not in scaffold:
        cores = safe.utils.list_individual_attach_points(mol, depth=3)
        if not cores:
            raise ValueError("Scaffold has no available attachment sites.")
        scaffold = random.choice(cores)
    return encode_smiles(scaffold) + "."


def filter_by_substructure(sequences: List[str], fragment: str) -> List[str]:
    """Keep molecules containing the specified fragment without its attachment atoms."""
    safe = _require_safe()
    fragment = safe.utils.standardize_attach(fragment)
    query = Chem.MolFromSmarts(fragment)
    if query is None:
        raise ValueError(f"Invalid fragment: {fragment!r}.")
    query = Chem.DeleteSubstructs(query, Chem.MolFromSmiles("*"))
    if query.GetNumAtoms() == 0:
        raise ValueError("Fragment must contain at least one non-attachment atom.")
    query = Chem.MolFromSmarts(Chem.MolToSmiles(query))
    return safe.utils.filter_by_substructure_constraints(sequences, query, n_jobs=1)


class GenMolTokenizer:
    """The fixed GenMol V1 vocabulary and SAFE pre-tokenizer."""

    def __init__(self, tokenizer_json: Optional[str] = None):
        _require_safe()
        from safe.tokenizer import SAFETokenizer

        self._tokenizer_json = (
            (Path(__file__).with_name("assets") / "tokenizer.json").read_text(encoding="utf-8")
            if tokenizer_json is None
            else tokenizer_json
        )
        tokenizer = SAFETokenizer.from_dict(json.loads(self._tokenizer_json))
        self.backend = tokenizer.get_pretrained()
        self._splitter = tokenizer.splitter
        for name in ("bos", "eos", "pad", "mask", "unk"):
            setattr(self, f"{name}_token_id", getattr(self.backend, f"{name}_token_id"))
        self.vocab_size = self.backend.vocab_size

    def __len__(self) -> int:
        return len(self.backend)

    def to_json(self) -> str:
        """Return the vocabulary and SAFE splitting configuration for checkpointing."""
        return self._tokenizer_json

    def encode(
        self,
        value: str,
        max_len: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> List[int]:
        if not isinstance(value, str) or not value:
            raise ValueError("SAFE input must be a nonempty string.")
        try:
            self._splitter.tokenize(value)
        except ValueError as exc:
            raise ValueError("SAFE input contains unsupported characters.") from exc
        ids = self.backend.encode(value, add_special_tokens=add_special_tokens, truncation=False)
        if self.unk_token_id in ids:
            raise ValueError("SAFE input contains a token outside the GenMol vocabulary.")
        if max_len is not None and len(ids) > max_len:
            raise ValueError(f"Molecule needs {len(ids)} tokens, exceeding max_len={max_len}.")
        return ids

    def pad(self, sequences: Sequence[Sequence[int]]) -> dict:
        """Right-pad token sequences and return their attention mask."""
        lengths = [len(sequence) for sequence in sequences]
        input_ids = torch.full((len(sequences), max(lengths)), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : lengths[index]] = torch.as_tensor(sequence, dtype=torch.long)
            attention_mask[index, : lengths[index]] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, ids, skip_special_tokens: bool = True) -> List[str]:
        return self.backend.batch_decode(ids, skip_special_tokens=skip_special_tokens)
