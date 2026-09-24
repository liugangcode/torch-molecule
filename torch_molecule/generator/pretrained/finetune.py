"""Fine-tuning utilities for Hugging Face pretrained generators."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .registry import MOLEXAR_FAMILIES, SEQ2SEQ_FAMILIES


def corrupt_token_ids(
    input_ids: torch.Tensor,
    tokenizer: Any,
    *,
    mask_prob: float = 0.15,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Replace a random subset of non-special tokens with the tokenizer mask id.

    MolGen is trained as a denoising seq2seq model: corrupted SELFIES in,
    clean SELFIES as labels. Special tokens (BOS/EOS/PAD/mask) are left intact.
    """
    if tokenizer.mask_token_id is None:
        raise ValueError(
            "MolGen denoising fine-tuning requires tokenizer.mask_token_id. "
            "MolGen tokenizers provide a <mask> token."
        )

    corrupted = input_ids.clone()
    special = torch.zeros_like(corrupted, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        special |= corrupted == special_id
    if tokenizer.pad_token_id is not None:
        special |= corrupted == tokenizer.pad_token_id

    probs = torch.rand(corrupted.shape, generator=generator, device=corrupted.device)
    to_mask = (probs < mask_prob) & ~special
    corrupted[to_mask] = tokenizer.mask_token_id
    return corrupted


class _TokenizedDataset(Dataset):
    def __init__(self, input_ids: List[List[int]], attention_mask: List[List[int]]):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
        }


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _run_training_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    grad_norm_clip: Optional[float],
    verbose: str,
) -> Tuple[List[float], int]:
    model.train()
    epoch_losses: List[float] = []
    last_epoch = 0

    for epoch in range(epochs):
        last_epoch = epoch
        batch_losses: List[float] = []
        iterator = train_loader
        if verbose in {"progress_bar", "print_statement"}:
            iterator = tqdm(train_loader, desc=f"Fine-tuning epoch {epoch + 1}/{epochs}")

        for batch in iterator:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)
        if verbose == "print_statement":
            print(f"Epoch {epoch + 1}/{epochs} loss: {epoch_losses[-1]:.4f}")

    model.eval()
    return epoch_losses, last_epoch


def finetune_causal_lm(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_norm_clip: Optional[float],
    verbose: str,
) -> Tuple[List[float], int]:
    """Fine-tune a causal language model with next-token prediction."""
    import transformers

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    dataset = _TokenizedDataset(tokenized["input_ids"], tokenized["attention_mask"])
    collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return _run_training_loop(model, train_loader, optimizer, device, epochs, grad_norm_clip, verbose)


def finetune_seq2seq(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_norm_clip: Optional[float],
    verbose: str,
    mask_prob: float = 0.15,
) -> Tuple[List[float], int]:
    """Fine-tune a seq2seq model with MolGen-style denoising.

    Encoder inputs are token-masked SELFIES; labels remain the original
    clean sequence. Causal LM and Molexar fine-tuning are unchanged.
    """
    import transformers

    class _Seq2SeqDataset(Dataset):
        def __init__(self, items: List[str]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            encoded = tokenizer(
                self.items[idx],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            item = {key: torch.tensor(value) for key, value in encoded.items()}
            item["labels"] = item["input_ids"].clone()
            item["input_ids"] = corrupt_token_ids(
                item["input_ids"],
                tokenizer,
                mask_prob=mask_prob,
            )
            return item

    collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    train_loader = DataLoader(
        _Seq2SeqDataset(texts),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return _run_training_loop(model, train_loader, optimizer, device, epochs, grad_norm_clip, verbose)


def finetune_molexar(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Any,
    texts: List[str],
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_norm_clip: Optional[float],
    verbose: str,
) -> Tuple[List[float], int]:
    """Fine-tune a Molexar model on Fragment-SELFIES training templates."""
    from molexar.templates import build_condition_template, build_training_text

    condition_block, _ = build_condition_template(config)
    training_texts = [build_training_text(config, condition_block, text) for text in texts]
    return finetune_causal_lm(
        model,
        tokenizer,
        training_texts,
        device,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_norm_clip=grad_norm_clip,
        verbose=verbose,
    )


def finetune_generator(
    family: str,
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
    *,
    config: Optional[Any] = None,
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_norm_clip: Optional[float],
    verbose: str,
) -> Tuple[List[float], int]:
    """Dispatch fine-tuning to the family-specific routine."""
    if family in SEQ2SEQ_FAMILIES:
        return finetune_seq2seq(
            model,
            tokenizer,
            texts,
            device,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_norm_clip=grad_norm_clip,
            verbose=verbose,
        )

    if family in MOLEXAR_FAMILIES:
        if config is None:
            raise ValueError("Molexar fine-tuning requires a model config.")
        return finetune_molexar(
            model,
            tokenizer,
            config,
            texts,
            device,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_norm_clip=grad_norm_clip,
            verbose=verbose,
        )

    return finetune_causal_lm(
        model,
        tokenizer,
        texts,
        device,
        max_length=max_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_norm_clip=grad_norm_clip,
        verbose=verbose,
    )
