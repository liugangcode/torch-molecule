import warnings
from tqdm import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List, Literal
from dataclasses import dataclass, field

import torch
import numpy as np

from ...base import BaseMolecularEncoder
    
known_repos = [
    "entropy/gpt2_zinc_87m", 
    "entropy/roberta_zinc_480m", 
    "ncfrey/ChemGPT-1.2B", 
    "ncfrey/ChemGPT-19M", 
    "ncfrey/ChemGPT-4.7M",
    "DeepChem/ChemBERTa-77M-MTR", 
    "DeepChem/ChemBERTa-77M-MLM",
    "DeepChem/ChemBERTa-10M-MTR", 
    "DeepChem/ChemBERTa-10M-MLM",
    "DeepChem/ChemBERTa-5M-MLM", 
    "DeepChem/ChemBERTa-5M-MTR",
    "unikei/bert-base-smiles",
    'seyonec/ChemBERTa-zinc-base-v1'
]

known_add_bos_eos_list = ["entropy/gpt2_zinc_87m"]

@dataclass(init=False)
class HFPretrainedMolecularEncoder(BaseMolecularEncoder):
    """Implements Hugging Face pretrained transformer models as molecular encoders.

    This class provides an interface to use pretrained transformer models from Hugging Face
    as molecular encoders. It handles tokenization and encoding of molecular representations.

    Tested models include:

    - ChemGPT series (1.2B/19M/4.7M): GPT-Neo based models pretrained on PubChem10M dataset with SELFIES strings.
      Output dimension: 2048.

      repo_id: "ncfrey/ChemGPT-1.2B" (https://huggingface.co/ncfrey/ChemGPT-1.2B)

      repo_id: "ncfrey/ChemGPT-19M" (https://huggingface.co/ncfrey/ChemGPT-19M)
      
      repo_id: "ncfrey/ChemGPT-4.7M" (https://huggingface.co/ncfrey/ChemGPT-4.7M)

    - GPT2-ZINC-87M: GPT-2 based model (87M parameters) pretrained on ZINC dataset with ~480M SMILES strings.
      Output dimension: 768.

      repo_id: "entropy/gpt2_zinc_87m" (https://huggingface.co/entropy/gpt2_zinc_87m)
    
    - RoBERTa-ZINC-480M: RoBERTa based model (102M parameters) pretrained on ZINC dataset with ~480M SMILES strings.
      Output dimension: 768.

      repo_id: "entropy/roberta_zinc_480m" (https://huggingface.co/entropy/roberta_zinc_480m)

    - ChemBERTa series: Available in multiple sizes (77M/10M/5M) and training objectives (MTR/MLM).
      Output dimension: 384.

      repo_id: "DeepChem/ChemBERTa-77M-MTR" (https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
      
      repo_id: "DeepChem/ChemBERTa-77M-MLM" (https://huggingface.co/DeepChem/ChemBERTa-77M-MLM)
      
      repo_id: "DeepChem/ChemBERTa-10M-MTR" (https://huggingface.co/DeepChem/ChemBERTa-10M-MTR)
      
      repo_id: "DeepChem/ChemBERTa-10M-MLM" (https://huggingface.co/DeepChem/ChemBERTa-10M-MLM)
      
      repo_id: "DeepChem/ChemBERTa-5M-MLM" (https://huggingface.co/DeepChem/ChemBERTa-5M-MLM)

      repo_id: "DeepChem/ChemBERTa-5M-MTR" (https://huggingface.co/DeepChem/ChemBERTa-5M-MTR)

    - UniKi/bert-base-smiles: BERT model pretrained on SMILES strings.
      Output dimension: 768.

      repo_id: "unikei/bert-base-smiles" (https://huggingface.co/unikei/bert-base-smiles)

    - ChemBERTa-zinc-base-v1: RoBERTa model pretrained on ZINC dataset with ~100k SMILES strings.
      Output dimension: 384.

      repo_id: "seyonec/ChemBERTa-zinc-base-v1" (https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

    Other models accessible through the transformers library have not been explicitly tested but may still be compatible with this interface.

    Parameters
    ----------
    repo_id : str
        The Hugging Face repository ID of the pretrained model.
    max_length : int, default=128
        Maximum sequence length for tokenization. Longer sequences will be truncated.
    batch_size : int, default=128
        Batch size used when encoding multiple molecules.
    add_bos_eos : Optional[bool], default=None
        Whether to add beginning/end of sequence tokens. If None, models in known_add_bos_eos_list will be set to True.
        The current known_add_bos_eos_list includes: ["entropy/gpt2_zinc_87m"].
    model_name : str, default="PretrainedMolecularEncoder"
        Name identifier for the model instance.
    verbose : bool, default=False
        Whether to display progress information during encoding.
    """

    repo_id: str

    # Default arguments
    max_length: int = 128
    batch_size: int = 128
    add_bos_eos: Optional[bool] = None
    model_name: str = "PretrainedMolecularEncoder"
    verbose: bool = False

    def __init__(self, repo_id: str, max_length: int = 128, batch_size: int = 128, add_bos_eos: Optional[bool] = None,
                 model_name: str = "PretrainedMolecularEncoder", verbose: bool = False, **kwargs):
        self.repo_id = repo_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.add_bos_eos = add_bos_eos
        self.model_name = model_name
        self.verbose = verbose
        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        self._require_transformers()
        self.fitting_epoch = -1
        self.fitting_loss = -1

        if self.repo_id not in known_repos:
            warnings.warn(f"Unknown repo_id: {self.repo_id}. The class will try to load the model from HuggingFace, but it might fail.")

    @staticmethod
    def _get_param_names() -> List[str]:
        """Get parameter names for the estimator.

        Returns
        -------
        List[str]
            List of parameter names that can be used for model configuration.
        """
        return ["repo_id", "max_length", "model_name", "add_bos_eos"]

    def _get_model_params(self) -> Dict[str, Any]:
        raise NotImplementedError("PretrainedMolecularEncoder does not support model parameters.")
    
    def _setup_optimizers(self) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support training.")

    def _train_epoch(self) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support training.")
    
    def save_to_local(self) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support saving to local.")
    
    def load_from_local(self) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support loading from local.")
    
    def save_to_hf(self) -> None:
        raise NotImplementedError("PretrainedMolecularEncoder does not support saving to huggingface.")
    
    def load_from_hf(self) -> None:
        """The same as fit()"""
        self.fit()
    
    def load(self) -> None:
        """The same as fit()"""
        self.fit()

    def fit(self) -> "HFPretrainedMolecularEncoder":
        """Load the pretrained model from HuggingFace."""
        assert self.repo_id is not None, "repo_id is not set"
        self._require_transformers()
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.repo_id, max_length=self.max_length)
        self.model = transformers.AutoModel.from_pretrained(self.repo_id)
        self.model.to(self.device)

        model_config = self.model.config
        model_type = model_config.model_type

        if self.add_bos_eos is None:
            self.add_bos_eos = self.repo_id in known_add_bos_eos_list

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if self.add_bos_eos:
            if self.tokenizer.bos_token is None:
                self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({'eos_token': '[EOS]'})
            self.model.resize_token_embeddings(len(self.tokenizer))

            warnings.warn("BOS and EOS tokens are not found in the tokenizer. They are added to the tokenizer since add_bos_eos is set to True.")

        self.collator = transformers.DataCollatorWithPadding(self.tokenizer, padding=True, return_tensors='pt')
        self.is_fitted_ = True        
        return self

    def encode(self, X: List[str], return_type: Literal["np", "pt"] = "pt") -> Union[np.ndarray, torch.Tensor]:
        """Encode molecules into vector representations.

        Parameters
        ----------
        X : List[str]
            List of SMILES strings
        return_type : Literal["np", "pt"], default="pt"
            Return type of the representations

        Returns
        -------
        representations : ndarray or torch.Tensor
            Molecular representations
        """
        self._require_transformers()
        self._check_is_fitted()
        X, _ = self._validate_inputs(X, return_rdkit_mol=False)
        
        # Process in batches
        all_embeddings = []
        iterator = tqdm(range(0, len(X), self.batch_size), desc="Encoding molecules", total=len(X) // self.batch_size, disable=not self.verbose)
        for i in iterator:
            batch_X = X[i:i + self.batch_size]
            
            if self.add_bos_eos:
                # For decoding models (e.g. GPT2), manually add BOS and EOS tokens
                processed_batch = [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in batch_X]
                inputs = self.collator(self.tokenizer(processed_batch))
            else:
                inputs = self.collator(self.tokenizer(batch_X))
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # get all attributes of outputs
            print('outputs', outputs.keys())
            # Extract embeddings based on model type
            if hasattr(outputs, 'hidden_states'):
                # For models that return a named tuple
                full_embeddings = outputs.hidden_states[-1]
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                # For models that return a tuple with hidden states
                full_embeddings = outputs[-1][-1]
            else:
                # For models that return last_hidden_state directly
                full_embeddings = outputs.last_hidden_state
            
            # Apply attention mask to get meaningful embeddings
            mask = inputs['attention_mask']
            batch_embeddings = ((full_embeddings * mask.unsqueeze(-1)).sum(1) / 
                          mask.sum(-1).unsqueeze(-1))
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings if return_type == "pt" else embeddings.cpu().numpy()

    @staticmethod
    def _require_transformers():
        try:
            import transformers  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for PretrainedMolecularEncoder. "
                "Please install it using `pip install transformers`."
            )