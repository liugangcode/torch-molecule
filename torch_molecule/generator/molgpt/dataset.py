from torch.utils.data import Dataset
import torch

class SmilesDataset(Dataset):
    """Dataset for SMILES strings for MolGPT training"""
    
    def __init__(self, data, regex, max_len, properties=None, 
                 scaffolds=None, scaffold_maxlen=None):
        """
        Initialize the dataset.
        
        Parameters
        ----------
        data : List[str]
            List of SMILES strings
        regex : str
            Regex for tokenization
        max_len : int
            Maximum sequence length
        properties : Optional[List[List[float]]]
            Property values for conditional generation
        scaffolds : Optional[List[str]]
            Scaffold SMILES for conditional generation
        scaffold_maxlen : Optional[int]
            Maximum length of scaffold SMILES
        """
        
        # Generate content by decomposing all SMILES strings
        content = set()
        for smile in data:
            tokens = regex.findall(smile.strip())
            content.update(tokens)
        
        # Add padding token
        content.add('<')
        
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.data = data
        self.properties = properties
        self.scaffolds = scaffolds
        self.scaffold_maxlen = scaffold_maxlen
        self.regex = regex
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get SMILES and property
        smiles = self.data[idx].strip()
        prop = self.properties[idx] if self.properties is not None else [0.0]
        
        # Get scaffold if available
        if self.scaffolds is not None:
            scaffold = self.scaffolds[idx].strip()
        else:
            scaffold = ""
        
        # Tokenize SMILES
        tokens = self.regex.findall(smiles)
        # Pad with '<' if needed
        tokens += ['<'] * (self.max_len - len(tokens))
        # Convert to indices
        indices = [self.stoi.get(t, 0) for t in tokens]
        
        # Create input and target tensors
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        
        # Create property tensor
        prop_tensor = torch.tensor(prop, dtype=torch.float)
        
        # Create scaffold tensor if needed
        if self.scaffolds is not None:
            scaffold_tokens = self.regex.findall(scaffold)
            scaffold_tokens += ['<'] * (self.scaffold_maxlen - len(scaffold_tokens))
            scaffold_indices = [self.stoi.get(t, 0) for t in scaffold_tokens]
            scaffold_tensor = torch.tensor(scaffold_indices, dtype=torch.long)
        else:
            scaffold_tensor = torch.tensor([], dtype=torch.long)
        
        return x, y, prop_tensor, scaffold_tensor