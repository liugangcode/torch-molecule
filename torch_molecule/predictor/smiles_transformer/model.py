import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ...utils import init_weights

class Transformer(nn.Module):
    """Transformer-based model for molecular property prediction.
    
    Parameters
    ----------
    num_task : int
        Number of prediction tasks.
    input_dim : int
        Size of vocabulary for SMILES tokenization.
    hidden_size : int
        Dimension of hidden layers and embeddings.
    n_heads : int
        Number of attention heads in transformer layers.
    num_layers : int
        Number of transformer encoder layers.
    max_input_len : int
        Maximum length of input sequences.
    dropout : float
        Dropout rate for layers.
    dim_feedforward : int
        Dimension of the feedforward network in transformer layers.

    Attributes
    ----------
    src_emb : nn.Embedding
        Embedding layer that converts token indices to dense vectors.
    pos_emb : PositionalEncoding
        Positional encoding layer for transformer.
    layers : nn.ModuleList
        List of transformer encoder layers.
    regressor : nn.Sequential
        Final fully connected layers for prediction.

    Notes
    -----
    The model architecture consists of:
    1. An embedding layer to convert tokens to vectors
    2. Positional encoding for transformer
    3. Multiple transformer encoder layers
    4. A regression head for prediction
    """
    def __init__(self, num_task, input_dim, hidden_size, n_heads, 
                 num_layers, max_input_len, 
                 dropout, dim_feedforward=None):
        super(Transformer, self).__init__()
        
        self.num_task = num_task
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_k = self.d_v = hidden_size // n_heads
        self.num_layers = num_layers
        self.max_input_len = max_input_len
        self.dropout = dropout
        if not dim_feedforward:
            dim_feedforward = hidden_size * 4   
        self.dim_feedforward = dim_feedforward
        
        # Embedding layers
        self.src_emb = nn.Embedding(input_dim, hidden_size)
        self.pos_emb = PositionalEncoding(hidden_size, max_input_len, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, n_heads, 
                                                 self.d_k, self.d_v, 
                                                 dropout, dim_feedforward) 
                                     for _ in range(num_layers)])
        
        # Prediction head - similar to your implementation but with configurable number of tasks
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_task)
        )

    def initialize_parameters(self, seed=None):
        """Initialize model parameters randomly.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize embedding
        init_weights(self.src_emb)
        
        # Initialize layers
        for layer in self.layers:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        
        # Initialize regressor
        for module in self.regressor:
            if hasattr(module, 'weight'):
                init_weights(module)

    def compute_loss(self, batched_input, batched_label, criterion):
        """Compute the loss for a batch of data.
        
        Parameters
        ----------
        batched_input : torch.Tensor
            Batch of input sequences, shape (batch_size, seq_len).
        batched_label : torch.Tensor
            Batch of target values, shape (batch_size, num_task).
        criterion : callable
            Loss function to use.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Get predictions
        prediction = self(batched_input)["prediction"]
        
        # Convert labels to float32
        target = batched_label.to(torch.float32)
        
        # Create mask for valid labels (handles NaN values)
        is_labeled = batched_label == batched_label
        
        # Compute loss only on valid labels
        loss = criterion(prediction.to(torch.float32)[is_labeled], target[is_labeled])
        return loss

    def forward(self, batched_input):
        """Forward pass of the model.
        
        Parameters
        ----------
        batched_input : torch.Tensor
            Batch of input sequences, shape (batch_size, seq_len).

        Returns
        -------
        dict
            Dictionary containing:
                - prediction: Model predictions (shape: [batch_size, num_task])
        """
        # Create padding mask
        pad_mask = get_attn_pad_mask(batched_input, batched_input)
        
        # Embedding
        enc_outputs = self.src_emb(batched_input)
        enc_outputs = self.pos_emb(enc_outputs)
        
        # Encoder layers
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, pad_mask)
            enc_self_attns.append(enc_self_attn)
        
        # Global pooling for sequence representation
        # Use CLS token (first token) as the sequence representation
        cls_token = enc_outputs[:, 0, :]
        
        # Prediction
        prediction = self.regressor(cls_token)
        
        return {"prediction": prediction}


class PositionalEncoding(nn.Module):
    """Positional encoding layer for transformer.
    
    Parameters
    ----------
    hidden_size : int
        Dimension of hidden layers.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability.
    """
    def __init__(self, hidden_size, max_len, dropout):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding table
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # consider the situation that hidden_size is not even
        div_terms_size = hidden_size // 2 if hidden_size % 2 == 0 else (hidden_size + 1) // 2
        div_term = torch.exp(torch.arange(0, div_terms_size * 2, 2).float() * (-math.log(10000.0) / hidden_size))

        # make sure the size of pe is (max_len, hidden_size)
        if hidden_size % 2 == 1:
            div_term = div_term[:div_terms_size]
        pe[:, 0::2] = torch.sin(position * div_term[:hidden_size//2 + hidden_size%2])
        pe[:, 1::2] = torch.cos(position * div_term[:hidden_size//2])
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter but should be saved)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns
        -------
        torch.Tensor
            Tensor with added positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Transformer encoder layer.
    
    Parameters
    ----------
    hidden_size : int
        Dimension of hidden layers.
    n_heads : int
        Number of attention heads.
    d_k : int
        Dimension of key/query.
    d_v : int
        Dimension of value.
    dropout : float
        Dropout probability.
    """
    def __init__(self, hidden_size, n_heads, d_k, d_v, dropout, dim_feedforward):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size, n_heads, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(hidden_size, dim_feedforward, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_inputs, self_attn_mask):
        """
        Parameters
        ----------
        enc_inputs : torch.Tensor
            Input tensor of shape (batch_size, seq_len, hidden_size)
        self_attn_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns
        -------
        torch.Tensor, torch.Tensor
            Output tensor and attention weights
        """
        # Self-attention with residual connection and layer normalization
        attn_outputs, attn = self.self_attn(enc_inputs, enc_inputs, enc_inputs, self_attn_mask)
        attn_outputs = self.dropout(attn_outputs)
        outputs = self.layer_norm1(enc_inputs + attn_outputs)
        
        # Feed-forward with residual connection and layer normalization
        ffn_outputs = self.pos_ffn(outputs)
        ffn_outputs = self.dropout(ffn_outputs)
        outputs = self.layer_norm2(outputs + ffn_outputs)
        
        return outputs, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer.
    
    Parameters
    ----------
    hidden_size : int
        Dimension of hidden layers.
    n_heads : int
        Number of attention heads.
    d_k : int
        Dimension of key/query.
    d_v : int
        Dimension of value.
    """
    def __init__(self, hidden_size, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(hidden_size, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(hidden_size, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(hidden_size, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, hidden_size, bias=False)
        
    def forward(self, Q, K, V, attn_mask):
        """
        Parameters
        ----------
        Q, K, V : torch.Tensor
            Query, key and value tensors of shape (batch_size, seq_len, hidden_size)
        attn_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns
        -------
        torch.Tensor, torch.Tensor
            Output tensor and attention weights
        """
        batch_size = Q.size(0)
        residual = Q
        
        # Transform to multi-head queries, keys, and values
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Expand mask for multi-head attention
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Calculate attention
        context, attn = scaled_dot_product_attention(Q, K, V, attn_mask)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network.
    
    Parameters
    ----------
    d_in : int
        Input dimension.
    d_ff : int
        Hidden dimension.
    dropout : float
        Dropout probability.
    """
    def __init__(self, d_in, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_in, d_ff)
        self.fc2 = nn.Linear(d_ff, d_in)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_in)
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_in)
        """
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Calculate scaled dot-product attention.
    
    Parameters
    ----------
    Q, K, V : torch.Tensor
        Query, key and value tensors
    mask : torch.Tensor, optional
        Attention mask
        
    Returns
    -------
    torch.Tensor, torch.Tensor
        Context tensor and attention weights
    """
    # Calculate attention scores
    scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.size(-1))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    
    # Apply softmax to get attention weights
    attn = F.softmax(scores, dim=-1)
    
    # Calculate context by weighted sum of values
    context = torch.matmul(attn, V)
    
    return context, attn


def get_attn_pad_mask(seq_q, seq_k):
    """Create padding mask for attention.
    
    Parameters
    ----------
    seq_q, seq_k : torch.Tensor
        Query and key sequences
        
    Returns
    -------
    torch.Tensor
        Boolean mask where True indicates padding positions
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    
    # Create mask for padding
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    
    # Expand mask to match attention matrix dimensions
    return pad_attn_mask.expand(batch_size, len_q, len_k)
