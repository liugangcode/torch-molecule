import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from ...utils import init_weights

# Define the PyTorch-based LSTM model
class LSTM(nn.Module):
    """LSTM-based model for molecular property prediction.
    
    Parameters
    ----------
    num_task : int
        Number of prediction tasks.
    input_dim : int
        Size of vocabulary for SMILES tokenization.
    output_dim : int
        Dimension of embedding vectors.
    LSTMunits : int
        Number of hidden units in LSTM layers.
    max_input_len : int
        Maximum length of input sequences.

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer that converts token indices to dense vectors.
    lstm1 : nn.LSTM
        First bidirectional LSTM layer.
    lstm2 : nn.LSTM
        Second bidirectional LSTM layer.
    timedist_dense : nn.Linear
        Time-distributed dense layer for feature transformation.
    relu : nn.ReLU
        ReLU activation function.
    fc : nn.Linear
        Final fully connected layer for prediction.

    Notes
    -----
    The model architecture consists of:
    1. An embedding layer to convert tokens to vectors
    2. Two stacked bidirectional LSTM layers
    3. A time-distributed dense layer with ReLU activation
    4. A final fully connected layer for prediction
    """

    def __init__(self, num_task, input_dim, output_dim, LSTMunits, max_input_len):
        """
        input_dim: Vocabulary size
        output_dim: Embedding dimension
        LSTMunits: Number of hidden units in LSTM (unidirectional)
        max_input_len: Input sequence length (used for later flattening)
        """
        super(LSTM, self).__init__()
        self.num_task = num_task
        hidden_dim = int(LSTMunits / 2)  
        self.embedding = nn.Embedding(input_dim, output_dim)
        self.lstm1 = nn.LSTM(input_size=output_dim, hidden_size=LSTMunits, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=LSTMunits * 2, hidden_size=LSTMunits, bidirectional=True, batch_first=True)
        self.timedist_dense = nn.Linear(LSTMunits * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim * max_input_len, num_task)

    def initialize_parameters(self, seed=None):
        """Initialize model parameters randomly.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Initialize the main components
        init_weights(self.embedding)
        init_weights(self.lstm1)
        init_weights(self.lstm2)
        init_weights(self.timedist_dense)
        init_weights(self.relu)
        init_weights(self.fc)
        
        # Reset all parameters using PyTorch Geometric's reset function
        def reset_parameters(module):
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
            elif hasattr(module, 'weight') and hasattr(module.weight, 'data'):
                init_weights(module)
        
        self.apply(reset_parameters) 

    def compute_loss(self, batched_input, batched_label, criterion):
        """Compute the loss for a batch of data.
        
        Parameters
        ----------
        batched_input : torch.Tensor
            Batch of input sequences, shape (batch_size, seq_len).
        batched_label : torch.Tensor
            Batch of target values, shape (batch_size, 1).
        criterion : callable
            Loss function to use.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        emb = self.embedding(batched_input)                
        emb, _ = self.lstm1(emb)              
        emb, _ = self.lstm2(emb)              
        emb = self.relu(self.timedist_dense(emb))  
        emb = emb.contiguous().view(emb.size(0), -1)  
        prediction = self.fc(emb)     
        target = batched_label.to(torch.float32)   
        is_labeled = batched_label == batched_label
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
                - prediction: Model predictions (shape: [batch_size, 1])
        """
        # batched_data: (batch_size, seq_len)
        emb = self.embedding(batched_input)                   # -> (batch, seq_len, output_dim)
        emb, _ = self.lstm1(emb)                    # -> (batch, seq_len, 2*LSTMunits)
        emb, _ = self.lstm2(emb)                    # -> (batch, seq_len, 2*LSTMunits)
        emb = self.relu(self.timedist_dense(emb))   # -> (batch, seq_len, hidden_dim)
        emb = emb.contiguous().view(emb.size(0), -1)    # flatten: (batch, seq_len * hidden_dim)
        prediction = self.fc(emb)                          # -> (batch, 1)
        return {"prediction": prediction}


