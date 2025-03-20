import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from ...utils import init_weights

# Define the PyTorch-based LSTM model
class LSTM(nn.Module):
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
        self.fc = nn.Linear(hidden_dim * max_input_len, 1)

    def initialize_parameters(self, seed=None):
        """
        Randomly initialize all model parameters using the init_weights function.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
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
        # batched_data: (batch_size, seq_len)
        emb = self.embedding(batched_input)                   # -> (batch, seq_len, output_dim)
        emb, _ = self.lstm1(emb)                    # -> (batch, seq_len, 2*LSTMunits)
        emb, _ = self.lstm2(emb)                    # -> (batch, seq_len, 2*LSTMunits)
        emb = self.relu(self.timedist_dense(emb))   # -> (batch, seq_len, hidden_dim)
        emb = emb.contiguous().view(emb.size(0), -1)    # flatten: (batch, seq_len * hidden_dim)
        prediction = self.fc(emb)                          # -> (batch, 1)
        return {"prediction": prediction}


