import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_task, input_size, hidden_size, output_size, num_layer, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.dropout = dropout
        if num_task == 0:
            self.input_dim = 1
        else:
            self.input_dim = num_task
        self.hidden_transform = nn.Linear(self.input_dim, num_layer * hidden_size)
        self.cell_transform = nn.Linear(self.input_dim, num_layer * hidden_size)
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=num_layer, dropout=dropout)
        self.initialize_parameters()

    def initialize_parameters(self):
        # encoder / decoder
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)
        # RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # LSTM remember gate bias should be initialised to 1
                # https://github.com/pytorch/pytorch/issues/750
                r_gate = param[int(0.25 * len(param)):int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, input, hidden, cell):
        embeds = self.encoder(input)
        output, (hidden, cell) = self.rnn(embeds, (hidden, cell))
        output = self.decoder(output)
        return output, hidden, cell

    def init_hidden(self, bsz, target):
        hidden = self.hidden_transform(target)
        cell = self.cell_transform(target)
        hidden = hidden.view(self.num_layer, bsz, self.hidden_size)
        cell = cell.view(self.num_layer, bsz, self.hidden_size)
        return hidden, cell
    
    def compute_loss(self, batch_data, criterion):
        ipt, tgt, y = batch_data
        hidden, cell = self.init_hidden(ipt.size(0), y)
        output, hidden, cell = self.forward(ipt, hidden, cell)
        output = output.view(output.size(0) * output.size(1), -1)
        loss = criterion(output, tgt.view(-1))
        return loss