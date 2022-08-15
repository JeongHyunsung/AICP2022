import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.D = 1

        self.LSTM = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, x, h, c):  # (batch_size, seq_length, input_dim)

        out, (h, c) = self.LSTM(x, (h, c))
        # (batch_size, seq_length, D*hidden_dim), ((D*num_layers, batch, hidden_dim), (D*num_layers, batch, hidden_dim))

        h = torch.transpose(h, 0, 1)
        h = torch.reshape(h, (self.batch_size, 1, -1))  # (batch_size, 1, D*num_layers*hidden_dim)
        return h