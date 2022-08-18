import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Face_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Face_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=True)

    def forward(self, x):  # input : (batch_size, seq_length), number range : 0~(max_words-1)
        batch_size = input.shape[0]
        h = torch.zeros((self.num_layers * self.D, batch_size, self.hidden_size))
        c = torch.zeros((self.num_layers * self.D, batch_size, self.hidden_size))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out, (h, c) = self.rnn(x, (h, c))  # (batch_size, seq_length, D*hidden_dim), ((D*num_layers, batch, hidden_dim), (D*num_layers, batch, hidden_dim))

        h = torch.transpose(h, 0, 1)  # (batch_size, 2, hidden_dim)
        h = torch.reshape(h, (self.batch_size, 1, -1))  # (batch_size, 1, 2*hidden_dim)
        return h
