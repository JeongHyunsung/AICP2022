import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class History_Encoder(nn.Module):
    def __init__(self, text_size, face_size, hidden_size, num_layers):
        super(History_Encoder, self).__init__()
        self.text_size = text_size
        self.face_size = face_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(self.text_size + self.face_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, x, y):  # (batch, history_length, text_size), (batch, history_length, face_size)
        batch_size = x.shape[0]
        assert x.shape[0] == y.shape[0]

        h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)
        c = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size)

        out = torch.cat([x, y], dim=2)  # (batch, history_length, text_size+face_size)
        out, (h, c) = self.rnn(out, (h, c))  # (batch, history_length, 2*hidden), ((2, batch_size, hidden), (2, batch_size, hidden)

        h = torch.transpose(h, 0, 1)  # (batch, 2, hidden)
        h = torch.reshape(h, (batch_size, 1, -1))  # (batch_size, 1, 2*hidden_dim)
        return h

