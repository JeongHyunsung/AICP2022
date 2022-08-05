import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Text_Encoder(nn.Module):
    def __init__(self, batch_size, max_words, embedded_dim=512, hidden_dim=1024, output_dim=1024, bidirectional=True, num_layers=1):

        super(Text_Encoder, self).__init__()
        self.batch_size = batch_size
        self.max_words = max_words
        self.embedded_dim = embedded_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.D = 1
        if self.bidirectional:
            self.D = 2

        self.embedding = nn.Embedding(self.max_words, self.embedded_dim, padding_idx=0)
        self.LSTM = nn.LSTM(input_size=self.embedded_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(self.D*self.hidden_dim*self.num_layers, self.output_dim)

    def forward(self, x):  # input : (batch_size, seq_length), number range : 0~(max_words-1)

        h = torch.zeros((self.num_layers*self.D, self.batch_size, self.hidden_dim))
        c = torch.zeros((self.num_layers*self.D, self.batch_size, self.hidden_dim))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.embedding(x)  # (batch_size, seq_length, input_dim)

        out, (h, c) = self.LSTM(out, (h, c))  # (batch_size, seq_length, D*hidden_dim), ((D*num_layers, batch, hidden_dim), (D*num_layers, batch, hidden_dim))

        h = torch.transpose(h, 0, 1)  # (batch, D*num_layers, hidden_dim)
        h = torch.reshape(h, (self.batch_size, -1))  # (batch, D*num_layers*hidden_dim)
        h = self.linear(h)  # (batch, output_dim)
        h = self.tanh(h)
        return h

        """
        out = torch.transpose(out, 0, 1)
        out = torch.reshape(out, (self.batch_size, -1))
        out = self.linear(out)
        out = self.tanh(out)
        out = torch.transpose(out, 0, 1)
        return out
        """
