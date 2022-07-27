import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Text_Encoder(nn.Module):
    def __init__(self, max_words, input_dim=512, hidden_dim=1024, lstm_layers=1, bidirectional=True):

        self.max_words = max_words
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.D = 1
        if self.bidirectional:
            self.D = 2

        self.embedding = nn.Embedding(self.max_words, self.input_dim, padding_idx=0)
        self.BiLSTM = nn.LSTM(input_size=self.input_dim,
                              hidden_Size=self.hiddn_dim,
                              num_layers=self.lstm_layers,
                              bidirectional=self.bidirectional,
                              batch_first=True)

    def forward(self, x):
        h = torch.zeros((self.lstm_layers, x.size(0), self.hidden_dim * self.D))
        c = torch.zeros((self.lstm_layers, x.size(0), self.hidden_dim * self.D))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        x = self.embedding(x)
        x, (h, c) = self.BILSTM(x, (h, c))

        return x

