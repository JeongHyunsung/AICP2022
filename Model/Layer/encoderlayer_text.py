import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class index2input(nn.Module):
    def __init__(self, max_words, input_size):
        super(index2input, self).__init__()
        self.max_words = max_words
        self.linear = nn.Linear(max_words, input_size)

    def forward(self, x):  # (batch_size, seq_len)
        x = F.one_hot(x, self.max_words)  # (batch_size, seq_len, vocab_size)
        x = x.type(torch.float)
        x = self.linear(x)  # (batch_size, seq_len, input_dim)
        return x


class Text_Encoder(nn.Module):
    def __init__(self, max_words, input_size, hidden_size, num_layers):
        super(Text_Encoder, self).__init__()
        self.max_words = max_words
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.index2input = index2input(self.max_words, self.input_size)
        self.LSTM = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x):  # input : (batch_size, seq_length), number range : 0~(max_words-1)
        batch_size = x.shape[0]
        h = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))
        c = torch.zeros((self.num_layers * 2, batch_size, self.hidden_size))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.index2input(x)  # (batch_size, seq_length, input_dim)

        out, (h, c) = self.LSTM(out, (h, c))  # (batch_size, seq_length, D*hidden_dim), ((D*num_layers, batch, hidden_dim), (D*num_layers, batch, hidden_dim))

        h = torch.transpose(h, 0, 1)  # (batch, 2, hidden)
        h = torch.reshape(h, (batch_size, 1, -1))  # (batch_size, 1, 2*hidden_dim)
        return h

