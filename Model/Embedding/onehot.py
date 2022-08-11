import torch
import torch.nn as nn
import torch.nn.functional as F


class index2input(nn.Module):
    def __init__(self, max_words, input_dim):
        super(index2input, self).__init__()
        self.max_words = max_words
        self.linear = nn.Linear(max_words, input_dim)

    def forward(self, x):  # (batch_size, seq_len)
        x = F.one_hot(x, self.max_words)  # (batch_size, seq_len, vocab_size)
        x = x.type(torch.float)
        x = self.linear(x)  # (batch_size, seq_len, input_dim)
        return x
