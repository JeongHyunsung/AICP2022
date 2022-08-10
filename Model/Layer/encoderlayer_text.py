import torch
import torch.nn as nn

from Model.Cell.BiLSTM import BiLSTM
from Model.Embedding.onehot import index2input


class encoderlayer_text(nn.Module):
    def __init__(self, max_words, batch_size, input_dim=512, hidden_dim=1024, num_layers=1):
        super(encoderlayer_text, self).__init__()
        self.batch_size = batch_size
        self.max_words = max_words
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.index2input = index2input(self.max_words, self.input_dim)
        self.BiLSTM = BiLSTM(self.batch_size, self.input_dim, self.hidden_dim, self.num_layers)

    def forward(self, x):
        x = self.index2input(x)
        x = self.BiLSTM(x)
        return x



