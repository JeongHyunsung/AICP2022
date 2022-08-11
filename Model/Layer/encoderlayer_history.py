import torch
import torch.nn as nn

from Model.Cell.BiLSTM import BiLSTM


class encoderlayer_history(nn.Module):
    def __init__(self, batch_size, text_input_dim, face_input_dim, hidden_dim):
        super(encoderlayer_history, self).__init__()
        self.batch_size = batch_size
        self.text_input_dim = text_input_dim
        self.face_input_dim = face_input_dim
        self.hidden_dim = hidden_dim

        self.BiLSTM = BiLSTM(self.batch_size, self.text_input_dim+self.face_input_dim, self.hidden_dim, 1)

    def forward(self, x, y):  # (batch_size, history_length, 2*hidden_dim), (batch_size, history_length, 2*hidden_dim)
        out = torch.cat([x, y], dim=2)  # (batch_size, history_length, 4*hidden_dim)
        out = self.BiLSTM(out)  # (batch_size, 1, hidden_dim)
        return out
