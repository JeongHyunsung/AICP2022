import torch
import torch.nn as nn

from Model.Layer.encoderlayer_text import encoderlayer_text
from Model.Layer.encoderlayer_history import encoderlayer_history


class encoder(nn.Module):
    def __init__(self, max_words, batch_size, input_dim=512, hidden_dim=1024, num_layers=1):
        super(encoder, self).__init__()
        self.batch_size = batch_size
        self.max_words = max_words
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoderlayer_text = encoderlayer_text(self.max_words,
                                                   self.batch_size,
                                                   self.input_dim,
                                                   self.hidden_dim,
                                                   self.num_layers)
        self.encoderlayer_history = encoderlayer_history(self.batch_size,
                                                         self.hidden_dim*2,
                                                         self.hidden_dim*2,
                                                         self.hidden_dim)

    def forward(self, x):
        # x : text (batch size, seq_len, history_len), y : face (batch size, seq_len, history_len, feature_dim)
        text_out_list = []  # list len : history_len-1, (batch_size, 1, 2*hidden_dim)
        text_out_query = self.encoderlayer_text(x[:, x.shape[1]-1, :])  # (batch_size, 1, 2*hidden_dim)
        for hst_idx in range(x.shape[1]):
            text_out_list.append(self.encoderlayer_text(x[:, hst_idx, :]))

        text_out = torch.cat(text_out_list[:x.shape[1]-1], dim=1)  # (batch_size, history_len-1, 2*hidden_dim)
        out = self.encoderlayer_history(text_out, text_out)  # (batch_size, 1, 2*hidden_dim)

        out = torch.cat([out, text_out_query], dim=1)  # (batch_size, 2, 2*hidden_dim)
        return out
    # face encoder required
















