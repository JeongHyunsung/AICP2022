import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.Layer.encoderlayer_text import Text_Encoder
from Model.Layer.encoderlayer_face import Face_Encoder
from Model.Layer.encoderlayer_history import History_Encoder


class encoder(nn.Module):
    def __init__(self, max_words, input_size=512, hidden_size=1024, num_layers=1):
        super(encoder, self).__init__()
        self.max_words = max_words
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoderlayer_text = Text_Encoder(self.max_words,
                                              self.input_size,
                                              self.hidden_size,
                                              self.num_layers)

        self.encoderlayer_face = Face_Encoder(self.input_size,
                                              self.hidden_size,
                                              self.num_layers)

        self.encoderlayer_history = History_Encoder(self.hidden_size * 2,
                                                    self.hidden_size * 2,
                                                    self.hidden_size,
                                                    self.num_layers)

    def forward(self, x):
        # x : text (batch size, seq_len, history_len), y : face (batch size, seq_len, history_len, feature_dim)
        text_out_list = []  # list len : history_len-1, (batch_size, 1, 2*hidden_dim)
        text_out_query = self.encoderlayer_text(x[:, x.shape[1] - 1, :])  # (batch_size, 1, 2*hidden_dim)

        for hst_idx in range(x.shape[1]):
            text_out_list.append(self.encoderlayer_text(x[:, hst_idx, :]))

        text_out = torch.cat(text_out_list[:x.shape[1] - 1], dim=1)  # (batch_size, history_len-1, 2*hidden_dim)

        out = self.encoderlayer_history(text_out, text_out)  # (batch_size, 1, 2*hidden_dim)
        out = torch.cat([out, text_out_query], dim=1)  # (batch_size, 2, 2*hidden_dim)
        return out

