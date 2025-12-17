# model/msc_model.py

import torch.nn as nn

class MSCModel(nn.Module):
    def __init__(self, encoder, projection, heads):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.heads = heads

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)
        x = self.projection(x)
        return self.heads(x)
