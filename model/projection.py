# model/projection.py

import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
