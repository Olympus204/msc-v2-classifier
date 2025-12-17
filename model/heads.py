# model/heads.py
import torch.nn as nn
from model.graph_label_head import GraphLabelHead

class MSCHeads(nn.Module):
    def __init__(self, in_dim, n_root, n_mid, n_full, label_embs=None, lambda_graph=0.5):
        super().__init__()
        label_embs = label_embs or {}

        self.root = GraphLabelHead(in_dim, n_root, label_embs.get("root"), lambda_graph=lambda_graph)
        self.mid  = GraphLabelHead(in_dim, n_mid,  label_embs.get("mid"),  lambda_graph=lambda_graph)
        self.full = GraphLabelHead(in_dim, n_full, label_embs.get("full"), lambda_graph=lambda_graph)

    def forward(self, z):
        return {
            "root": self.root(z),
            "mid":  self.mid(z),
            "full": self.full(z),
        }
