# model/graph_label_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLabelHead(nn.Module):
    """
    Produces:
      - text-only logits
      - graph-similarity logits
      - combined logits = text + λ · graph

    The separation exists so the training loss can decide
    how to use each signal (e.g. soft targets on full only).
    """

    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        label_emb: torch.Tensor | None,
        graph_dim: int = 64,
        lambda_graph: float = 0.5,
    ):
        super().__init__()

        self.text_head = nn.Linear(in_dim, num_labels)

        self.lambda_graph = lambda_graph
        self.has_graph = label_emb is not None

        if self.has_graph:
            # trainable label embeddings (fine for now)
            self.label_emb = nn.Parameter(label_emb)   # [C, graph_dim]
            self.proj = nn.Linear(in_dim, graph_dim)

    def forward(self, z, return_parts: bool = False):
        """
        Args:
            z: [B, D] encoder features
            return_parts: if True, return (combined, text, graph)

        Returns:
            logits or tuple of logits
        """
        text_logits = self.text_head(z)

        if not self.has_graph or self.lambda_graph == 0.0:
            if return_parts:
                return text_logits, text_logits, None
            return text_logits

        z_g = F.normalize(self.proj(z), dim=-1)      # [B, graph_dim]
        E   = F.normalize(self.label_emb, dim=-1)   # [C, graph_dim]

        graph_logits = z_g @ E.t()                   # [B, C]
        combined = text_logits + self.lambda_graph * graph_logits

        if return_parts:
            return combined, text_logits, graph_logits

        return combined
