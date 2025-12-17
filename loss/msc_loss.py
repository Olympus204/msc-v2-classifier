import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCLoss(nn.Module):
    def __init__(self, loss_weights, mid_full_mask=None):
        super().__init__()

        self.w = loss_weights
        self.mid_full_mask = mid_full_mask

        # Loss primitives
        self.ce_root = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels, train_heads, soft_targets=None, alpha=0.0):
        loss = 0.0

        # --- ROOT ---
        if "root" in train_heads:
            loss += self.w["root"] * self.ce_root(
                outputs["root"],
                labels["root"].argmax(dim=1)
            )

        # --- MID ---
        if "mid" in train_heads:
            loss += self.w["mid"] * self.bce(
                outputs["mid"],
                labels["mid"]
            )

        # --- FULL ---
        if "full" in train_heads:
            logits = outputs["full"]

            # Apply mid→full constraint if present
            if self.mid_full_mask is not None:
                allowed = labels["mid"] @ self.mid_full_mask   # [B, n_full]
                logits = logits.masked_fill(~allowed.bool(), float("-inf"))

            hard = self.bce(logits, labels["full"])

            if soft_targets is not None and alpha > 0:
                with torch.no_grad():
                    q = soft_targets[labels["full"].argmax(dim=1)]
                logp = F.log_softmax(logits, dim=1)
                soft = -(q * logp).sum(dim=1).mean()
                loss += self.w["full"] * ((1 - alpha) * hard + alpha * soft)
            else:
                loss += self.w["full"] * hard

        return loss
