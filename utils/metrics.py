# utils/metrics.py
import torch


@torch.no_grad()
def multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    logits: [B, C]
    targets: [B, C] in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


@torch.no_grad()
def multilabel_topk_accuracy(logits, targets, k=5):
    #logits: [B, C]
    #targets: [B, C] (multi-hot, float/bool)

    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    C = logits.size(1)
    if targets.size(1) != C:
        raise ValueError(f"targets has C={targets.size(1)} but logits has C={C}")

    k = min(k, C)  #critical

    topk = torch.topk(logits, k=k, dim=1).indices

    #extra paranoia: assert indices valid
    if topk.max().item() >= C or topk.min().item() < 0:
        raise ValueError(f"topk indices out of range: min={topk.min().item()} max={topk.max().item()} C={C}")

    hits = torch.gather(targets, 1, topk).sum(dim=1)
    return (hits > 0).float().mean().item()

