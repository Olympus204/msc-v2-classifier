# training/validate.py
import torch
from utils.metrics import multilabel_micro_f1, multilabel_topk_accuracy


@torch.no_grad()
def validate(model, dataloader, device: str, topk_root: int = 1):
    model.eval()

    total_root_topk = 0.0
    total_mid_f1 = 0.0
    total_full_f1 = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = {}
        for k,v in batch["labels"].items():
            assert isinstance(v, torch.Tensor), (k, type(v), getattr(v, "shape", None))
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        logits = model(input_ids, attention_mask)

        total_root_topk += multilabel_topk_accuracy(logits["root"], labels["root"], k=topk_root)

        total_mid_f1 += multilabel_micro_f1(logits["mid"], labels["mid"], threshold=0.5)
        total_full_f1 += multilabel_micro_f1(logits["full"], labels["full"], threshold=0.5)

        n_batches += 1

    model.train()

    if n_batches == 0:
        return {"root_topk_acc": 0.0, "mid_micro_f1": 0.0, "full_micro_f1": 0.0}

    return {
        "root_acc": total_root_topk / n_batches,
        "mid_f1": total_mid_f1 / n_batches,
        "full_f1": total_full_f1 / n_batches,
    }
