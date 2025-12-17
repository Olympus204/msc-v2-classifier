import json
import torch
import torch.nn.functional as F

def load_label_embeddings(path, mappers, dim):
    with open(path) as f:
        raw = json.load(f)

    out = {}

    for level in ["root", "mid", "full"]:
        mapping = getattr(mappers, f"{level}_to_idx")
        n = len(mapping)

        emb = torch.zeros(n, dim)

        level_embs = raw.get(level, {})

        for code, idx in mapping.items():
            if code in level_embs:
                emb[idx] = torch.tensor(level_embs[code], dtype=torch.float32)

        # normalize AFTER emb is definitely defined
        emb = F.normalize(emb, dim=1)

        out[level] = emb

    return out
