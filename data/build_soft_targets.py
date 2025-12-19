# data/build_soft_targets.py

import numpy as np
import torch
from utils.graph_embeddings import load_label_embeddings
from data.build_mappers import build_mappers_from_shards
from training import config

def main():
    mappers = build_mappers_from_shards(config.TRAIN_SHARDS)
    n_full = mappers.n_full
    print(f"[soft-targets] n_full = {n_full}")
    label_embs = load_label_embeddings(
        config.GRAPH_EMB_PATH,
        mappers,
        dim=config.GRAPH_DIM
    )["full"]

    assert label_embs.shape[0] == n_full, (
        f"Embeddings {label_embs.shape[0]} != mapper {n_full}"
    )
    norms = np.linalg.norm(label_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = label_embs / norms
    S = E @ E.T
    T = config.SOFT_TARGET_TEMP
    S = np.exp(S / T)
    S = S / S.sum(axis=1, keepdims=True)
    out_path = "data/graphs/soft_targets_full.npy"
    np.save(out_path, S)

    print(f"[soft-targets] saved {S.shape} -> {out_path}")

if __name__ == "__main__":
    main()
