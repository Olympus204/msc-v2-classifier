# data/build_label_embeddings.py
import json
import torch

def _build_matrix(code_to_idx: dict, emb_dict: dict, dim: int):
    C = len(code_to_idx)
    M = torch.zeros(C, dim, dtype=torch.float32)

    missing = 0
    for code, idx in code_to_idx.items():
        vec = emb_dict.get(code)
        if vec is None:
            missing += 1
            continue
        M[idx] = torch.tensor(vec, dtype=torch.float32)

    return M, missing

def load_label_embeddings(path, mappers, dim=64):
    """
    Returns dict of tensors:
      {"root": [n_root,dim], "mid": [n_mid,dim], "full": [n_full,dim]}
    """
    with open(path, "r", encoding="utf-8") as f:
        all_emb = json.load(f)

    mats = {}

    root_mat, root_missing = _build_matrix(mappers.root_code_to_idx, all_emb["root"], dim)
    mid_mat,  mid_missing  = _build_matrix(mappers.mid_code_to_idx,  all_emb["mid"],  dim)
    full_mat, full_missing = _build_matrix(mappers.full_code_to_idx, all_emb["full"], dim)

    mats["root"] = root_mat
    mats["mid"] = mid_mat
    mats["full"] = full_mat

    print(f"[graph] missing embeddings: root={root_missing} mid={mid_missing} full={full_missing}")
    return mats
