# training/build.py

import torch
from transformers import AutoTokenizer

from model.encoder import TextEncoder
from model.projection import Projection
from model.heads import MSCHeads
from model.msc_model import MSCModel
from utils.graph_embeddings import load_label_embeddings
from loss.msc_loss import MSCLoss
from data.build_mappers import build_mappers_from_shards
from training import config


def build_tokenizer():
    return AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        use_fast=False
    )

def build_mid_full_mask(mappers):
    mid_to_full = mappers.build_mid_to_full()

    mask = torch.zeros(
        (mappers.n_mid, mappers.n_full),
        dtype=torch.float32
    )

    for mid_idx, full_list in mid_to_full.items():
        mask[mid_idx, full_list] = 1.0
    return mask



def build_mappers():
    return build_mappers_from_shards(config.TRAIN_SHARDS)

def build_model(mappers):
    encoder = TextEncoder(config.MODEL_NAME)
    projector = Projection(encoder.hidden_size, config.PROJ_DIM)

    label_embs = None
    if getattr(config, "GRAPH_EMB_PATH", None):
        label_embs = load_label_embeddings(
            config.GRAPH_EMB_PATH,
            mappers,
            dim=config.GRAPH_DIM
        )

    heads = MSCHeads(
        config.PROJ_DIM,
        mappers.n_root,
        mappers.n_mid,
        mappers.n_full,
        label_embs=label_embs,
        lambda_graph=getattr(config, "LAMBDA_GRAPH", 0.5),
    )

    if label_embs is not None:
        print(
            heads.root.label_emb.shape,
            heads.mid.label_emb.shape,
            heads.full.label_emb.shape,
        )

    model = MSCModel(encoder, projector, heads)
    return model.to(config.DEVICE)



def build_loss(mappers):
    mid_full_mask = build_mid_full_mask(mappers)

    return MSCLoss(
        config.LOSS_WEIGHTS,
        mid_full_mask=mid_full_mask.to(config.DEVICE)
    )




def build_optimizer(model):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR
    )
