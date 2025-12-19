# training/train.py
from tqdm import tqdm
import torch
import json
import os
import tempfile
import time
import numpy as np
from torch.utils.data import DataLoader
from training.checkpoint import load_checkpoint, save_checkpoint
from training import config
from training.build import (
    build_model,
    build_loss,
    build_optimizer,
    build_tokenizer,
    build_mappers,
)
from training.validate import validate
from data.dataset import MSCDataset, collate_fn

def find_latest_run(runs_dir="runs"):
    if not os.path.exists(runs_dir):
        return None

    candidates = []
    for d in os.listdir(runs_dir):
        full = os.path.join(runs_dir, d)
        if (
            os.path.isdir(full)
            and d.startswith("20")
            and os.path.isdir(os.path.join(full, "checkpoints"))
        ):
            candidates.append(d)

    if not candidates:
        return None

    candidates.sort()
    return candidates[-1]


def find_latest_checkpoint(run_dir):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None

    def key(name):
        if name.startswith("epoch_"):
            return (1, int(name.split("_")[1].split(".")[0]))
        if name.startswith("step_"):
            return (0, int(name.split("_")[1].split(".")[0]))
        return (-1, -1)

    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not ckpts:
        return None

    ckpts.sort(key=key)
    return ckpts[-1]


def serialisable_config(cfg):
    return {k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}

def save_batch_logs(batch_logs, log_dir, suffix="latest"):
    filename = f"batch_metrics_{suffix}.json"
    path = os.path.join(log_dir, filename)

    def dump_phase(f, phase_name, phase_data, indent="  "):
        f.write(f'{indent}"{phase_name}": {{\n')
        keys = list(phase_data.keys())
        for i, k in enumerate(keys):
            arr = phase_data[k]
            line = json.dumps(arr, separators=(", ", ": "))
            comma = "," if i < len(keys) - 1 else ""
            f.write(f'{indent*2}"{k}": {line}{comma}\n')
        f.write(f"{indent}}}")

    fd, tmp = tempfile.mkstemp(dir=log_dir, prefix=".tmp_batch_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("{\n")
            phases = list(batch_logs.keys())
            for i, p in enumerate(phases):
                dump_phase(f, p, batch_logs[p])
                if i < len(phases) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("}\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def hit_at_k(logits, targets, k):
    """
    logits: [B, C]
    targets: [B, C] multi-hot
    """
    topk = torch.topk(logits, k=k, dim=1).indices
    hits = torch.gather(targets, 1, topk).sum(dim=1)
    return (hits > 0).float().mean().item()

def load_batch_logs(log_dir, phases):
    merged = {}
    for p in phases:
        name = p["name"]
        path = os.path.join(log_dir, f"batch_metrics_{name}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                merged[name] = data[name]
    return merged if merged else None

def save_phase_results(phase_results, log_dir):
    path = os.path.join(log_dir, "phase_results.json")
    with open(path, "w") as f:
        json.dump(phase_results, f, indent=2)

def load_phase_results(log_dir):
    path = os.path.join(log_dir, "phase_results.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def main():
    phase_results = []
    global_step = 0

    device = config.DEVICE

    #Build components
    tokenizer = build_tokenizer()
    mappers = build_mappers()
    model = build_model(mappers)

    assert hasattr(model, "heads"), "Model has no .heads attribute"

    for param in model.encoder.parameters():
        param.requires_grad = False

    criterion = build_loss(mappers)
    optimizer = build_optimizer(model)

    soft_targets = None
    if config.USE_SOFT_TARGETS:
        print(f"[config] Using soft targets: alpha={config.SOFT_TARGET_ALPHA}")
        soft_targets = np.load(config.SOFT_TARGETS_PATH)
        soft_targets = torch.tensor(soft_targets, dtype=torch.float32, device=device)
        soft_targets = torch.clamp(soft_targets, min=1e-9)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        assert soft_targets.shape[0] == mappers.n_full, (
            f"Soft targets size {soft_targets.shape[0]} "
            f"!= n_full {mappers.n_full}"
        )

    start_phase = None
    start_epoch = 0

    if config.RESUME:
        if config.RESUME_RUN_ID is not None:
            RUN_ID = config.RESUME_RUN_ID
        else:
            RUN_ID = find_latest_run()
            assert RUN_ID is not None, "No runs found to resume from"
        
        RUN_DIR = os.path.join("runs", RUN_ID)
        CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
        LOG_DIR = os.path.join(RUN_DIR, "logs")
        BEST_DIR = os.path.join(RUN_DIR, "best")
        FINAL_DIR = os.path.join(RUN_DIR, "final")

        if config.RESUME_CHECKPOINT is not None:
            ckpt_name = config.RESUME_CHECKPOINT
        else:
            ckpt_name = find_latest_checkpoint(RUN_DIR)
            assert ckpt_name is not None, f"No checkpoints found in {RUN_DIR}"
        
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
        print(f"[RESUME] run={RUN_ID} checkpoint={ckpt_name}")
        state = load_checkpoint(
            ckpt_path,
            model,
            optimizer)
        start_phase = state["phase"]
        start_epoch = state["epoch"]
        global_step = state["global_step"]

        
        print(
            f"[RESUME] phase={start_phase} "
            f"epoch={start_epoch} "
            f"global_step={global_step} "
        )

        for d in [CKPT_DIR, LOG_DIR, BEST_DIR, FINAL_DIR]:
            os.makedirs(d, exist_ok=True)

    else:
        RUN_ID = time.strftime("%Y%m%d_%H%M%S")
        if config.DEBUG:
            RUN_ID +="_debug"
    RUN_DIR = os.path.join("runs", RUN_ID)
    CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
    LOG_DIR = os.path.join(RUN_DIR, "logs")
    BEST_DIR = os.path.join(RUN_DIR, "best")
    FINAL_DIR = os.path.join(RUN_DIR, "final")

    if config.RESUME:
        phase_results = load_phase_results(LOG_DIR)
        print(f"[RESUME] Loaded {len(phase_results)} completed phase results")
    else:
        phase = []

    batch_logs = None
    if config.RESUME:
        loaded = load_batch_logs(LOG_DIR, config.PHASES)
        if loaded is not None:
            batch_logs = loaded
            print("[RESUME] Loaded existing batch logs")
    
    if batch_logs is None:
        batch_logs = {
            phase["name"]: {
                "epoch": [],
                "step_in_phase": [],
                "global_step": [],
                "root_acc": [],
                "root_acc_ema": [],
                "mid_hit3": [],
                "full_hit5": [],
                "debug": [],
                "alpha": [],
            }
            for phase in config.PHASES
        }
    if config.RESUME:
        for phase in config.PHASES:
            name = phase["name"]
            if name not in batch_logs:
                batch_logs[name] = {
                    "epoch": [],
                    "step_in_phase": [],
                    "global_step": [],
                    "root_acc": [],
                    "root_acc_ema": [],
                    "mid_hit3": [],
                    "full_hit5": [],
                    "debug": [],
                    "alpha": [],
                }
                print(f"[RESUME] Initialised missing log phase: {name}")


    if not config.RESUME:
        for d in [CKPT_DIR, LOG_DIR, BEST_DIR, FINAL_DIR]:
            os.makedirs(d, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    print(f"[RUN] Using run directory: {RUN_DIR}")

    #Build datasets

    train_shards = config.TRAIN_SHARDS
    val_shards = config.VAL_SHARDS

    if config.DEBUG:
        train_shards = train_shards[:config.DEBUG_SHARDS_PER_SPLIT]
        val_shards = val_shards[:config.DEBUG_SHARDS_PER_SPLIT]

    train_dataset = MSCDataset(
        shard_paths=train_shards,
        tokenizer=tokenizer,
        mappers=mappers,
        max_len=config.MAX_LEN,
    )

    val_dataset = MSCDataset(
        shard_paths=val_shards,
        tokenizer=tokenizer,
        mappers=mappers,
        max_len=config.MAX_LEN,
    )

    #DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    best_full_f1 = -1.0

    resume_phase = start_phase
    resume_epoch = start_epoch

    if config.RESUME and resume_phase is not None:
        phase_names = [p["name"] for p in config.PHASES]
        resume_phase_index = phase_names.index(resume_phase)
    else:
        resume_phase_index = 0

    #Training loop
    for i, phase in enumerate(config.PHASES):
        phase_name = phase["name"]

        if config.RESUME and i < resume_phase_index:
            print(f"[RESUME] Skipping completed phase: {phase_name}")
            continue

        if phase["name"] == start_phase and start_epoch > phase["epochs"]:
            print(f"[RESUME] Phase {phase['name']} already complete, skipping")
            start_phase = None
            continue

        print(f"\n=== Phase: {phase['name']} ===")

        for p in model.encoder.parameters():
            p.requires_grad = not phase["freeze_encoder"]

        optimizer = build_optimizer(model)

        if config.RESUME and (i == resume_phase_index) and (not config.RESET_OPTIMIZER):
            try:
                optimizer.load_state_dict(state["optimizer"])
                print("[RESUME] Loaded optimizer state")
            except ValueError as e:
                print(f"[RESUME] Optimizer mismatch, resetting optimizer ({e})")


        lam = phase.get("lambda_graph", 0.5)
        model.heads.root.lambda_graph = 0.0
        model.heads.mid.lambda_graph = 0.0
        model.heads.full.lambda_graph = lam

        if config.RESUME and phase_name == resume_phase:
            epoch_start = resume_epoch + 1
        else:
            epoch_start = 1

        for e in range(epoch_start, phase["epochs"] + 1):
            start_epoch = 0

            ema_root = None

            print(f"\n[{phase['name']}] Epoch {e}/{phase['epochs']}")
            model.train()
            running_loss = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"{phase['name']} E{e}",
                leave=True,
                ncols=100,
            )

            for step, batch in enumerate(pbar, 1):
                if config.DEBUG and step >= config.DEBUG_MAX_BATCHES:
                    pbar.set_postfix_str("DEBUG STOP")
                    break

                if global_step > 0 and global_step % config.CHECKPOINT_EVERY_STEPS == 0:
                    save_checkpoint(
                        f"{CKPT_DIR}/step_{global_step}.pt",
                        model,
                        optimizer,
                        {
                            "run_id": RUN_ID,
                            "phase": phase["name"],
                            "epoch": e,
                            "global_step": global_step,
                            "config": serialisable_config(config),
                        }
                    )

                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = {k: v.to(device) for k, v in batch["labels"].items()}

                if config.DEBUG:
                    for k, v in labels.items():
                        if not isinstance(v,torch.Tensor):
                            raise TypeError(f"Label {k} is not tensor: {type(v)}")

                outputs = model(input_ids, attention_mask)

                loss = criterion(
                    outputs=outputs,
                    labels=labels,
                    train_heads=phase["train_heads"],
                    soft_targets=soft_targets,
                    alpha=config.SOFT_TARGET_ALPHA,
                )


                loss.backward()
                optimizer.step()

                #metrics
                running_loss += loss.item()

                with torch.no_grad():
                    root_acc = (
                        torch.argmax(outputs["root"], dim=1)
                        == labels["root"].argmax(dim=1)
                    ).float().mean().item()

                    mid_hit3 = (
                        hit_at_k(outputs["mid"], labels["mid"], k=3)
                        if "mid" in phase["train_heads"] else None
                    )

                    full_hit5 = (
                        hit_at_k(outputs["full"], labels["full"], k=5)
                        if "full" in phase["train_heads"] else None
                    )

                    alpha = 0.9
                    if ema_root is None:
                        ema_root = root_acc
                    else:
                        ema_root = alpha * ema_root + (1 - alpha) * root_acc


                phase_log = batch_logs[phase["name"]]

                phase_log["epoch"].append(e)
                phase_log["step_in_phase"].append(step)
                phase_log["global_step"].append(global_step)
                phase_log["root_acc"].append(root_acc)
                phase_log["root_acc_ema"].append(ema_root)
                phase_log["mid_hit3"].append(mid_hit3)
                phase_log["full_hit5"].append(full_hit5)
                phase_log["debug"].append(config.DEBUG)
                phase_log["alpha"].append(config.SOFT_TARGET_ALPHA)

                global_step += 1

                if step % 1000 == 0:
                    save_batch_logs(batch_logs, LOG_DIR, suffix="latest")

                avg_loss = running_loss / step

                m_str = f"{mid_hit3:.2f}" if mid_hit3 is not None else "--"

                f_str = f"{full_hit5:.2f}" if full_hit5 is not None else "--"

                pbar.set_postfix_str(
                    f"L={avg_loss:.4f} "
                    f"R={root_acc:.2f} "
                    f"EMA={ema_root:.2f} "
                    f"M@3={m_str} "
                    f"F@5={f_str} "
                )

            save_checkpoint(
                f"{CKPT_DIR}/epoch_{e}.pt",
                model,
                optimizer,
                {
                    "run_id": RUN_ID,
                    "phase": phase["name"],
                    "epoch": e,
                    "global_step": global_step,
                    "config": serialisable_config(config),
                }
            )

            if phase["name"] == "full":
                metrics = validate(model, val_loader, device)

                print(
                    f"[VAL] "
                    f"epoch={e} | "
                    f"root_acc={metrics['root_acc']:.3f} | "
                    f"mid_f1={metrics['mid_f1']:.3f} | "
                    f"full_f1={metrics['full_f1']:.3f}"
                )

                if metrics["full_f1"] > best_full_f1:
                    best_full_f1 = metrics["full_f1"]
                    best_epoch = e
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "metrics": metrics,
                            "phase": "full",
                            "epoch": best_epoch,
                            "run_id": RUN_ID,
                        },
                        os.path.join(BEST_DIR, "best_full.pt")
                    )
                    print(f"[SAVE] New best full model: F1={best_full_f1:.3f}")


        #validation (per phase)
        metrics = validate(model, val_loader, device)

        print(
            f"[VAL] "
            f"root_acc={metrics['root_acc']:.3f} | "
            f"mid_f1={metrics['mid_f1']:.3f} | "
            f"full_f1={metrics['full_f1']:.3f}"
        )

        phase_results.append({
            "phase": phase["name"],
            "root_acc": metrics["root_acc"],
            "mid_f1": metrics["mid_f1"],
            "full_f1": metrics["full_f1"],
        })

        save_batch_logs(batch_logs, LOG_DIR, suffix=phase["name"])
        save_phase_results(phase_results, LOG_DIR)

        if start_phase == phase["name"]:
            start_phase = None


    print("\n=== Phase summary ===")
    print(f"{'Phase':<10} {'Root Acc':>10} {'Mid F1':>10} {'Full F1':>10}")
    print("-" * 44)

    for r in phase_results:
        print(
            f"{r['phase']:<10} "
            f"{r['root_acc']:>10.3f} "
            f"{r['mid_f1']:>10.3f} "
            f"{r['full_f1']:>10.3f}"
        )
    final_path = os.path.join(FINAL_DIR, "model.pt")
    best_path = os.path.join(BEST_DIR, "best_full.pt")

    if os.path.exists(best_path):
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model_state = best["model_state"]
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "model_state": model_state,
            "config": serialisable_config(config),
            "run_id": RUN_ID,
            "final_metrics": phase_results,
            "note": "Best full model if available, else last epoch",
        },
        final_path
    )

    with open(os.path.join(FINAL_DIR, "mappers.json"), "w") as f:
        json.dump(
            {
                "root_to_idx": mappers.root_to_idx,
                "mid_to_idx": mappers.mid_to_idx,
                "full_to_idx": mappers.full_to_idx,
                "n_root": mappers.n_root,
                "n_mid": mappers.n_mid,
                "n_full": mappers.n_full,
            },
            f,
            indent=2
        )

    with open(os.path.join(FINAL_DIR, "schema.json"), "w") as f:
        json.dump(
            {
                "model": config.MODEL_NAME,
                "proj_dim": config.PROJ_DIM,
                "n_root": mappers.n_root,
                "n_mid": mappers.n_mid,
                "n_full": mappers.n_full,
            },
            f,
            indent=2
        )

    with open(os.path.join(FINAL_DIR, "README.txt"), "w") as f:
        f.write(
            f"MSC v2 trained model \n"
            f"Run ID: {RUN_ID}\n"
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"This folder contains everything needed to load and use the model"
        )


if __name__ == "__main__":
    main()
