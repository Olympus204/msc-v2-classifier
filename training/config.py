# training/config.py

from pathlib import Path

# Paths
SHARD_DIR = Path("data/shards")

TRAIN_SHARDS = sorted((SHARD_DIR / "train").glob("*.gz"))
VAL_SHARDS   = sorted((SHARD_DIR / "val").glob("*.gz"))
TEST_SHARDS  = sorted((SHARD_DIR / "test").glob("*.gz"))
CHECKPOINT_DIR = Path("checkpoints")

# Model
MODEL_NAME = "microsoft/deberta-v3-base"
PROJ_DIM = 512

PHASES = [
    {
        "name": "root",
        "epochs": 1,
        "train_heads": ["root"],
        "freeze_encoder": True,
        "lambda_graph":0.0,
    },
    {
        "name": "mid",
        "epochs": 1,
        "train_heads": ["root", "mid"],
        "freeze_encoder": False,
        "lambda_graph":0.0,
    },
    {
        "name": "full",
        "epochs": 3,
        "train_heads": ["root", "mid", "full"],
        "freeze_encoder": False,
        "lambda_graph":0.5,
    },
]

DEBUG = False
DEBUG_MAX_BATCHES = 5
DEBUG_SHARDS_PER_SPLIT = 1

GRAPH_EMB_PATH = "data/graphs/graph_embeddings.json"
GRAPH_DIM = 64

SOFT_TARGET_TEMP = 0.07
USE_SOFT_TARGETS = True
SOFT_TARGETS_PATH = "data/graphs/soft_targets_full.npy"
SOFT_TARGET_ALPHA = 0.3

RESUME = False

RESUME_RUN_ID = None
RESUME_CHECKPOINT = None
CHECKPOINT_EVERY_STEPS = 2000
RESET_OPTIMIZER = True
# Training
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
MAX_LEN = 256
NUM_WORKERS = 4

# Loss weights
LOSS_WEIGHTS = {
    "root": 0.2,
    "mid": 0.3,
    "full": 0.5,
}

DEVICE = "cuda"
