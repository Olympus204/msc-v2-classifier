import gzip
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

GRAPH_DIR = Path("data/graphs")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

def build_cooccurrence(shard_paths, level="full"):
    """
    level ∈ {"root", "mid", "full"}
    """
    counter = Counter()

    key = {
        "root": "root_codes",
        "mid": "mid_codes",
        "full": "full_codes",
    }[level]

    for shard in shard_paths:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                codes = sorted(set(row[key]))
                for a, b in combinations(codes, 2):
                    counter[(a, b)] += 1
                    counter[(b, a)] += 1

    return counter


def save(counter, path: Path):
    serialisable = {
        f"{a}|{b}": c for (a, b), c in counter.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f)


if __name__ == "__main__":
    from training.config import TRAIN_SHARDS

    for level in ["root", "mid", "full"]:
        cooc = build_cooccurrence(TRAIN_SHARDS, level)
        out_path = GRAPH_DIR / f"cooc_{level}.json"
        save(cooc, out_path)
