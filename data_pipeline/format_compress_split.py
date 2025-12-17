import csv
import sys
import gzip
import json
import hashlib
from pathlib import Path
from typing import Dict
csv.field_size_limit(sys.maxsize)

from data_pipeline.clean_raw import clean_row

# ---------------- CONFIG ----------------

RAW_CSV = "data/raw_harvest_rewritten.csv"
OUTPUT_DIR = "data/shards"

TRAIN_FRACTION = 0.9
VAL_FRACTION   = 0.05
TEST_FRACTION  = 0.05

ROWS_PER_SHARD = 25_000
ENCODING = "utf-8"

# --------------------------------------


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def open_shard(split: str, shard_idx: int):
    fname = f"{split}_{shard_idx:03d}.jsonl.gz"
    path = Path(OUTPUT_DIR) / fname
    return gzip.open(path, "wt", encoding="utf-8")

    
def choose_split_from_id(paper_id: str):
    h = hashlib.md5(paper_id.encode()).hexdigest()
    r = int(h[:8], 16) / 2**32

    if r < TRAIN_FRACTION:
        return "train"
    elif r < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    else:
        return "test"


def count_rows(csv_path: str) -> int:
    print("Counting rows...")
    with open(csv_path, "r", encoding=ENCODING, errors="ignore", newline="") as f:
        return sum(1 for _ in f) - 1  # subtract header


def main():
    ensure_dir(OUTPUT_DIR)

    total_rows = count_rows(RAW_CSV)
    print(f"Total raw rows: {total_rows}")

    shard_idx = {"train": 0, "val": 0, "test": 0}
    shard_rows = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    cleaned = 0
    skipped = 0

    shard_files = {
        split: open_shard(split, 0)
        for split in ("train", "val", "test")
    }

    with open(RAW_CSV, "r", encoding=ENCODING, errors="ignore", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            try:
                clean = clean_row(row)
            except Exception as e:
                skipped += 1
                continue

            if clean is None:
                skipped += 1
                continue

            split = choose_split_from_id(clean["id"])

            shard_files[split].write(
                json.dumps(clean, ensure_ascii=False) + "\n"
            )

            shard_rows[split] += 1
            counts[split] += 1
            cleaned += 1

            if shard_rows[split] >= ROWS_PER_SHARD:
                shard_files[split].close()
                shard_idx[split] += 1
                shard_files[split] = open_shard(split, shard_idx[split])
                shard_rows[split] = 0

            if (i + 1) % 50_000 == 0:
                print(
                    f"[PROGRESS] {i+1:,}/{total_rows:,} "
                    f"cleaned={cleaned:,} skipped={skipped:,}"
                )

    for f in shard_files.values():
        f.close()

    print("\n=== DONE ===")
    print(f"Cleaned rows: {cleaned}")
    print(f"Skipped rows: {skipped}")
    for split in ("train", "val", "test"):
        print(f"{split.upper():5s}: {counts[split]:,}")


if __name__ == "__main__":
    main()
