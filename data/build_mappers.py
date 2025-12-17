# data/build_mappers.py

import gzip
import json
from pathlib import Path
from data.mappers import MSCMappers

def build_mappers_from_shards(shard_paths):
    root_codes = set()
    mid_codes = set()
    full_codes = set()

    for shard in shard_paths:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                root_codes.update(row["root_codes"])
                mid_codes.update(row["mid_codes"])
                full_codes.update(row["full_codes"])

    return MSCMappers(root_codes, mid_codes, full_codes)
