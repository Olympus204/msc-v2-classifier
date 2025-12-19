import gzip
import json
import torch
from torch.utils.data import Dataset


class MSCDataset(Dataset):
    def __init__(self, shard_paths, tokenizer, mappers, max_len):
        self.shard_paths = shard_paths
        self.tokenizer = tokenizer
        self.mappers = mappers
        self.max_len = max_len

        self.index = self._build_index()
        self._shard_cache = {}

    def _build_index(self):
        index = []
        for shard_path in self.shard_paths:
            with gzip.open(shard_path, "rt", encoding="utf-8") as f:
                for offset, _ in enumerate(f):
                    index.append((shard_path, offset))
        return index

    def _safe_codes(self, x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        shard_path, line_no = self.index[idx]

        #Load + cache shard
        if shard_path not in self._shard_cache:
            with gzip.open(shard_path, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]

            texts = [r["text"] for r in rows]
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )

            self._shard_cache[shard_path] = {
                "rows": rows,
                "encodings": encodings,
            }

        cache = self._shard_cache[shard_path]
        row = cache["rows"][line_no]

        return {
            "input_ids": cache["encodings"]["input_ids"][line_no],
            "attention_mask": cache["encodings"]["attention_mask"][line_no],
            "labels": {
                "root": self.mappers.root(self._safe_codes(row.get("root_codes"))),
                "mid":  self.mappers.mid(self._safe_codes(row.get("mid_codes"))),
                "full": self.mappers.full(self._safe_codes(row.get("full_codes"))),
            },
        }


def collate_fn(batch):
    #Hard sanity check so this never silently breaks again
    for x in batch:
        for k in ("root", "mid", "full"):
            assert isinstance(x["labels"][k], torch.Tensor), (
                k, type(x["labels"][k])
            )

    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": {
            "root": torch.stack([x["labels"]["root"] for x in batch]),
            "mid":  torch.stack([x["labels"]["mid"]  for x in batch]),
            "full": torch.stack([x["labels"]["full"] for x in batch]),
        },
    }
