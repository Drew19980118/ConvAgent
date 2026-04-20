#!/usr/bin/env python3
import os
from datasets import load_dataset

SAVE_ROOT = "data_conv"

targets = [
    ("topiocqa", "train", os.path.join(SAVE_ROOT, "topiocqa_train", "train.parquet")),
    ("topiocqa", "test",  os.path.join(SAVE_ROOT, "topiocqa_test",  "test.parquet")),
]

for subset, split, out_path in targets:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds = load_dataset(
        "slupart/chatr1-convqa-all",
        subset,
        split=split,
    )

    ds.to_parquet(out_path)
    print(f"Saved {subset}/{split} -> {out_path} ({len(ds)} rows)")
