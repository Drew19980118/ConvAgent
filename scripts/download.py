import argparse
import os
import glob
from huggingface_hub import hf_hub_download

import pyarrow.parquet as pq
import orjson

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, required=True)
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)

# ---- Index ----
repo_id = "slupart/qrecc-e5-index"

for file in ["part_aa", "part_ab", "part_ac", "part_ad"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        subfolder="index",
        repo_type="dataset",
        local_dir=args.save_path,
    )

print(f"Index files saved to {args.save_path}")

repo_id = "slupart/qrecc-passages"

# Download all parquet shards
# for i in range(9):
#     hf_hub_download(
#         repo_id=repo_id,
#         filename=f"train-{i:05d}-of-00009.parquet",
#         subfolder="data",
#         repo_type="dataset",
#         local_dir=args.save_path,
#     )

# Download all parquet shards (50 files: train-000.parquet ~ train-049.parquet)
for i in range(55):
    hf_hub_download(
        repo_id=repo_id,
        filename=f"train-{i:03d}.parquet",   # 三位数字，如 train-000.parquet
        subfolder="data",
        repo_type="dataset",
        local_dir=args.save_path,
    )

# Merge shards into one JSONL with only id and contents
parquet_files = sorted(glob.glob(os.path.join(args.save_path, "data", "*.parquet")))
out_file = os.path.join(args.save_path, "qrecc.jsonl")


with open(out_file, "wb") as fout:
    for pf in parquet_files:
        pf_reader = pq.ParquetFile(pf)
        for batch in pf_reader.iter_batches(batch_size=50_000, columns=["id", "contents"]):
            rows = batch.to_pylist()
            fout.write(b"\n".join(orjson.dumps(r) for r in rows))
            fout.write(b"\n")

print(f"Collection saved to {out_file}")
