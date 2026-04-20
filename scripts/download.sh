
save_path=collection

python download.py --save_path "$save_path"

cat "$save_path"/index/part_* > "$save_path"/e5_Flat.index

python scripts/download_data_conv.py
