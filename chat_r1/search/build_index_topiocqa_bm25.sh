#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=index_topiocqa
##SBATCH --partition gpu
#SBATCH --partition gpu_a100
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=240gb #120gb
#SBATCH -c 8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate retriever

corpus_file=collection/topiocqa.jsonl
save_dir=collection/topiocqa/index_bm25

python index_builder.py \
    --retrieval_method bm25 \
    --corpus_path $corpus_file \
    --save_dir $save_dir
