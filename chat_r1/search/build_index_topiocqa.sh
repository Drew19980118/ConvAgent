#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=index_topiocqa
##SBATCH --partition gpu
#SBATCH --partition gpu_h100
##SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --begin=now+1hour
#SBATCH --time=2-00:00:00
#SBATCH --mem=360gb #120gb
#SBATCH -c 8
#SBATCH --output=slurm-%j.out

source ~/.bashrc
conda activate retriever

corpus_file=collection/topiocqa.jsonl
save_dir=collection/topiocqa/index
retriever_name=e5
retriever_model=intfloat/e5-base-v2

CUDA_VISIBLE_DEVICES=0,1 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
