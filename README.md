# ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.13312)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/collections/slupart/chatr1)

## Installation

### ChatR1 environment
```bash
conda create -n chatr1 python=3.9
conda activate chatr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment
```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

## Quick start

Train a reasoning + search LLM on TopiOCQA dataset with e5 as the retriever tool.

(1) Download the indexing and corpus.
```bash
save_path=collection
python scripts/download.py --save_path $save_path
rm "$save_path"/data/*.parquet

cat "$save_path"/index/part_* > "$save_path"/topiocqa/index/e5_Flat.index
rm "$save_path"/index/part_*
```

(2) Download the TopiOCQA dataset.
```bash
python scripts/download_data_conv.py
```

(3) This launch a local retrieval server in the background and then train ChatR1.
```bash
sbatch train_ppo_3b_topiocqa_retrieval.sh
```
(This uses the address: http://127.0.0.1:8002/retrieve for retrieval)

## Inference
#### You can evaluate our ChatR1 trained model on TopiOCQA
(1) Download ChatR1 model weights.
```bash
hf download \
  slupart/ChatR1-topiocqa-qwen2.5-3b-it-ppo \
  --local-dir verl_checkpoints/chatr1-topiocqa-qwen2.5-3b-it-ppo \
  --local-dir-use-symlinks False
```

(2) Load retrieval and run ChatR1.
```bash
sbatch eval_topiocqa.sh
```

(3) Evaluate with F1 score and reference answers.
```bash
run="verl_checkpoints/chatr1-topiocqa-qwen2.5-3b-it-ppo/val_topiocqa/validation_results_step0.json"
python scripts/eval_f1.py --predictions "$run"
```

#### or play with the trained ChatR1 model with your own questions.
(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch_topiocqa.sh &
```

(2) Run inference on single question.
```bash
conda activate chatr1
python infer.py
```

## More Resources

Additional ChatR1 resources are available in the Hugging Face collection of [ChatR1](https://huggingface.co/collections/slupart/chatr1).


- All datasets collection in jsonl format for TopiOCQA, QReCC, INSCIT, MultiDoc2Dial and FaithDial.
- All E5 retrieval indexes are also provided for the datasets above.
- ChatR1 checkpoints for models trained on QReCC and TopiOCQA on 3B and 7B backbones.
- A unified dataset format for training and evaluation conversation of TopiOCQA, QReCC, INSCIT, MultiDoc2Dial and FaithDial in [here](https://huggingface.co/datasets/slupart/chatr1-convqa-all)


## Acknowledge

ChatR1 is built upon [Search R1](https://github.com/PeterGriffinJin/Search-R1), and inspired by [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main).
Its implementation is built from [Search R1](https://github.com/PeterGriffinJin/Search-R1) using [veRL](https://github.com/volcengine/verl) and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main).


## Citations

```bibtex
@article{lupart2025chatr1,
  title={Chatr1: Reinforcement learning for conversational reasoning and retrieval augmented question answering},
  author={Lupart, Simon and Aliannejadi, Mohammad and Kanoulas, Evangelos},
  journal={arXiv preprint arXiv:2510.13312},
  year={2025}
}
```

