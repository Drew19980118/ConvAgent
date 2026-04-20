#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tensordict import TensorDict

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
)

# verl imports (same as your trainer)
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.dataset import SFTDataset


def strip_info_blocks(text: str) -> str:
    """Remove <information> and <search> blocks (content + tags)."""
    text = re.sub(r"<information>.*?</information>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<search>.*?</search>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def extract_answer_block(text: str) -> str:
    """If an <answer>...</answer> block exists, return its inside; else return whole text."""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def batch_prompts_and_gold(batch: TensorDict, tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
    """
    Given a batch with {input_ids, attention_mask, loss_mask}, split into prompt-only and gold.
    We use the first index where loss_mask==1 as the response start.
    """
    input_ids = batch["input_ids"]  # [B, T]
    loss_mask = batch["loss_mask"]  # [B, T]
    B, T = input_ids.shape

    results = []
    for i in range(B):
        lm = loss_mask[i].tolist()
        try:
            resp_start = lm.index(1)
        except ValueError:
            # Fallback: no supervised tokens; treat everything as prompt
            resp_start = T

        prompt_ids = input_ids[i, :resp_start]
        gold_ids   = input_ids[i, resp_start:]

        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        gold_text   = tokenizer.decode(gold_ids,   skip_special_tokens=True)

        results.append({
            "prompt_ids": prompt_ids.unsqueeze(0).to(input_ids.device),  # [1, Lp]
            "prompt_attn": torch.ones_like(prompt_ids, device=input_ids.device).unsqueeze(0),  # [1, Lp]
            "prompt_text": prompt_text,
            "gold_text": gold_text,
        })
    return results


def left_pad_batch(seqs: List[torch.Tensor], pad_id: int):
    """
    Left-pad a list of 1D CUDA LongTensors to the same length.
    Returns (input_ids[B, T], attention_mask[B, T]).
    """
    max_len = max(s.size(0) for s in seqs)
    padded, attn = [], []
    for s in seqs:
        pad_len = max_len - s.size(0)
        if pad_len > 0:
            pad_tokens = torch.full((pad_len,), pad_id, dtype=s.dtype, device=s.device)
            pad_mask   = torch.zeros(pad_len, dtype=torch.long, device=s.device)
            padded.append(torch.cat([pad_tokens, s], dim=0))
            attn.append(torch.cat([pad_mask, torch.ones_like(s, dtype=torch.long, device=s.device)], dim=0))
        else:
            padded.append(s)
            attn.append(torch.ones_like(s, dtype=torch.long, device=s.device))
    return torch.stack(padded, dim=0), torch.stack(attn, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Single-GPU generation from an SFT checkpoint.")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to checkpoint directory saved by save_checkpoint (contains config.json, model weights, tokenizer).")
    parser.add_argument("--val_files", type=str, required=True,
                        help="Comma-separated list or a glob passed through shell of parquet files for evaluation.")
    parser.add_argument("--prompt_key", type=str, required=True,
                        help="Key in the data for the prompt (e.g., 'prompt').")
    parser.add_argument("--response_key", type=str, required=True,
                        help="Key in the data for the response (e.g., 'response').")
    parser.add_argument("--prompt_dict_keys", type=str, default="",
                        help="Optional comma-sep dict keys to stitch prompt from; leave empty to ignore.")
    parser.add_argument("--response_dict_keys", type=str, default="",
                        help="Optional comma-sep dict keys to stitch response from; leave empty to ignore.")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length for tokenization.")
    parser.add_argument("--truncation", action="store_true", help="Enable truncation in SFTDataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-batch generation batch size (prompts).")
    parser.add_argument("--max_batches", type=int, default=None, help="Stop early after this many batches.")
    parser.add_argument("--data_source_name", type=str, default="unknown", help="Populates 'data_source' in outputs.")

    # generation args
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--clean_answer", action="store_true",
                        help="If set, strip <information>/<search> blocks and extract <answer>...</answer>.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output", type=str, required=True, help="Path to write JSON results.")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "This script is intended for a single CUDA GPU."

    # ---- Resolve model path (supports HDFS via copy_local_path_from_hdfs for consistency) ----
    local_ckpt = copy_local_path_from_hdfs(src=args.ckpt_dir, verbose=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(local_ckpt, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        # fallback: use eos as pad
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- Model (eager kernels for safer generation) ----
    base_config = AutoConfig.from_pretrained(local_ckpt, trust_remote_code=args.trust_remote_code)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        local_ckpt,
        config=base_config,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,  # bfloat16 for speed/VRAM; change to float16 if needed
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = True
    torch.set_grad_enabled(False)

    # ---- Dataset / Loader ----
    val_files = [s for s in args.val_files.split(",") if s.strip()]
    prompt_dict_keys = [k for k in args.prompt_dict_keys.split(",") if k.strip()]
    response_dict_keys = [k for k in args.response_dict_keys.split(",") if k.strip()]

    val_dataset = SFTDataset(
        parquet_files=val_files,
        tokenizer=tokenizer,
        prompt_key=args.prompt_key,
        prompt_dict_keys=prompt_dict_keys if prompt_dict_keys else None,
        response_key=args.response_key,
        response_dict_keys=response_dict_keys if response_dict_keys else None,
        max_length=args.max_length,
        truncation=args.truncation,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # ---- GenerationConfig ----
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = args.max_new_tokens
    gen_cfg.num_beams = args.num_beams
    gen_cfg.pad_token_id = pad_id
    gen_cfg.eos_token_id = eos_id
    if args.do_sample:
        gen_cfg.do_sample = True
        gen_cfg.temperature = args.temperature
        gen_cfg.top_p = args.top_p
    else:
        gen_cfg.do_sample = False
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None

    # ---- Run ----
    results: List[Dict[str, Any]] = []
    total_batches = len(val_loader)
    print(f"[eval-gen] single-GPU: {total_batches} batches @ bs={args.batch_size}, "
          f"max_new_tokens={args.max_new_tokens}, beams={args.num_beams}, do_sample={args.do_sample}",
          flush=True)

    for b_idx, batch in enumerate(val_loader, start=1):
        if args.max_batches is not None and b_idx > int(args.max_batches):
            print(f"[eval-gen] reached --max_batches={args.max_batches}, stopping early.", flush=True)
            break

        bsz = batch["input_ids"].size(0)
        batch = TensorDict.from_dict(batch, batch_size=[bsz]).cuda()

        items = batch_prompts_and_gold(batch, tokenizer)
        seqs = [x["prompt_ids"].squeeze(0) for x in items]
        input_ids, attention = left_pad_batch(seqs, pad_id)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                generation_config=gen_cfg,
            )

        gen_texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)

        for i, it in enumerate(items):
            full = gen_texts[i]
            prefix = it["prompt_text"].strip()
            raw_pred = full.strip()

            # cut off the prompt prefix if model copied it
            if prefix and raw_pred.startswith(prefix):
                raw_pred = raw_pred[len(prefix):].lstrip()

            predicted = raw_pred
            if args.clean_answer:
                predicted = strip_info_blocks(predicted)
                predicted = extract_answer_block(predicted)

            results.append({
                "data_source": args.data_source_name,
                "question": it["prompt_text"],
                "golden_answers": [it["gold_text"]],
                "predicted_answer": predicted,
                "raw_predicted_answer": raw_pred,
            })

        if (b_idx % 10) == 0:
            print(f"[eval-gen] done {b_idx}/{total_batches} batches…", flush=True)

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[eval-gen] wrote {len(results)} items to {args.output}", flush=True)


if __name__ == "__main__":
    # Some users like long NCCL timeouts; harmless here but OK to mirror your env:
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("NCCL_TIMEOUT", "10800")
    main()
