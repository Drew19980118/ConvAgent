import os
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Union, Any, Dict
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
import datasets as hf_datasets

class SFTDataset(Dataset):
    """
    Supervised FT dataset for causal-LM training.

    Produces dicts with:
      - input_ids:      LongTensor [max_length]
      - attention_mask: LongTensor [max_length]
      - position_ids:   LongTensor [max_length]
      - loss_mask:      FloatTensor [max_length]   (0 on prompt & padding, 1 on answer tokens)

    It supports two ways to locate text:
      1) Flat fields:
         - prompt_key="question", response_key="answer"
      2) Nested fields via dict/list paths:
         - prompt_key="prompt", prompt_dict_keys=["0","content"]
         - response_dict_keys=["reward_model","ground_truth","target","0"]

    Truncation behavior:
      - truncation="error": raise if combined length > max_length
      - truncation="right": truncate from the right (keeps the beginning)
      - truncation="left":  truncate from the left (keeps the end)
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key: Optional[str],
        response_key: Optional[str],
        max_length: int = 1024,
        truncation: str = "error",
        prompt_dict_keys: Optional[Union[List[Union[str,int]], str]] = None,
        response_dict_keys: Optional[Union[List[Union[str,int]], str]] = None,
        balance_dp_token: bool = False,   # accepted but unused here; trainer handles it
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.truncation = truncation
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.prompt_path = self._normalize_path(prompt_dict_keys)
        self.response_path = self._normalize_path(response_dict_keys)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            # Safe default: use eos as pad if pad missing
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load parquet(s)
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        dsets = []
        for pf in parquet_files:
            # datasets.load_dataset can stream parquet or local paths
            # Using 'parquet' builder for robustness
            # If pf is a directory with *.parquet, pass a dict of data_files
            if os.path.isdir(pf):
                files = [os.path.join(pf, f) for f in os.listdir(pf) if f.endswith(".parquet")]
                if not files:
                    raise FileNotFoundError(f"No parquet files under directory: {pf}")
                d = load_dataset("parquet", data_files={"train": files})["train"]
            else:
                d = load_dataset("parquet", data_files={"train": pf})["train"]
            dsets.append(d)

        self.ds: HFDataset = dsets[0] if len(dsets) == 1 else concatenate_datasets(dsets)

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _normalize_path(path: Optional[Union[List[Union[str,int]], str]]) -> Optional[List[Union[str,int]]]:
        if path is None:
            return None
        if isinstance(path, str):
            # Hydra may pass like "[a,b,c]" already parsed into list; but if string, split on commas?
            # Expect explicit list string like "[reward_model,ground_truth,target,0]"
            s = path.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
                if not s:
                    return []
                parts = [p.strip() for p in s.split(",")]
            else:
                parts = [s]
            # Convert numeric strings to int indices when possible
            norm = []
            for p in parts:
                if p.isdigit():
                    norm.append(int(p))
                else:
                    norm.append(p)
            return norm
        # list provided
        norm = []
        for p in path:
            if isinstance(p, str) and p.isdigit():
                norm.append(int(p))
            else:
                norm.append(p)
        return norm

    @staticmethod
    def _dig(obj: Any, path: Optional[List[Union[str,int]]]) -> Any:
        """Traverse nested dict/list by path. Returns None if any hop missing."""
        if path is None or path == []:
            return obj
        cur = obj
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            elif isinstance(cur, list) and isinstance(key, int) and 0 <= key < len(cur):
                cur = cur[key]
            else:
                return None
        return cur

    def _extract_prompt(self, row: Dict[str, Any]) -> Optional[str]:
        # Preferred: use prompt_key; if nested expected, start from row[prompt_key] then follow prompt_path
        if self.prompt_key is not None and self.prompt_key in row:
            x = row[self.prompt_key]
            if self.prompt_path:
                x = self._dig(x, self.prompt_path)
            # If "prompt" is a chat list but path not provided, try best-effort:
            if isinstance(x, list):
                # common case: [{"role":"user","content": "..."}]
                if len(x) > 0 and isinstance(x[0], dict) and "content" in x[0]:
                    x = x[0]["content"]
                else:
                    x = str(x)
            return None if x is None else str(x)

        # Fallback: common fields
        for key in ("question", "prompt_text"):
            if key in row and isinstance(row[key], str):
                return row[key]
        return None

    def _extract_answer(self, row: Dict[str, Any]) -> Optional[str]:
        # Preferred: response_key + response_path
        if self.response_key is not None and self.response_key in row:
            y = row[self.response_key]
            if self.response_path:
                y = self._dig(y, self.response_path)
            # If list of answers, pick the first
            if isinstance(y, list) and y:
                y = y[0]
            return None if y is None else str(y)

        # If response_key is None, try nested path from root
        if self.response_key is None and self.response_path:
            y = self._dig(row, self.response_path)
            if isinstance(y, list) and y:
                y = y[0]
            return None if y is None else str(y)

        # Fallback: common fields
        for key in ("answer", "gold", "target"):
            if key in row:
                y = row[key]
                if isinstance(y, list) and y:
                    y = y[0]
                return None if y is None else str(y)
        return None

    def _truncate_pair(self, ids_a: List[int], ids_b: List[int]) -> (List[int], List[int]):
        """Ensure len(a)+len(b) <= max_length by truncating according to self.truncation."""
        total = len(ids_a) + len(ids_b)
        if total <= self.max_length:
            return ids_a, ids_b

        if self.truncation == "error":
            raise ValueError(f"Sequence length {total} exceeds max_length={self.max_length}")

        # Number of tokens to remove
        overflow = total - self.max_length
        if self.truncation == "right":
            # Drop from the end of the prompt first (keep answer intact if possible)
            if overflow <= len(ids_a):
                return ids_a[:-overflow], ids_b
            else:
                # If prompt fully gone, truncate from the answer tail
                rem = overflow - len(ids_a)
                return [], ids_b[:-rem] if rem < len(ids_b) else []
        elif self.truncation == "left":
            # Drop from the beginning of the prompt first
            if overflow <= len(ids_a):
                return ids_a[overflow:], ids_b
            else:
                rem = overflow - len(ids_a)
                return [], ids_b[rem:] if rem < len(ids_b) else []
        else:
            raise ValueError(f"Unsupported truncation mode: {self.truncation}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.ds[idx]

        prompt_text = self._extract_prompt(row)
        answer_text = self._extract_answer(row)

        if prompt_text is None or answer_text is None:
            # Skip invalid rows by returning an all-pad sample (DataLoader still needs a dict)
            # but better to raise—keeps visibility.
            raise ValueError(f"Missing prompt/answer at index {idx}")

        # Tokenize separately (so we can mask loss on answer only)
        # Do NOT pad here; pad manually to fixed max_length later.
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # Optionally add a separator before answer; many setups include a newline or space.
        sep = ""
        answer_with_eos = answer_text
        if getattr(self.tokenizer, "eos_token", None) is not None:
            answer_with_eos = answer_with_eos + self.tokenizer.eos_token
        answer_ids = self.tokenizer.encode(answer_with_eos, add_special_tokens=False)

        # Truncate pair if needed
        prompt_ids, answer_ids = self._truncate_pair(prompt_ids, answer_ids)

        # Build combined and masks
        input_ids = prompt_ids + answer_ids
        seq_len = len(input_ids)

        # Pad to max_length
        pad_len = self.max_length - seq_len
        pad_id = self.tokenizer.pad_token_id
        input_ids = input_ids + [pad_id] * pad_len

        attention_mask = [1] * seq_len + [0] * pad_len

        # Loss mask: 0 on prompt tokens, 1 on answer tokens, 0 on padding
        loss_mask = [0.0] * len(prompt_ids) + [1.0] * len(answer_ids) + [0.0] * pad_len

        # Position ids: 0..(max_length-1)
        position_ids = list(range(self.max_length))

        # Convert to tensors
        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        }
        return out
