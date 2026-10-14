import glob
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer import ASSISTANT_TOKEN, BOS_TOKEN, END_TOKEN, USER_TOKEN, VakTokenizer


class ShardedPretrainDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 1024, pattern: str = "train_*.bin"):
        self.seq_len = seq_len
        self.shard_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.shard_paths:
            self.shard_paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not self.shard_paths:
            raise FileNotFoundError(f"No matching shards ({pattern}) found in {data_dir}")

        self.shard_lengths = []
        self.cumulative_lengths = [0]
        total_samples = 0

        for path in self.shard_paths:
            file_bytes = os.path.getsize(path)
            num_tokens = file_bytes // 2  # uint16 is 2 bytes per token
            # Number of non-overlapping sequences of length seq_len
            num_samples = max(0, (num_tokens - 1) // seq_len)
            self.shard_lengths.append(num_samples)
            total_samples += num_samples
            self.cumulative_lengths.append(total_samples)

        self.total_samples = total_samples
        self._current_shard_idx = -1
        self._current_mmap = None

    def __len__(self) -> int:
        return self.total_samples

    def _get_mmap(self, shard_idx: int) -> np.memmap:
        if shard_idx != self._current_shard_idx:
            self._current_shard_idx = shard_idx
            self._current_mmap = np.memmap(self.shard_paths[shard_idx], dtype=np.uint16, mode="r")
        return self._current_mmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_samples}")

        shard_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side="right")
        local_idx = idx - self.cumulative_lengths[shard_idx]

        mmap_arr = self._get_mmap(shard_idx)
        start_token = local_idx * self.seq_len
        end_token = start_token + self.seq_len + 1

        tokens = torch.from_numpy(mmap_arr[start_token:end_token].astype(np.int64))
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


class SFTDataset(Dataset):
    def __init__(self, conversations: List[List[Dict[str, str]]], tokenizer: VakTokenizer, seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        for conv in conversations:
            item = self._process_conversation(conv)
            if item is not None:
                self.samples.append(item)

    def _process_conversation(self, conv: List[Dict[str, str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Format tokens and construct target tensor with loss computed ONLY on assistant responses
        all_token_ids = [self.tokenizer.bos_id]
        all_loss_masks = [0]  # 0: ignore, 1: compute loss

        for turn in conv:
            role = turn["role"]
            content = turn["content"].strip()
            if role == "user":
                turn_str = f"{USER_TOKEN}{content}{END_TOKEN}"
                turn_ids = self.tokenizer.encode(turn_str)
                all_token_ids.extend(turn_ids)
                all_loss_masks.extend([0] * len(turn_ids))
            elif role == "assistant":
                turn_str = f"{ASSISTANT_TOKEN}{content}{END_TOKEN}"
                turn_ids = self.tokenizer.encode(turn_str)
                # Compute loss on assistant content and its closing tag, not the role tag itself
                prefix_ids = self.tokenizer.encode(ASSISTANT_TOKEN)
                prefix_len = len(prefix_ids)
                all_token_ids.extend(turn_ids)
                # Mask prefix tag, keep assistant tokens
                all_loss_masks.extend([0] * prefix_len + [1] * (len(turn_ids) - prefix_len))

        if len(all_token_ids) < 2:
            return None

        # Truncate to seq_len + 1 if needed
        all_token_ids = all_token_ids[: self.seq_len + 1]
        all_loss_masks = all_loss_masks[: self.seq_len + 1]

        input_ids = all_token_ids[:-1]
        target_ids = all_token_ids[1:]
        target_masks = all_loss_masks[1:]

        # Apply -100 to positions where loss should be ignored
        labels = [tgt if mask == 1 else -100 for tgt, mask in zip(target_ids, target_masks)]

        # Pad to seq_len
        pad_len = self.seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.tokenizer.end_id] * pad_len)
            labels.extend([-100] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
