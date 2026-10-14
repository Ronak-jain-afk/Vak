import glob
import json
import os
from typing import Dict, Iterator, List
import numpy as np

from src.tokenizer import BOS_TOKEN, END_TOKEN, VakTokenizer


def text_iterator_from_jsonl(jsonl_path: str, max_samples: int = 50000) -> Iterator[str]:
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            yield doc["text"]
            count += 1
            if count >= max_samples:
                break


def train_vak_tokenizer(
    train_jsonl: str,
    output_dir: str = "tokenizer",
    vocab_size: int = 8192,
    max_train_samples: int = 50000,
) -> VakTokenizer:
    tokenizer = VakTokenizer()
    iterator = text_iterator_from_jsonl(train_jsonl, max_samples=max_train_samples)
    tokenizer.train_from_iterator(iterator, vocab_size=vocab_size)
    tok_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"Tokenizer trained with vocab_size={tokenizer.vocab_size} and saved to {tok_path}")
    return tokenizer


def shard_and_tokenize(
    jsonl_path: str,
    tokenizer: VakTokenizer,
    output_dir: str,
    split_name: str,
    tokens_per_shard: int = 10_000_000,
) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    shard_idx = 0
    buffer = []
    total_tokens = 0
    total_docs = 0

    def flush_shard(tokens_list: List[int], s_idx: int) -> str:
        arr = np.array(tokens_list, dtype=np.uint16)
        path = os.path.join(output_dir, f"{split_name}_shard_{s_idx:05d}.bin")
        with open(path, "wb") as f:
            f.write(arr.tobytes())
        return path

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            text = doc["text"]
            doc_tokens = [tokenizer.bos_id] + tokenizer.encode(text) + [tokenizer.end_id]
            buffer.extend(doc_tokens)
            total_tokens += len(doc_tokens)
            total_docs += 1

            while len(buffer) >= tokens_per_shard:
                flush_shard(buffer[:tokens_per_shard], shard_idx)
                buffer = buffer[tokens_per_shard:]
                shard_idx += 1

    # Flush remainder
    if buffer:
        flush_shard(buffer, shard_idx)
        shard_idx += 1

    return {
        "split": split_name,
        "shards": shard_idx,
        "total_tokens": total_tokens,
        "total_docs": total_docs,
    }
