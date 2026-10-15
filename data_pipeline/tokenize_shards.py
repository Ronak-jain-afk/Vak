import argparse
import glob
import json
import os
import random
from typing import Dict, Iterator, List, Optional
import numpy as np

from data_pipeline.manifest import ManifestManager, compute_file_sha256
from src.tokenizer import BOS_TOKEN, END_TOKEN, VakTokenizer


def balanced_text_iterator_from_jsonl(
    jsonl_path: str,
    samples_per_source: int = 10000,
    max_total_samples: int = 100000,
    seed: int = 42,
) -> Iterator[str]:
    """Yields a balanced sample of documents across all dataset sources.

    Reservoir sampling per source avoids head-of-stream bias (train.jsonl is
    ordered by source, so a naive first-N pass over-represents early dumps).
    """
    reservoirs: Dict[str, list] = {}
    rng = random.Random(seed)
    total_seen = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            src = doc.get("source", "default")
            batch = reservoirs.setdefault(src, [])
            total_seen += 1
            if len(batch) < samples_per_source:
                batch.append(doc["text"])
            else:
                j = rng.randrange(total_seen)
                if j < samples_per_source:
                    batch[j] = doc["text"]

    yielded = 0
    for batch in reservoirs.values():
        for text in batch:
            if yielded >= max_total_samples:
                return
            yielded += 1
            yield text


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
    samples_per_source: int = 10000,
    max_total_samples: int = 100000,
) -> VakTokenizer:
    """Trains VakTokenizer on a balanced multi-domain sample prior to pretraining."""
    tokenizer = VakTokenizer()
    print(f"Training tokenizer (vocab_size={vocab_size}) on balanced multi-domain sample from {train_jsonl}...")
    iterator = balanced_text_iterator_from_jsonl(
        train_jsonl,
        samples_per_source=samples_per_source,
        max_total_samples=max_total_samples,
    )
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
    manifest: Optional[ManifestManager] = None,
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
        if manifest is not None:
            sha = compute_file_sha256(path)
            manifest.record_artifact(path, f"{split_name} shard {s_idx:05d} ({len(tokens_list):,} tokens, sha256={sha[:12]})")
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


def main():
    parser = argparse.ArgumentParser(description="Tokenizer training and binary token sharding for Vak")
    parser.add_argument("--train_jsonl", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--output_dir", type=str, default="data/tokenized")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--samples_per_source", type=int, default=10000)
    parser.add_argument("--tokens_per_shard", type=int, default=10_000_000)
    parser.add_argument("--train_tokenizer", action="store_true", help="Retrain tokenizer before pretraining")
    parser.add_argument("--shard", action="store_true", help="Tokenize and write binary shards")
    args = parser.parse_args()

    manifest = ManifestManager("data/manifest.json")

    if args.train_tokenizer:
        train_vak_tokenizer(
            train_jsonl=args.train_jsonl,
            output_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
            samples_per_source=args.samples_per_source,
        )

    if args.shard:
        tok_file = os.path.join(args.tokenizer_dir, "tokenizer.json")
        if not os.path.exists(tok_file):
            raise FileNotFoundError(f"Tokenizer not found at {tok_file}. Train tokenizer first using --train_tokenizer.")
        tokenizer = VakTokenizer(tok_file)

        split_dir = os.path.dirname(os.path.abspath(args.train_jsonl))
        for split in ["train", "val", "test"]:
            path = os.path.join(split_dir, f"{split}.jsonl")
            if os.path.exists(path):
                print(f"Sharding {split} split from {path}...")
                res = shard_and_tokenize(
                    path,
                    tokenizer,
                    output_dir=args.output_dir,
                    split_name=split,
                    tokens_per_shard=args.tokens_per_shard,
                    manifest=manifest,
                )
                print(f"Result for {split}: {res}")


if __name__ == "__main__":
    main()
