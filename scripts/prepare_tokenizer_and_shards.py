import argparse
import os
import yaml

from data_pipeline.manifest import ManifestManager
from data_pipeline.tokenize_shards import shard_and_tokenize, train_vak_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer and generate binary shards")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    model_cfg = cfg["model"]

    manifest = ManifestManager("data/manifest.json")
    train_jsonl = os.path.join(paths["processed_data_dir"], "train.jsonl")
    val_jsonl = os.path.join(paths["processed_data_dir"], "val.jsonl")

    # 1. Train Tokenizer
    print("Training Vak Byte-level BPE Tokenizer (Vocab size: 8,192)...")
    tokenizer = train_vak_tokenizer(
        train_jsonl=train_jsonl,
        output_dir=paths["tokenizer_dir"],
        vocab_size=model_cfg["vocab_size"],
    )

    manifest.record_tokenization({
        "tokenizer_type": "byte-level BPE",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": ["<|bos|>", "<|user|>", "<|assistant|>", "<|end|>"],
    })
    manifest.record_artifact(
        os.path.join(paths["tokenizer_dir"], "tokenizer.json"),
        "Trained Byte-level BPE Tokenizer",
    )

    # 2. Shard and tokenize train split
    print("\nTokenizing and sharding train split...")
    train_stats = shard_and_tokenize(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        output_dir=paths["tokenized_dir"],
        split_name="train",
        tokens_per_shard=10_000_000,
    )
    print(f"Train sharding stats: {train_stats}")

    # 3. Shard and tokenize val split
    print("\nTokenizing and sharding val split...")
    val_stats = shard_and_tokenize(
        jsonl_path=val_jsonl,
        tokenizer=tokenizer,
        output_dir=paths["tokenized_dir"],
        split_name="val",
        tokens_per_shard=1_000_000,
    )
    print(f"Val sharding stats: {val_stats}")

    manifest.record_processing("sharding_and_tokenization", {
        "train": train_stats,
        "val": val_stats,
    })

    print("\nTokenizer training and binary sharding successfully completed!")


if __name__ == "__main__":
    main()
