import json
import os
import tempfile
import numpy as np
import pytest

from data_pipeline.manifest import ManifestManager
from data_pipeline.tokenize_shards import (
    balanced_text_iterator_from_jsonl,
    shard_and_tokenize,
    train_vak_tokenizer,
)
from src.tokenizer import BOS_TOKEN, END_TOKEN, VakTokenizer


def test_balanced_text_iterator():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "train.jsonl")
        sources = ["fineweb-edu", "cosmopedia-v2", "python-edu", "open-web-math", "tinystories"]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            # 50 docs for fineweb, 20 for others
            for s in sources:
                count = 50 if s == "fineweb-edu" else 20
                for i in range(count):
                    f.write(json.dumps({"text": f"Sample document {i} from {s}", "source": s}) + "\n")

        # Request 10 per source
        sampled = list(balanced_text_iterator_from_jsonl(jsonl_path, samples_per_source=10))
        assert len(sampled) == 50  # 10 from each of the 5 sources

        # Verify that all 5 sources are represented equally in the iterator
        source_counts = {s: 0 for s in sources}
        for text in sampled:
            for s in sources:
                if s in text:
                    source_counts[s] += 1
        for s in sources:
            assert source_counts[s] == 10, f"Source {s} was sampled {source_counts[s]} times, expected 10"


def test_tokenizer_training_on_balanced_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "train.jsonl")
        tok_dir = os.path.join(tmpdir, "tok")
        # Multi-domain sample lines
        docs = [
            {"text": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.", "source": "fineweb-edu"},
            {"text": "Chapter 3: Introduction to Newtonian Mechanics and Gravitation in Modern Physics.", "source": "cosmopedia-v2"},
            {"text": "def compute_fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return compute_fibonacci(n - 1) + compute_fibonacci(n - 2)", "source": "python-edu"},
            {"text": "Let $X$ be a Hilbert space and let $T: X \\to X$ be a bounded linear operator with spectrum $\\sigma(T)$.", "source": "open-web-math"},
            {"text": "Once upon a time, a little girl named Lily found a shiny red ball in the garden.", "source": "tinystories"},
        ]
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for d in docs * 20:
                f.write(json.dumps(d) + "\n")

        tok = train_vak_tokenizer(
            train_jsonl=jsonl_path,
            output_dir=tok_dir,
            vocab_size=300,
            samples_per_source=20,
        )
        assert tok.vocab_size == 300
        assert tok.bos_id is not None
        assert tok.end_id is not None

        # Verify encoding and decoding roundtrip across code, math, and prose
        test_code = "def compute_fibonacci(n: int) -> int:"
        ids = tok.encode(test_code)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert decoded == test_code


def test_shard_and_tokenize_boundaries():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "split.jsonl")
        out_dir = os.path.join(tmpdir, "shards")
        manifest_path = os.path.join(tmpdir, "manifest.json")
        manifest = ManifestManager(manifest_path)

        tok = VakTokenizer()
        tok.train_from_iterator(["hello world", "test code def return", "1 2 3"], vocab_size=100)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(15):
                f.write(json.dumps({"text": "hello world test code"}) + "\n")

        # Shard with small token capacity (e.g. 20 tokens per shard)
        stats = shard_and_tokenize(
            jsonl_path=jsonl_path,
            tokenizer=tok,
            output_dir=out_dir,
            split_name="test_split",
            tokens_per_shard=20,
            manifest=manifest,
        )
        assert stats["shards"] > 1
        assert stats["total_tokens"] > 0

        # Read back first shard as uint16
        first_shard = os.path.join(out_dir, "test_split_shard_00000.bin")
        assert os.path.exists(first_shard)
        arr = np.fromfile(first_shard, dtype=np.uint16)
        assert len(arr) == 20
        # The first token must be BOS_TOKEN
        assert arr[0] == tok.bos_id
