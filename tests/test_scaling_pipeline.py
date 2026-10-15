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


def test_balanced_iterator_reservoir_is_deterministic_and_spread():
    import tempfile as _tmp
    from data_pipeline.tokenize_shards import balanced_text_iterator_from_jsonl as _it
    with _tmp.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "t.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for i in range(100):
                f.write(json.dumps({"text": f"doc number {i} filler words here", "source": "s"}) + "\n")
        first = list(_it(p, samples_per_source=10, seed=42))
        second = list(_it(p, samples_per_source=10, seed=42))
        assert first == second
        assert len(first) == 10
        # Reservoir must reach deep into the stream (head-only sampling caps at doc 9)
        idx = sorted(int(t.split()[2]) for t in first)
        assert idx[-1] >= 50, idx


def test_balanced_iterator_reservoir_covers_partner_sources():
    import tempfile as _tmp
    from data_pipeline.tokenize_shards import balanced_text_iterator_from_jsonl as _it
    with _tmp.TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "t.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            # A short source first, then a long one: the reservoir position must be
            # counted per source, otherwise the long source is only ever sampled
            # from its first samples_per_source documents
            for i in range(10):
                f.write(json.dumps({"text": f"short doc {i} filler words here", "source": "short"}) + "\n")
            for i in range(400):
                f.write(json.dumps({"text": f"long doc {i} filler words here", "source": "long"}) + "\n")

        sampled = list(_it(p, samples_per_source=10, seed=42))
        long_idx = sorted(int(t.split()[2]) for t in sampled if t.startswith("long doc"))
        assert len(long_idx) == 10
        assert long_idx[-1] >= 200, long_idx


def test_deduplicator_bounded_window_and_eviction():
    from data_pipeline.deduplicate import Deduplicator

    # Create a deduplicator with a tiny window of 2 documents
    dedup = Deduplicator(threshold=0.70, max_lsh_window=2)
    doc0 = "The quick brown fox jumps over the lazy dog repeatedly in the morning sunlight across the green field."
    doc1 = "Artificial intelligence and machine learning are revolutionizing software development across all industries."
    doc2 = "Quantum computing harnesses the strange laws of physics to perform calculations beyond classical supercomputers."

    # Insert doc0 and doc1
    is_d0, _ = dedup.is_duplicate(doc0, doc_id="d0")
    is_d1, _ = dedup.is_duplicate(doc1, doc_id="d1")
    assert not is_d0
    assert not is_d1

    # Near match to doc0 while it's in the window
    near_doc0 = "The quick brown fox jumps over the lazy dog repeatedly in the morning sunlight across the quiet field."
    is_near, reason = dedup.is_duplicate(near_doc0, doc_id="near_d0")
    assert is_near
    assert "near_match_with_d0" in reason

    # Insert doc2 -> this pushes window count to 2 (d1, d2) and evicts d0 from LSH
    is_d2, _ = dedup.is_duplicate(doc2, doc_id="d2")
    assert not is_d2
    assert len(dedup.lsh_window) == 2
    assert "d0" not in dedup.lsh_window

    # Now near_doc0 should no longer trigger a near match since d0 was evicted
    another_near_doc0 = "The quick brown fox jumps over the lazy dog repeatedly in the morning sunlight across the golden field."
    is_near2, _ = dedup.is_duplicate(another_near_doc0, doc_id="another_near")
    assert not is_near2

    # But exact match to doc0 is still caught globally
    is_exact, reason = dedup.is_duplicate(doc0, doc_id="exact_d0")
    assert is_exact
    assert reason == "exact"

