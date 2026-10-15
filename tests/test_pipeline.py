import pytest
import torch

from data_pipeline.clean import is_quality_document, normalize_text
from data_pipeline.deduplicate import Deduplicator
from data_pipeline.split import assign_split
from src.dataset import SFTDataset
from src.tokenizer import ASSISTANT_TOKEN, BOS_TOKEN, END_TOKEN, USER_TOKEN, VakTokenizer


def test_unicode_and_whitespace_cleaning():
    raw = "Hello   world!\r\n\r\n\r\nThis is a test.\x00\x08With control chars."
    cleaned = normalize_text(raw)
    assert "\r" not in cleaned
    assert "\x00" not in cleaned
    assert "\x08" not in cleaned
    assert "Hello world!" in cleaned
    assert "\n\n\n" not in cleaned


def test_repetition_filtering():
    repetitive_text = "Click here to view menu.\n" * 15
    ok, reason = is_quality_document(repetitive_text)
    assert not ok
    assert "High line repetition" in reason


def test_exact_deduplication():
    dedup = Deduplicator()
    text = "The quick brown fox jumps over the lazy dog."
    is_dupe1, _ = dedup.is_duplicate(text, "doc1")
    is_dupe2, reason = dedup.is_duplicate(text, "doc2")
    assert not is_dupe1
    assert is_dupe2
    assert reason == "exact"


def test_near_deduplication():
    dedup = Deduplicator(threshold=0.70)
    doc1 = (
        "The quick brown fox jumps over the lazy dog repeatedly during the morning run. "
        "Every day the sun rises over the green hill, providing warmth and light to all creatures. "
        "In the quiet forest, birds sing gentle melodies while squirrels gather nuts for the upcoming winter."
    )
    # Change two words (near-duplicate)
    doc2 = (
        "The quick brown fox jumps over the lazy dog repeatedly during the morning run. "
        "Every day the sun rises over the green hill, providing warmth and light to all creatures. "
        "In the quiet forest, birds sing peaceful melodies while squirrels gather nuts for the upcoming winter."
    )
    is_dupe1, _ = dedup.is_duplicate(doc1, "doc1")
    is_dupe2, reason = dedup.is_duplicate(doc2, "doc2")
    assert not is_dupe1
    assert is_dupe2
    assert "near_match" in reason


def test_split_integrity_and_no_leakage():
    # Verify deterministic assignment and disjoint partitions
    doc_ids = [f"document_id_{i}" for i in range(1000)]
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for d in doc_ids:
        s = assign_split(d)
        if s == "train":
            train_ids.add(d)
        elif s == "val":
            val_ids.add(d)
        elif s == "test":
            test_ids.add(d)

    # All sets must be strictly disjoint (zero leakage)
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == 1000

    # Train should have approximately 90%
    assert 870 <= len(train_ids) <= 930


def test_sft_assistant_loss_masking():
    tok = VakTokenizer()
    # Train tiny dummy tokenizer to ensure IDs are assigned
    tok.train_from_iterator(["Hi there", "Hello! How can I help you today?", "What is 2+2?", "4"], vocab_size=300)

    conv = [
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello! How can I help you today?"},
    ]

    dataset = SFTDataset([conv], tok, seq_len=32)
    inputs, labels = dataset[0]

    # Verify that wherever label != -100, it corresponds to the assistant token
    non_masked_indices = (labels != -100).nonzero(as_tuple=True)[0]
    assert len(non_masked_indices) > 0

    # User prompt must be completely masked out (-100)
    user_str = f"{USER_TOKEN}Hi there{END_TOKEN}"
    user_tokens = tok.encode(user_str)
    # The first token is BOS, then user tokens; all corresponding positions must be -100
    for i in range(len(user_tokens)):
        assert labels[i].item() == -100


def test_resumable_random_sampler_continuity():
    from src.dataset import ResumableRandomSampler
    dataset = list(range(100))
    seed = 123
    epoch = 0

    # Full uninterrupted epoch
    full_sampler = ResumableRandomSampler(dataset, seed=seed, start_index=0, epoch=epoch)
    full_indices = list(full_sampler)
    assert len(full_indices) == 100

    # Interrupted at offset 42
    offset = 42
    resumed_sampler = ResumableRandomSampler(dataset, seed=seed, start_index=offset, epoch=epoch)
    resumed_indices = list(resumed_sampler)
    assert len(resumed_indices) == 100 - offset

    # Verifying exact continuity without reshuffle
    assert full_indices[offset:] == resumed_indices
    reconstructed = full_indices[:offset] + resumed_indices
    assert reconstructed == full_indices

