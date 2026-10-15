import json
import math
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.eval import evaluate_pretrain_validation_loss
from scripts.train_pretrain import evaluate_val
from src.dataset import SFTDataset
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer


def tiny_config(vocab_size: int, seq_len: int = 16) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden_dim=32,
        max_seq_len=seq_len,
    )


class RecordingDataset(Dataset):
    """Sequence of tokens that records which indices the loader actually reads."""

    def __init__(self, length: int, vocab_size: int, seq_len: int = 8):
        self.length = length
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.accessed = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.accessed.append(idx)
        x = torch.full((self.seq_len,), idx % self.vocab_size, dtype=torch.long)
        return x, x


def test_val_eval_visits_all_shards_not_just_the_first(tmp_path):
    cfg = tiny_config(vocab_size=32)
    model = VakModel(cfg)
    val_dataset = RecordingDataset(length=128, vocab_size=cfg.vocab_size)
    loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    loss = evaluate_val(model, loader, torch.device("cpu"), vocab_size=cfg.vocab_size, max_batches=4)

    assert math.isfinite(loss)
    # 32 batches, 4 evaluated: the last batch must come from the end of the
    # validation set instead of the first four batches
    assert max(val_dataset.accessed) >= 96, val_dataset.accessed


def test_pretrain_val_sampling_and_bits_per_byte(tmp_path):
    tok = VakTokenizer()
    tok.train_from_iterator(["the quick brown fox jumps over the lazy dog"] * 5, vocab_size=200)

    cfg = tiny_config(vocab_size=tok.vocab_size, seq_len=64)
    model = VakModel(cfg)

    long_doc = "the quick brown fox jumps over the lazy dog " * 40  # far more than 64 tokens
    short_doc = "hello world"
    val_path = tmp_path / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for text in [long_doc, long_doc, short_doc, long_doc]:
            f.write(json.dumps({"text": text}) + "\n")

    metrics = evaluate_pretrain_validation_loss(
        model, tok, str(val_path), torch.device("cpu"), max_docs=2, seq_len=64
    )

    # stride = 4 // 2 = 2 -> documents 0 and 2 (a truncated long document and the
    # short one). First-N sampling would have scored two full 63-token documents.
    assert metrics["total_tokens"] < 2 * (64 - 1), metrics
    # Bytes must be counted for the retained text only: the full document is
    # longer than the 63 tokens that are actually scored
    assert metrics["total_bytes"] < len(long_doc.encode("utf-8")), metrics


def test_sft_sample_with_no_assistant_targets_is_dropped():
    tok = VakTokenizer()
    tok.train_from_iterator(["hello world", "how can I help you today"], vocab_size=200)

    long_user_turn = [
        {"role": "user", "content": "question " * 500},
        {"role": "assistant", "content": "hi"},
    ]
    dataset = SFTDataset([long_user_turn], tok, seq_len=64)
    assert len(dataset) == 0

    normal_turn = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert len(SFTDataset([normal_turn], tok, seq_len=64)) == 1
