import pytest
import torch

from src.generate import generate_response
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer


@pytest.fixture
def chat_setup():
    tok = VakTokenizer()
    sample_texts = [
        "Hello! How are you?",
        "I am doing well, thank you.",
        "What is your name?",
        "My name is Vak.",
        "This is a longer document explaining how tokenizers work with byte level pairs.",
    ]
    tok.train_from_iterator(sample_texts, vocab_size=500)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden_dim=256,
        max_seq_len=1024,
    )
    model = VakModel(cfg)
    return tok, model


def test_empty_input_handling(chat_setup):
    tok, _ = chat_setup
    tokens, err = tok.prepare_chat_input([])
    assert tokens is None
    assert "Empty input" in err

    tokens2, err2 = tok.prepare_chat_input([{"role": "user", "content": "   "}])
    assert tokens2 is None
    assert "non-empty user message" in err2


def test_turn_truncation_fits_max_prompt(chat_setup):
    tok, _ = chat_setup
    # Create multi-turn conversation that exceeds small max_prompt_len limit (e.g. 50 tokens)
    messages = [
        {"role": "user", "content": "Hello! How are you?"},
        {"role": "assistant", "content": "I am doing well, thank you."},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name is Vak."},
        {"role": "user", "content": "Tell me more."},
    ]

    tokens, err = tok.prepare_chat_input(messages, max_prompt_len=25)
    assert err is None
    assert tokens is not None
    assert len(tokens) <= 25


def test_single_turn_exceeding_max_prompt_returns_error(chat_setup):
    tok, _ = chat_setup
    huge_message = "What is your name? " * 100
    messages = [{"role": "user", "content": huge_message}]

    tokens, err = tok.prepare_chat_input(messages, max_prompt_len=50)
    assert tokens is None
    assert "exceeds maximum prompt length" in err


def test_non_english_input_generation(chat_setup):
    tok, model = chat_setup
    messages = [{"role": "user", "content": "Bonjour le monde! ¿Cómo estás?"}]
    tokens, err = tok.prepare_chat_input(messages, max_prompt_len=768)
    assert err is None
    assert tokens is not None

    # Should generate without crashing
    response = generate_response(
        model,
        tok,
        tokens,
        max_new_tokens=10,
        temperature=0.7,
        top_p=0.9,
    )
    assert isinstance(response, str)
