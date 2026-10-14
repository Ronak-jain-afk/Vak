import math
import os
import tempfile
import pytest
import torch
import torch.nn.functional as F

from src.model import ModelConfig, VakModel
from src.optim import build_optimizers


@pytest.fixture
def base_config():
    return ModelConfig(
        vocab_size=8192,
        d_model=384,
        n_layers=16,
        n_heads=6,
        n_kv_heads=2,
        ffn_hidden_dim=1024,
        max_seq_len=1024,
        rope_theta=10000.0,
        norm_eps=1e-5,
        init_std=0.02,
        tie_embeddings=True,
    )


def test_parameter_count(base_config):
    model = VakModel(base_config)
    num_params = model.count_parameters()
    # Acceptance range: 28.0M to 30.0M counted parameters
    print(f"Total counted parameters: {num_params:,}")
    assert 28_000_000 <= num_params <= 30_000_000, f"Parameter count {num_params} outside [28.0M, 30.0M]"


def test_weight_tying(base_config):
    model = VakModel(base_config)
    # Forward check: verify logits use the exact same tensor as tok_embeddings.weight
    x = torch.randint(0, base_config.vocab_size, (1, 8))
    logits, _ = model(x)
    assert logits.shape == (1, 8, base_config.vocab_size)

    # Gradient check: gradient on tied output flows directly to tok_embeddings.weight
    loss = logits.sum()
    loss.backward()
    assert model.tok_embeddings.weight.grad is not None
    assert not torch.isnan(model.tok_embeddings.weight.grad).any()


def test_initial_loss(base_config):
    torch.manual_seed(42)
    model = VakModel(base_config)
    batch_size = 4
    seq_len = 64
    x = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))

    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, base_config.vocab_size), targets.view(-1))

    expected_loss = math.log(base_config.vocab_size)  # ln(8192) ≈ 9.0109
    diff = abs(loss.item() - expected_loss)
    print(f"Initial loss: {loss.item():.4f}, expected: {expected_loss:.4f}, diff: {diff:.4f}")
    assert diff < 0.25, f"Initial loss {loss.item()} differs too much from ln(vocab_size)={expected_loss}"


def test_causality(base_config):
    torch.manual_seed(1337)
    model = VakModel(base_config)
    model.eval()

    seq_len = 32
    x1 = torch.randint(0, base_config.vocab_size, (1, seq_len))
    x2 = x1.clone()
    # Modify future token at position 20
    x2[0, 20] = (x2[0, 20] + 1) % base_config.vocab_size

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    # Positions before 20 must remain perfectly identical
    max_diff_past = (logits1[:, :20, :] - logits2[:, :20, :]).abs().max().item()
    assert max_diff_past == 0.0, f"Future token affected past logits! Max diff: {max_diff_past}"

    # Position 20 and after must differ
    diff_future = (logits1[:, 20:, :] - logits2[:, 20:, :]).abs().max().item()
    assert diff_future > 0.0, "Modified token did not change future logits"


def test_kv_cache_equivalence(base_config):
    torch.manual_seed(42)
    model = VakModel(base_config)
    model.eval()

    seq_len = 16
    x = torch.randint(0, base_config.vocab_size, (1, seq_len))

    # Full forward pass without KV cache
    with torch.no_grad():
        logits_full, _ = model(x)

    # Step-by-step forward pass with KV cache
    kv_caches = model.init_kv_cache(batch_size=1, device=x.device, dtype=torch.float32)
    cached_logits_list = []

    # Prefill first token
    with torch.no_grad():
        out, _ = model(x[:, :1], start_pos=0, kv_caches=kv_caches)
        cached_logits_list.append(out)

        # Autoregressive generation of remaining tokens
        for pos in range(1, seq_len):
            out, _ = model(x[:, pos : pos + 1], start_pos=pos, kv_caches=kv_caches)
            cached_logits_list.append(out)

    logits_cached = torch.cat(cached_logits_list, dim=1)

    max_diff = (logits_full - logits_cached).abs().max().item()
    print(f"KV cache max logit difference: {max_diff}")
    assert max_diff < 1e-4, f"KV cache produced divergent logits, max diff: {max_diff}"


def test_save_load_roundtrip(base_config):
    torch.manual_seed(99)
    model1 = VakModel(base_config)
    x = torch.randint(0, base_config.vocab_size, (2, 16))

    with torch.no_grad():
        logits1, _ = model1(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(model1.state_dict(), tmp_path)
        model2 = VakModel(base_config)
        model2.load_state_dict(torch.load(tmp_path, weights_only=True))

        with torch.no_grad():
            logits2, _ = model2(x)

        max_diff = (logits1 - logits2).abs().max().item()
        assert max_diff == 0.0, f"Roundtrip discrepancy: {max_diff}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_overfit_single_batch(base_config):
    # Overfit a small model on a single micro-batch to prove optimization dynamics
    torch.manual_seed(123)
    tiny_cfg = ModelConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=256,
        max_seq_len=64,
    )
    model = VakModel(tiny_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, tiny_cfg.vocab_size, (2, 32))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 32))

    initial_loss = None
    final_loss = None

    for step in range(80):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, tiny_cfg.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"Overfit test: initial={initial_loss:.4f}, final={final_loss:.4f}")
    assert final_loss < 0.1, f"Model failed to overfit tiny batch: final loss {final_loss}"


def test_determinism(base_config):
    # Verify exact loss curve reproducibility given the same seed
    def run_steps(seed: int) -> list:
        torch.manual_seed(seed)
        tiny_cfg = ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            ffn_hidden_dim=128,
            max_seq_len=32,
        )
        model = VakModel(tiny_cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            x = torch.randint(0, tiny_cfg.vocab_size, (1, 16))
            y = torch.randint(0, tiny_cfg.vocab_size, (1, 16))
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, tiny_cfg.vocab_size), y.view(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        return losses

    run1 = run_steps(42)
    run2 = run_steps(42)
    assert run1 == run2, "Different loss trajectories with the same seed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_numerical_health_gpu(base_config):
    # Verify bf16 forward and backward pass stability on GPU without NaN or Inf
    device = torch.device("cuda:0")
    model = VakModel(base_config).to(device=device, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, base_config.vocab_size, (2, 128), device=device)
    y = torch.randint(0, base_config.vocab_size, (2, 128), device=device)

    for _ in range(5):
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, base_config.vocab_size), y.view(-1))
        assert not torch.isnan(loss) and not torch.isinf(loss), f"Loss is NaN or Inf: {loss}"
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"Gradient NaN in {name}"
                assert not torch.isinf(p.grad).any(), f"Gradient Inf in {name}"

        opt.step()

