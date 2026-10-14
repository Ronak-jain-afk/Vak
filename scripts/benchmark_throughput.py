import argparse
import time
import torch
import torch.nn.functional as F

from src.model import ModelConfig, VakModel
from src.optim import build_optimizers
from src.utils import get_gpu_memory_mb, get_gpu_temperature


def benchmark_batch_size(
    cfg: ModelConfig,
    batch_size: int,
    seq_len: int = 1024,
    num_warmup: int = 5,
    num_steps: int = 15,
    use_compile: bool = False,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = VakModel(cfg).to(device=device, dtype=torch.bfloat16)
    opt_muon, opt_adamw = build_optimizers(model)

    if use_compile and hasattr(torch, "compile"):
        print(f"Compiling model with torch.compile for batch_size={batch_size}...")
        model = torch.compile(model)

    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(num_warmup):
        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        opt_muon.step()
        opt_adamw.step()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_steps):
        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_muon.step()
        opt_adamw.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    total_tokens = batch_size * seq_len * num_steps
    tokens_per_sec = total_tokens / elapsed
    peak_vram_mb = get_gpu_memory_mb()
    gpu_temp = get_gpu_temperature()

    return {
        "batch_size": batch_size,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_mb": peak_vram_mb,
        "gpu_temp_c": gpu_temp,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark throughput and VRAM on Vak model")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    cfg = ModelConfig()
    print("=" * 60)
    print(f"Model: {cfg.d_model}d, {cfg.n_layers}L, {cfg.n_heads}H, {cfg.n_kv_heads}KV")
    print(f"Counted parameters: {VakModel(cfg).count_parameters():,}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    candidate_batches = [4, 8, 16, 24]
    results = []

    for bs in candidate_batches:
        print(f"\nBenchmarking micro-batch size {bs} (tokens per batch = {bs * args.seq_len:,})...")
        try:
            res = benchmark_batch_size(cfg, bs, seq_len=args.seq_len, use_compile=args.compile)
            results.append(res)
            print(
                f"  Throughput: {res['tokens_per_sec']:,.0f} tok/s | "
                f"Peak VRAM: {res['peak_vram_mb']:.1f} MB | "
                f"Temp: {res['gpu_temp_c']}°C"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at micro-batch size {bs}")
            break

    if not results:
        print("No valid batch sizes completed.")
        return

    best = max(results, key=lambda r: r["tokens_per_sec"])
    print("\n" + "=" * 60)
    print(f"BEST CONFIGURATION: Micro-batch size = {best['batch_size']}")
    print(f"Peak Throughput: {best['tokens_per_sec']:,.0f} tokens/second")
    print(f"Peak VRAM: {best['peak_vram_mb']:.1f} MB / 6,141 MB")

    target_step_tokens = 524288  # ~0.5M tokens
    micro_tokens = best["batch_size"] * args.seq_len
    grad_accum = max(1, round(target_step_tokens / micro_tokens))
    actual_step_tokens = grad_accum * micro_tokens

    print(f"Gradient accumulation steps: {grad_accum} (Actual tokens/step = {actual_step_tokens:,})")

    # Time budget analysis
    pretrain_budget_hours = 36.0
    total_seconds = pretrain_budget_hours * 3600
    est_total_tokens = best["tokens_per_sec"] * total_seconds
    target_token_budget = min(3_000_000_000, int(est_total_tokens))

    print(f"\n36-Hour Pretraining Projection:")
    print(f"  Estimated maximum trainable tokens: {est_total_tokens / 1e9:.2f}B tokens")
    print(f"  Recommended token budget: {target_token_budget / 1e9:.2f}B tokens ({target_token_budget:,} tokens)")
    print("=" * 60)


if __name__ == "__main__":
    main()
