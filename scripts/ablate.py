import argparse
import copy
import json
import os
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import ShardedPretrainDataset
from src.model import ModelConfig, VakModel
from src.optim import build_optimizers, get_lr_multiplier
from src.utils import set_seed

VARIANTS = {
    "baseline": {},
    "layernorm": {"norm_type": "layernorm"},
    "gelu_mlp": {"ffn_type": "gelu"},
    "learned_pos": {"pos_emb": "learned"},
    "mha": {"n_kv_heads": 6},
    "no_qk_norm": {"use_qk_norm": False},
    "untied_embeddings": {"tie_embeddings": False},
    "adamw_only": {"_optimizer_type": "adamw_only"},
    "8_layers_wide": {
        "n_layers": 8,
        "d_model": 512,
        "n_heads": 8,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 1536,
    },
}


def run_ablation_variant(variant_name: str, steps: int = 50, batch_size: int = 8, seq_len: int = 1024):
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant_name}. Available: {list(VARIANTS.keys())}")

    variant_overrides = copy.deepcopy(VARIANTS[variant_name])
    opt_type = variant_overrides.pop("_optimizer_type", "muon_adamw")

    cfg_kwargs = {
        "vocab_size": 8192,
        "d_model": 384,
        "n_layers": 16,
        "n_heads": 6,
        "n_kv_heads": 2,
        "ffn_hidden_dim": 1024,
        "max_seq_len": 1024,
        **variant_overrides,
    }
    model_cfg = ModelConfig(**cfg_kwargs)

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VakModel(model_cfg).to(device=device)
    param_count = model.count_parameters()
    print(f"\n[{variant_name.upper()}] Instantiated: {param_count:,} parameters")

    opt_muon, opt_adamw = build_optimizers(
        model,
        muon_lr=0.02,
        adamw_lr=0.002,
        weight_decay=0.1,
        optimizer_type=opt_type,
    )

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tokenized_dir = cfg["paths"]["tokenized_dir"]

    train_ds = ShardedPretrainDataset(tokenized_dir, seq_len=seq_len, pattern="train_*.bin")
    val_ds = ShardedPretrainDataset(tokenized_dir, seq_len=seq_len, pattern="val_*.bin")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.train()
    loader_iter = iter(train_loader)

    print(f"Training for {steps} steps...")
    t0 = time.perf_counter()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for step in range(steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)

        if opt_muon is not None:
            opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, dtype=dtype):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, model_cfg.vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if opt_muon is not None:
            opt_muon.step()
        opt_adamw.step()

    elapsed = time.perf_counter() - t0

    # Bounded validation evaluation (25 batches)
    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    max_eval_batches = 25
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(val_loader):
            if i >= max_eval_batches:
                break
            x_val, y_val = x_val.to(device), y_val.to(device)
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, _ = model(x_val)
                loss = F.cross_entropy(logits.view(-1, model_cfg.vocab_size), y_val.view(-1))
            val_loss_sum += loss.item()
            val_count += 1

    avg_val_loss = val_loss_sum / max(1, val_count)
    print(f"[{variant_name}] Params: {param_count:,} | Val Loss: {avg_val_loss:.4f} | Time: {elapsed:.1f}s")
    return {
        "variant": variant_name,
        "parameters": param_count,
        "val_loss": avg_val_loss,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Vak architecture ablation experiments")
    parser.add_argument("--variant", type=str, default="all", choices=list(VARIANTS.keys()) + ["all"])
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    targets = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    results = []

    print("=" * 60)
    print(f"Running Vak Ablation Experiments: {len(targets)} variants ({args.steps} steps each)")
    print("=" * 60)

    for var in targets:
        res = run_ablation_variant(var, steps=args.steps)
        results.append(res)

    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<25} | {'Parameters':<12} | {'Val Loss':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['variant']:<25} | {r['parameters']:<12,d} | {r['val_loss']:<10.4f}")
    print("=" * 60)

    # Save to json report
    os.makedirs("reports", exist_ok=True)
    with open("reports/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
