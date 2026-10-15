import argparse
import math
import os
import time
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import ResumableRandomSampler, ShardedPretrainDataset
from src.model import ModelConfig, VakModel
from src.optim import build_optimizers, get_lr_multiplier, resume_rewarm_multiplier
from src.utils import get_gpu_memory_mb, get_gpu_temperature, load_checkpoint, save_checkpoint, set_seed


@torch.no_grad()
def evaluate_val(model, val_loader, device, vocab_size, max_batches=20):
    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    for i, (x_val, y_val) in enumerate(val_loader):
        if i >= max_batches:
            break
        x_val = x_val.to(device, non_blocking=True)
        y_val = y_val.to(device, non_blocking=True)
        device_type = "cuda" if device.type == "cuda" else "cpu"
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        with torch.autocast(device_type=device_type, dtype=dtype):
            logits, _ = model(x_val)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y_val.view(-1))
        val_loss_sum += loss.item()
        val_count += 1
    model.train()
    return val_loss_sum / max(1, val_count)


def train():
    parser = argparse.ArgumentParser(description="Pretrain Vak language model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum optimizer steps")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict["model"])
    train_cfg = cfg_dict["training"]
    paths_cfg = cfg_dict["paths"]

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Initializing Vak Pretraining Engine")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Vocab size: {model_cfg.vocab_size} | Layers: {model_cfg.n_layers} | d_model: {model_cfg.d_model}")
    print("=" * 60)

    model = VakModel(model_cfg).to(device=device)
    param_count = model.count_parameters()
    print(f"Counted Model Parameters: {param_count:,}")

    opt_muon, opt_adamw = build_optimizers(
        model,
        muon_lr=train_cfg["learning_rate_muon"],
        adamw_lr=train_cfg["learning_rate_adamw"],
        weight_decay=train_cfg["weight_decay"],
    )

    micro_batch_size = train_cfg["micro_batch_size"]
    seq_len = train_cfg["seq_len"]
    tokens_per_micro = micro_batch_size * seq_len
    grad_accum_steps = (
        args.grad_accum_steps
        if args.grad_accum_steps is not None
        else train_cfg.get("grad_accum_steps", max(1, train_cfg["target_batch_tokens"] // tokens_per_micro))
    )
    step_tokens = grad_accum_steps * tokens_per_micro
    samples_per_step = grad_accum_steps * micro_batch_size
    print(f"Tokens per micro-batch: {tokens_per_micro:,} | Accumulation steps: {grad_accum_steps} | Step tokens: {step_tokens:,}")

    tokenized_dir = paths_cfg["tokenized_dir"]
    train_dataset = ShardedPretrainDataset(tokenized_dir, seq_len=train_cfg["seq_len"], pattern="train_*.bin")
    val_dataset = ShardedPretrainDataset(tokenized_dir, seq_len=train_cfg["seq_len"], pattern="val_*.bin")
    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=1,
    )

    batches_per_epoch = len(train_dataset) // micro_batch_size

    start_step = 0
    tokens_seen = 0
    current_epoch = 0
    micro_batches_in_epoch = 0
    start_sample_offset = 0
    data_seed = 42
    last_val_loss = None
    latest_ckpt_path = os.path.join(paths_cfg["checkpoint_dir"], "pretrain_latest.pt")

    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming checkpoint from {latest_ckpt_path}...")
        ckpt_state = load_checkpoint(latest_ckpt_path, model, opt_muon, opt_adamw, device=device)
        start_step = ckpt_state.get("step", 0)
        tokens_seen = ckpt_state.get("tokens_seen", 0)
        data_state = ckpt_state.get("data_state", {})

        if "micro_batches_in_epoch" in data_state:
            current_epoch = data_state.get("epoch", 0)
            micro_batches_in_epoch = data_state.get("micro_batches_in_epoch", 0)
            if data_state.get("dataset_len") not in (None, len(train_dataset)):
                print(
                    f"Corpus changed since checkpoint "
                    f"({data_state.get('dataset_len')} -> {len(train_dataset)} samples). "
                    f"Restarting data position at epoch 0 (model and optimizer weights kept)."
                )
                current_epoch = 0
                micro_batches_in_epoch = 0
        else:
            total_micro_batches = start_step * grad_accum_steps
            current_epoch = total_micro_batches // batches_per_epoch
            micro_batches_in_epoch = total_micro_batches % batches_per_epoch

        start_sample_offset = micro_batches_in_epoch * micro_batch_size
        data_seed = data_state.get("seed", 42)
        last_val_loss = ckpt_state.get("val_loss", None)
        print(
            f"Resumed at step {start_step:,} ({tokens_seen:,} tokens seen, "
            f"epoch {current_epoch}, micro-batch {micro_batches_in_epoch:,}/{batches_per_epoch:,}, "
            f"sample offset {start_sample_offset:,}/{len(train_dataset):,})"
        )

    train_sampler = ResumableRandomSampler(
        train_dataset,
        seed=data_seed,
        start_index=start_sample_offset,
        epoch=current_epoch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    if train_cfg.get("compile", True) and not args.no_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        compiled_model = torch.compile(model)
    else:
        compiled_model = model

    total_steps = args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 954)
    warmup_steps = min(train_cfg.get("warmup_steps", 50), max(5, int(total_steps * 0.1)))
    decay_schedule = train_cfg.get("decay_schedule", "cosine")
    min_lr_ratio = train_cfg.get("min_lr_ratio", 0.1)
    resume_rewarm_steps = train_cfg.get("resume_rewarm_steps", 100) if start_step > 0 else 0

    compiled_model.train()
    opt_muon.zero_grad(set_to_none=True)
    opt_adamw.zero_grad(set_to_none=True)

    accum_loss = 0.0
    accum_count = 0
    step = start_step
    last_save_time = time.time()
    t_start = time.perf_counter()
    tokens_in_window = 0

    loader_iter = iter(train_loader)

    print(f"\nStarting pretraining loop for {total_steps - start_step} steps...")
    while step < total_steps:
        # Dynamic LR adjustment according to schedule
        lr_mult = get_lr_multiplier(step, total_steps, warmup_steps, min_lr_ratio, schedule=decay_schedule)
        # Re-warmup after resume: extending total_steps under cosine would otherwise
        # jump LR ~6x (0.10x -> ~0.60x at step 946/1908) and spike loss on converged weights
        lr_mult = resume_rewarm_multiplier(lr_mult, step, start_step, resume_rewarm_steps, min_lr_ratio)
        for group in opt_muon.param_groups:
            group["lr"] = train_cfg["learning_rate_muon"] * lr_mult
        for group in opt_adamw.param_groups:
            group["lr"] = train_cfg["learning_rate_adamw"] * lr_mult

        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(loader_iter)
                micro_batches_in_epoch += 1
            except StopIteration:
                current_epoch += 1
                micro_batches_in_epoch = 0
                train_sampler = ResumableRandomSampler(
                    train_dataset,
                    seed=data_seed,
                    start_index=0,
                    epoch=current_epoch,
                )
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=micro_batch_size,
                    sampler=train_sampler,
                    drop_last=True,
                    num_workers=2,
                    pin_memory=True,
                )
                loader_iter = iter(train_loader)
                x, y = next(loader_iter)
                micro_batches_in_epoch += 1

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            autocast_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype):
                logits, _ = compiled_model(x)
                loss = F.cross_entropy(logits.view(-1, model_cfg.vocab_size), y.view(-1))
                loss = loss / grad_accum_steps

            loss.backward()
            accum_loss += loss.item() * grad_accum_steps
            accum_count += 1
            tokens_in_window += tokens_per_micro
            tokens_seen += tokens_per_micro

        # Gradient clipping & optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
        opt_muon.step()
        opt_adamw.step()
        opt_muon.zero_grad(set_to_none=True)
        opt_adamw.zero_grad(set_to_none=True)

        step += 1
        avg_loss = accum_loss / max(1, accum_count)
        accum_loss = 0.0
        accum_count = 0

        # Periodic logging and telemetry
        if step % 10 == 0 or step == total_steps:
            elapsed = time.perf_counter() - t_start
            tok_per_sec = tokens_in_window / max(1e-5, elapsed)
            tokens_in_window = 0
            t_start = time.perf_counter()
            gpu_temp = get_gpu_temperature()
            vram_mb = get_gpu_memory_mb()
            bpt = avg_loss / math.log(2)

            print(
                f"Step {step:05d}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"BPT: {bpt:.3f} | "
                f"Throughput: {tok_per_sec:,.0f} tok/s | "
                f"LR mult: {lr_mult:.3f} | "
                f"VRAM: {vram_mb:.0f} MB | "
                f"Temp: {gpu_temp}°C"
            )

        # Periodic validation evaluation
        if step % train_cfg.get("eval_interval_steps", 25) == 0 or step == total_steps:
            val_loss = evaluate_val(model, val_loader, device, vocab_size=model_cfg.vocab_size, max_batches=20)
            last_val_loss = val_loss
            val_bpt = val_loss / math.log(2)
            print(f"--> Step {step:05d} Eval | Val Loss: {val_loss:.4f} | Val BPT: {val_bpt:.3f}")

        # Checkpointing (every interval or every 30 minutes)
        time_since_save = (time.time() - last_save_time) / 60.0
        if step % train_cfg.get("checkpoint_interval_steps", 500) == 0 or time_since_save >= train_cfg.get("save_checkpoint_mins", 30) or step == total_steps:
            step_ckpt_path = os.path.join(paths_cfg["checkpoint_dir"], f"pretrain_step_{step:06d}.pt")
            if last_val_loss is None:
                last_val_loss = evaluate_val(model, val_loader, device, vocab_size=model_cfg.vocab_size, max_batches=20)
            saved_val = last_val_loss

            current_sample_offset = micro_batches_in_epoch * micro_batch_size
            total_samples_consumed = (current_epoch * batches_per_epoch + micro_batches_in_epoch) * micro_batch_size
            data_state = {
                "step": step,
                "total_samples_consumed": total_samples_consumed,
                "epoch": current_epoch,
                "micro_batches_in_epoch": micro_batches_in_epoch,
                "sample_offset": current_sample_offset,
                "dataset_len": len(train_dataset),
                "seed": data_seed,
            }

            save_checkpoint(
                step_ckpt_path,
                model,
                opt_muon,
                opt_adamw,
                step=step,
                tokens_seen=tokens_seen,
                data_state=data_state,
                val_loss=saved_val,
            )
            # Update latest checkpoint symlink/copy
            save_checkpoint(
                latest_ckpt_path,
                model,
                opt_muon,
                opt_adamw,
                step=step,
                tokens_seen=tokens_seen,
                data_state=data_state,
                val_loss=saved_val,
            )
            last_save_time = time.time()
            print(f"Checkpoint saved at step {step} to {step_ckpt_path}")

    print("\nPretraining completed successfully.")


if __name__ == "__main__":
    train()
