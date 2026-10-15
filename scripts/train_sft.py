import argparse
import json
import math
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import SFTDataset
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer
from src.utils import load_checkpoint, save_checkpoint, set_seed


def train_sft():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) for Vak")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--base_checkpoint", type=str, default=None, help="Base pretraining checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict["model"])
    sft_cfg = cfg_dict["sft"]
    paths_cfg = cfg_dict["paths"]

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Initializing Vak Supervised Fine-Tuning (Chat)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    tokenizer = VakTokenizer(os.path.join(paths_cfg["tokenizer_dir"], "tokenizer.json"))
    model = VakModel(model_cfg).to(device=device)

    # Load base pretrained weights
    base_ckpt = args.base_checkpoint
    if base_ckpt is None:
        base_ckpt = os.path.join(paths_cfg["checkpoint_dir"], "pretrain_latest.pt")

    if os.path.exists(base_ckpt):
        print(f"Loading base weights from {base_ckpt}...")
        load_checkpoint(base_ckpt, model, device=device)
    else:
        print(f"Warning: Base checkpoint {base_ckpt} not found. Training from initialized weights.")

    # Load conversation threads (disjoint train and validation splits)
    train_file = os.path.join(paths_cfg["processed_data_dir"], "sft_train.jsonl")
    if not os.path.exists(train_file):
        train_file = os.path.join(paths_cfg["processed_data_dir"], "sft_conversations.jsonl")

    val_file = os.path.join(paths_cfg["processed_data_dir"], "sft_val.jsonl")

    train_conversations = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_conversations.append(json.loads(line)["messages"])

    val_conversations = []
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_conversations.append(json.loads(line)["messages"])

    batch_size = args.batch_size if args.batch_size is not None else sft_cfg["micro_batch_size"]
    train_dataset = SFTDataset(train_conversations, tokenizer, seq_len=sft_cfg["seq_len"])
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = None
    if val_conversations:
        val_dataset = SFTDataset(val_conversations, tokenizer, seq_len=sft_cfg["seq_len"])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    print(f"Loaded {len(train_dataset)} SFT training sequences ({len(loader)} batches/epoch)")
    if val_loader is not None:
        print(f"Loaded {len(val_dataset)} held-out SFT validation sequences ({len(val_loader)} batches)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
    )

    model.train()
    step = 0
    total_loss = 0.0
    running_count = 0

    total_steps = len(loader) * args.epochs
    warmup_steps = max(10, int(0.05 * total_steps))

    final_ckpt = os.path.join(paths_cfg["checkpoint_dir"], "vak_chat_sft.pt")

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            # Cosine decay schedule for SFT
            if step < warmup_steps:
                lr_mult = float(step + 1) / float(max(1, warmup_steps))
            else:
                prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * lr_mult

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            device_type = "cuda" if device.type == "cuda" else "cpu"
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, _ = model(x)
                # Compute loss ONLY on assistant tokens (labels != -100)
                loss = F.cross_entropy(
                    logits.view(-1, model_cfg.vocab_size),
                    y.view(-1),
                    ignore_index=-100,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            total_loss += loss.item()
            running_count += 1
            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                avg_l = total_loss / running_count
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(loader)} | Assistant Loss: {avg_l:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)
                total_loss = 0.0
                running_count = 0

        # Evaluate on held-out SFT validation split
        val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            v_total_loss = 0.0
            v_tokens = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    with torch.autocast(device_type=device_type, dtype=dtype):
                        v_logits, _ = model(vx)
                        v_l = F.cross_entropy(
                            v_logits.view(-1, model_cfg.vocab_size),
                            vy.view(-1),
                            ignore_index=-100,
                        )
                    n_toks = (vy != -100).sum().item()
                    if n_toks > 0:
                        v_total_loss += v_l.item() * n_toks
                        v_tokens += n_toks
            val_loss = v_total_loss / max(1, v_tokens)
            val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"--> Epoch {epoch + 1} SFT Held-out Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}", flush=True)
            model.train()

        # Save checkpoint after each epoch
        save_checkpoint(
            final_ckpt,
            model,
            opt_muon=None,
            opt_adamw=optimizer,
            step=step,
            tokens_seen=0,
            data_state={"epoch": epoch + 1},
            val_loss=val_loss if val_loss is not None else (epoch_loss / len(loader)),
        )
        print(f"Epoch {epoch + 1} complete. Checkpoint updated at {final_ckpt}", flush=True)

    print(f"\nSFT training completed. Final model saved to {final_ckpt}", flush=True)


if __name__ == "__main__":
    train_sft()
