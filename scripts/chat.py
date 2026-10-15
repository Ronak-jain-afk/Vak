import argparse
import os
import sys
import yaml
import torch

from src.generate import stream_generate
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer
from src.utils import load_checkpoint


def chat_cli():
    parser = argparse.ArgumentParser(description="Vak interactive chat CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict["model"])
    paths_cfg = cfg_dict["paths"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer_path = os.path.join(paths_cfg["tokenizer_dir"], "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Run data preparation first.")
        sys.exit(1)

    tokenizer = VakTokenizer(tokenizer_path)
    model = VakModel(model_cfg).to(device=device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(paths_cfg["checkpoint_dir"], "vak_chat_sft.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(paths_cfg["checkpoint_dir"], "pretrain_latest.pt")

    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        load_checkpoint(ckpt_path, model, device=device)
    else:
        print(f"Notice: No checkpoint found at {ckpt_path}. Running with initialized weights.")

    model.eval()
    print("=" * 60)
    print("Vak Chat CLI (Type 'exit', 'quit', or Ctrl+C to quit; 'clear' to reset history)")
    print("=" * 60)

    messages = []
    max_prompt_len = cfg_dict.get("inference", {}).get("max_prompt_len", 768)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        # Edge case 1: Empty input -> re-prompt user without generating or crashing
        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = []
            print("Conversation history cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        # Keep at most last 3 full turns (6 messages) to prevent context pollution
        if len(messages) > 6:
            messages = messages[-6:]

        # Edge case 2: Context length checking and oldest-turn truncation
        prompt_tokens, err_msg = tokenizer.prepare_chat_input(messages, max_prompt_len=max_prompt_len)
        if err_msg:
            print(f"System Error: {err_msg}")
            messages.pop()  # Drop the invalid user message
            continue

        print("\nVak: ", end="", flush=True)
        response_pieces = []

        try:
            for piece in stream_generate(
                model,
                tokenizer,
                prompt_tokens,
                max_new_tokens=256,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            ):
                sys.stdout.write(piece)
                sys.stdout.flush()
                response_pieces.append(piece)
            print()
        except Exception as e:
            print(f"\n[Generation error: {e}]")

        full_response = "".join(response_pieces).strip()
        messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    chat_cli()
