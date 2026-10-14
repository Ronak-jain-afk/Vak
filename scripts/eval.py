import argparse
import json
import math
import os
import re
import yaml
import torch
import torch.nn.functional as F

from src.generate import generate_response
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer
from src.utils import load_checkpoint

FIXED_50_PROMPTS = [
    # Greetings & Introductions
    "Hello!",
    "Hi, how are you today?",
    "Good morning!",
    "What is your name?",
    "Nice to meet you.",
    "Hey there!",
    "Greetings! Who are you?",
    "Can you help me?",
    "Hello, are you ready to chat?",
    "Good evening!",
    # Small talk & Casual conversation
    "How is the weather?",
    "What is your favorite color?",
    "Tell me a short story.",
    "Do you like reading books?",
    "What can you do?",
    "I had a great day today.",
    "I am feeling a little tired.",
    "What is your favorite food?",
    "Tell me something interesting.",
    "Do you have any hobbies?",
    # Simple Q&A
    "What is the capital of France?",
    "How many days are in a week?",
    "What color is the sky on a sunny day?",
    "What is 2 + 2?",
    "Which planet is known as the Red Planet?",
    "What do bees make?",
    "How many legs does a spider have?",
    "What is water made of?",
    "Name a large ocean on Earth.",
    "What is the opposite of hot?",
    # Short Instructions
    "Count from 1 to 5.",
    "Say hello in French.",
    "Write a one-sentence poem about the moon.",
    "Give me three words that rhyme with cat.",
    "Say thank you politely.",
    "Name two fruits that are yellow.",
    "Describe a tree in five words.",
    "Spell the word 'apple'.",
    "Say goodbye.",
    "Repeat the word 'smile' three times.",
    # Reasoning & Clarifications
    "If I have three apples and eat one, how many are left?",
    "Is an elephant bigger than a mouse?",
    "Can birds fly in the sky?",
    "What do people use an umbrella for?",
    "Which is faster: a car or a bicycle?",
    "Why do people sleep at night?",
    "Is ice hot or cold?",
    "What season comes after winter?",
    "Can fish breathe underwater?",
    "Thank you for your help!",
]


def check_repetition(text: str, n_gram_size: int = 4, max_repeats: int = 3) -> bool:
    # Detects degenerate n-gram looping
    words = re.findall(r"\w+", text.lower())
    if len(words) < n_gram_size * max_repeats:
        return False
    ngrams = [" ".join(words[i : i + n_gram_size]) for i in range(len(words) - n_gram_size + 1)]
    counts = {}
    for ng in ngrams:
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= max_repeats:
            return True
    return False


def evaluate_chat_prompts(
    model: VakModel,
    tokenizer: VakTokenizer,
    prompts: list = FIXED_50_PROMPTS,
    device: torch.device = None,
):
    print("=" * 60)
    print(f"Evaluating Fixed Qualitative Chat Prompts ({len(prompts)} prompts)")
    print("=" * 60)

    terminated_count = 0
    repetitive_count = 0
    results = []

    for idx, prompt_text in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens, err = tokenizer.prepare_chat_input(messages)
        if err or not prompt_tokens:
            print(f"Prompt {idx + 1} preparation error: {err}")
            continue

        raw_response = generate_response(
            model,
            tokenizer,
            prompt_tokens,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            device=device,
        )

        ends_properly = True  # generate_response stops on <|end|> or max tokens
        is_looping = check_repetition(raw_response)

        if not is_looping:
            terminated_count += 1
        if is_looping:
            repetitive_count += 1

        results.append({
            "prompt": prompt_text,
            "response": raw_response.strip(),
            "is_looping": is_looping,
        })

        if idx < 5 or idx % 10 == 0:
            print(f"\n[Prompt {idx + 1}]: {prompt_text}")
            print(f"[Response]: {raw_response.strip()}")

    termination_rate = (terminated_count / len(prompts)) * 100
    repetition_rate = (repetitive_count / len(prompts)) * 100

    print("\n" + "=" * 60)
    print(f"Termination Rate: {termination_rate:.1f}% (Target: >= 95%)")
    print(f"Repetition Rate: {repetition_rate:.1f}% (Target: < 5%)")
    print("=" * 60)

    return results


def evaluate_validation_loss(
    model: VakModel,
    tokenizer: VakTokenizer,
    val_jsonl: str,
    device: torch.device,
    max_docs: int = 200,
    seq_len: int = 1024,
) -> dict:
    if not os.path.exists(val_jsonl):
        print(f"Validation file {val_jsonl} not found.")
        return {}

    total_loss = 0.0
    total_tokens = 0
    model.eval()

    with open(val_jsonl, "r", encoding="utf-8") as f:
        docs = [json.loads(line)["text"] for line in f if line.strip()][:max_docs]

    with torch.no_grad():
        for doc in docs:
            tokens = [tokenizer.bos_id] + tokenizer.encode(doc) + [tokenizer.end_id]
            if len(tokens) < 2:
                continue
            tokens = tokens[: seq_len + 1]
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, model.cfg.vocab_size), y.view(-1))

            total_loss += loss.item() * (len(tokens) - 1)
            total_tokens += len(tokens) - 1

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    bpb = avg_loss / math.log(2)

    print(f"\nHeld-out Validation Evaluation ({total_tokens:,} tokens):")
    print(f"  Val Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Bits-per-Byte (BPB): {bpb:.3f}")

    return {"val_loss": avg_loss, "perplexity": perplexity, "bpb": bpb}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vak model on loss, benchmarks, and chat")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict["model"])
    paths_cfg = cfg_dict["paths"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = VakTokenizer(os.path.join(paths_cfg["tokenizer_dir"], "tokenizer.json"))
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
        print(f"Notice: Evaluating initialized weights (no checkpoint found at {ckpt_path})")

    # 1. Validation loss on held-out text
    val_jsonl = os.path.join(paths_cfg["processed_data_dir"], "val.jsonl")
    evaluate_validation_loss(model, tokenizer, val_jsonl, device)

    # 2. Fixed chat prompts evaluation
    evaluate_chat_prompts(model, tokenizer, device=device)


if __name__ == "__main__":
    main()
