import argparse
import json
import math
import os
import re
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import SFTDataset
from src.generate import generate_response
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer
from src.utils import load_checkpoint

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "held_out_benchmark.json")

def load_held_out_benchmark() -> list:
    """Loads the canonical single source of truth for held-out evaluation prompts."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================================
# TIER 1: IN-TEMPLATE RECALL (MEMORIZATION BENCHMARK - 50 PROMPTS)
# Prompts that overlap with training templates to evaluate recall fidelity
# =====================================================================
IN_TEMPLATE_50_PROMPTS = [
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
    """Detects degenerate n-gram looping."""
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


def grade_semantic_correctness(response: str, acceptable: list, forbidden: list) -> bool:
    """Grades whether the response is semantically / factually correct."""
    resp_lower = response.lower()
    for bad in forbidden:
        if bad.lower() in resp_lower:
            return False
    for good in acceptable:
        if good.lower() in resp_lower:
            return True
    return False


def evaluate_chat_tier1(
    model: VakModel,
    tokenizer: VakTokenizer,
    prompts: list = IN_TEMPLATE_50_PROMPTS,
    device: torch.device = None,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Tier 1: In-Template Recall (Memorization Benchmark) ({len(prompts)} prompts)")
    print("=" * 60)

    terminated_count = 0
    repetitive_count = 0
    results = []

    for idx, prompt_text in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens, err = tokenizer.prepare_chat_input(messages)
        if err or not prompt_tokens:
            continue

        raw_response, ends_properly = generate_response(
            model,
            tokenizer,
            prompt_tokens,
            max_new_tokens=256,
            temperature=0.5,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            device=device,
            return_end_flag=True,
        )

        is_looping = check_repetition(raw_response)
        if ends_properly:
            terminated_count += 1
        if is_looping:
            repetitive_count += 1

        results.append({
            "prompt": prompt_text,
            "response": raw_response.strip(),
            "ends_properly": ends_properly,
            "is_looping": is_looping,
        })

        if idx < 3 or idx % 10 == 0:
            print(f"[Prompt {idx + 1}]: {prompt_text}")
            print(f"[Response]: {raw_response.strip()}")
            print(f"[Terminated: {ends_properly} | Repetitive: {is_looping}]\n")

    term_rate = (terminated_count / len(prompts)) * 100
    rep_rate = (repetitive_count / len(prompts)) * 100

    print(f"Summary for Tier 1: In-Template Recall:")
    print(f"  • Format Termination Rate: {term_rate:.1f}% ({terminated_count}/{len(prompts)} stopped on <|end|>)")
    print(f"  • Repetition Loop Rate:    {rep_rate:.1f}% ({repetitive_count}/{len(prompts)} looping)")
    print("=" * 60)

    return {
        "tier": "Tier 1: In-Template Recall",
        "termination_rate": term_rate,
        "repetition_rate": rep_rate,
        "terminated_count": terminated_count,
        "repetitive_count": repetitive_count,
        "total_prompts": len(prompts),
        "prompts": results,
    }


def evaluate_chat_tier2_held_out(
    model: VakModel,
    tokenizer: VakTokenizer,
    benchmark_items: list,
    device: torch.device = None,
) -> dict:
    print("\n" + "=" * 60)
    print(f"Tier 2: Held-Out Generalization (Disjoint Benchmark) ({len(benchmark_items)} prompts)")
    print("=" * 60)

    terminated_count = 0
    repetitive_count = 0
    correct_count = 0
    results = []

    for idx, item in enumerate(benchmark_items):
        prompt_text = item["prompt"]
        messages = [{"role": "user", "content": prompt_text}]
        prompt_tokens, err = tokenizer.prepare_chat_input(messages)
        if err or not prompt_tokens:
            continue

        raw_response, ends_properly = generate_response(
            model,
            tokenizer,
            prompt_tokens,
            max_new_tokens=256,
            temperature=0.5,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.15,
            device=device,
            return_end_flag=True,
        )

        is_looping = check_repetition(raw_response)
        is_correct = grade_semantic_correctness(
            raw_response,
            acceptable=item.get("acceptable_keywords", []),
            forbidden=item.get("forbidden_keywords", [])
        )

        if ends_properly:
            terminated_count += 1
        if is_looping:
            repetitive_count += 1
        if is_correct:
            correct_count += 1

        results.append({
            "prompt": prompt_text,
            "category": item.get("category"),
            "response": raw_response.strip(),
            "ends_properly": ends_properly,
            "is_looping": is_looping,
            "semantically_correct": is_correct,
        })

        if idx < 4 or idx % 10 == 0:
            print(f"[Prompt {idx + 1} ({item.get('category')})]: {prompt_text}")
            print(f"[Response]: {raw_response.strip()}")
            print(f"[Terminated: {ends_properly} | Semantically Correct: {is_correct}]\n")

    term_rate = (terminated_count / len(benchmark_items)) * 100
    rep_rate = (repetitive_count / len(benchmark_items)) * 100
    accuracy_rate = (correct_count / len(benchmark_items)) * 100

    print(f"Summary for Tier 2: Held-Out Generalization:")
    print(f"  • Format Termination Rate:   {term_rate:.1f}% ({terminated_count}/{len(benchmark_items)} stopped on <|end|>)")
    print(f"  • Repetition Loop Rate:      {rep_rate:.1f}% ({repetitive_count}/{len(benchmark_items)} looping)")
    print(f"  • Semantic Accuracy Rate:    {accuracy_rate:.1f}% ({correct_count}/{len(benchmark_items)} semantically correct)")
    print("=" * 60)

    return {
        "tier": "Tier 2: Held-Out Generalization",
        "termination_rate": term_rate,
        "repetition_rate": rep_rate,
        "semantic_accuracy_rate": accuracy_rate,
        "terminated_count": terminated_count,
        "repetitive_count": repetitive_count,
        "correct_count": correct_count,
        "total_prompts": len(benchmark_items),
        "prompts": results,
    }


def evaluate_sft_val_loss(
    model: VakModel,
    tokenizer: VakTokenizer,
    sft_val_path: str,
    device: torch.device,
    seq_len: int = 1024,
) -> dict:
    if not os.path.exists(sft_val_path):
        return {}

    conversations = []
    with open(sft_val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line)["messages"])

    if not conversations:
        return {}

    dataset = SFTDataset(conversations, tokenizer, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, model.cfg.vocab_size), y.view(-1), ignore_index=-100)

            n_toks = (y != -100).sum().item()
            if n_toks > 0:
                total_loss += loss.item() * n_toks
                total_tokens += n_toks

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    print(f"\nHeld-out SFT Validation Split Evaluation ({len(dataset)} sequences, {total_tokens:,} assistant tokens):")
    print(f"  • SFT Val Assistant Loss: {avg_loss:.4f} nats")
    print(f"  • SFT Val Perplexity:     {perplexity:.2f}")

    return {
        "sft_val_loss": avg_loss,
        "sft_val_perplexity": perplexity,
        "assistant_tokens": total_tokens,
        "sequences": len(dataset),
    }


def evaluate_pretrain_validation_loss(
    model: VakModel,
    tokenizer: VakTokenizer,
    val_jsonl: str,
    device: torch.device,
    max_docs: int = 200,
    seq_len: int = 1024,
) -> dict:
    if not os.path.exists(val_jsonl):
        return {}

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    model.eval()

    with open(val_jsonl, "r", encoding="utf-8") as f:
        docs = [json.loads(line)["text"] for line in f if line.strip()][:max_docs]

    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.no_grad():
        for doc in docs:
            tokens = [tokenizer.bos_id] + tokenizer.encode(doc) + [tokenizer.end_id]
            if len(tokens) < 2:
                continue
            tokens = tokens[: seq_len + 1]
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

            with torch.autocast(device_type=device_type, dtype=dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, model.cfg.vocab_size), y.view(-1))

            num_toks = len(tokens) - 1
            total_loss += loss.item() * num_toks
            total_tokens += num_toks
            total_bytes += len(doc.encode("utf-8"))

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    bpt = avg_loss / math.log(2)
    bytes_per_token = total_bytes / max(1, total_tokens)
    bpb = bpt / max(1e-5, bytes_per_token)

    print(f"\nHeld-out Pretraining Validation Evaluation ({total_tokens:,} tokens, {total_bytes:,} UTF-8 bytes):")
    print(f"  • Val Loss:               {avg_loss:.4f} nats")
    print(f"  • Perplexity:             {perplexity:.2f}")
    print(f"  • Bits-per-Token (BPT):   {bpt:.3f}")
    print(f"  • Bits-per-Byte (BPB):    {bpb:.3f} (at {bytes_per_token:.2f} bytes/token)")

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "bpt": bpt,
        "bpb": bpb,
        "bytes_per_token": bytes_per_token,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
    }


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

    # 1. Validation loss on held-out pretraining text
    val_jsonl = os.path.join(paths_cfg["processed_data_dir"], "val.jsonl")
    pretrain_val_metrics = evaluate_pretrain_validation_loss(model, tokenizer, val_jsonl, device)

    # 2. Validation loss on held-out SFT split
    sft_val_jsonl = os.path.join(paths_cfg["processed_data_dir"], "sft_val.jsonl")
    sft_val_metrics = evaluate_sft_val_loss(model, tokenizer, sft_val_jsonl, device)

    # 3. Tier 1: In-Template Recall (50 prompts)
    tier1_metrics = evaluate_chat_tier1(model, tokenizer, IN_TEMPLATE_50_PROMPTS, device=device)

    # 4. Tier 2: Held-Out Generalization with Semantic Accuracy Grading (50 prompts)
    held_out_benchmark = load_held_out_benchmark()
    tier2_metrics = evaluate_chat_tier2_held_out(model, tokenizer, held_out_benchmark, device=device)

    # Save structured results to reports/eval_results.json
    os.makedirs("reports", exist_ok=True)
    report_data = {
        "checkpoint": ckpt_path,
        "pretrain_validation": pretrain_val_metrics,
        "sft_validation": sft_val_metrics,
        "in_template_recall": {
            "termination_rate": tier1_metrics.get("termination_rate"),
            "repetition_rate": tier1_metrics.get("repetition_rate"),
            "terminated_count": tier1_metrics.get("terminated_count"),
            "total_prompts": tier1_metrics.get("total_prompts"),
            "samples": tier1_metrics.get("prompts", [])[:10],
        },
        "held_out_generalization": {
            "termination_rate": tier2_metrics.get("termination_rate"),
            "repetition_rate": tier2_metrics.get("repetition_rate"),
            "semantic_accuracy_rate": tier2_metrics.get("semantic_accuracy_rate"),
            "terminated_count": tier2_metrics.get("terminated_count"),
            "correct_count": tier2_metrics.get("correct_count"),
            "total_prompts": tier2_metrics.get("total_prompts"),
            "samples": tier2_metrics.get("prompts", [])[:12],
        },
    }
    with open("reports/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nEvaluation results saved cleanly to reports/eval_results.json")


if __name__ == "__main__":
    main()
