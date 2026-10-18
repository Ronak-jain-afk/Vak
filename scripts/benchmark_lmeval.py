import argparse
import json
import os
import torch
import yaml
from lm_eval import evaluator

from src.eval_harness import VakLMEvalAdapter
from src.model import ModelConfig, VakModel
from src.tokenizer import VakTokenizer
from src.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness benchmarks on Vak model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to evaluate (default: checkpoint_dir/pretrain_latest.pt)")
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy,piqa")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples per task (None for full)")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output", type=str, default="reports/lmeval_results.json")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict["model"])
    paths_cfg = cfg_dict["paths"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = VakModel(model_cfg).to(device)

    # Resolved from the config rather than a hardcoded path: a stale default would
    # silently benchmark a randomly initialized model and report believable numbers
    ckpt_path = args.checkpoint or os.path.join(paths_cfg["checkpoint_dir"], "pretrain_latest.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading weights from {ckpt_path}...")
    load_checkpoint(ckpt_path, model, device=device)

    tokenizer_path = os.path.join(paths_cfg["tokenizer_dir"], "tokenizer.json")
    tokenizer = VakTokenizer(tokenizer_path)

    adapter = VakLMEvalAdapter(model, tokenizer, device=device, max_seq_len=model_cfg.max_seq_len)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"Running lm-evaluation-harness on tasks: {tasks} (limit={args.limit}, fewshot={args.num_fewshot})...")
    results = evaluator.simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=1,
    )

    print("\n" + "=" * 60)
    print("LM-Evaluation-Harness Benchmark Results:")
    print("=" * 60)

    formatted_results = {
        "checkpoint": ckpt_path,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "tasks": {},
    }

    if "results" in results:
        for task_name, task_metrics in results["results"].items():
            formatted_results["tasks"][task_name] = {
                k: float(v) for k, v in task_metrics.items() if isinstance(v, (int, float)) and not k.endswith("_stderr")
            }
            primary_acc = task_metrics.get("acc_norm,none", task_metrics.get("acc,none", 0.0))
            print(f"  • {task_name:20s}: Accuracy = {primary_acc * 100:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=2)

    print("=" * 60)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
