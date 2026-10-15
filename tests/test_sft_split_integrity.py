import hashlib
import json
import os
import re
import pytest

def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def test_sft_splits_exist_and_non_empty():
    train_path = "data/processed/sft_train.jsonl"
    val_path = "data/processed/sft_val.jsonl"
    bench_path = "data/held_out_benchmark.json"

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(val_path), f"Missing {val_path}"
    assert os.path.exists(bench_path), f"Missing {bench_path}"

    with open(train_path, "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]
    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    assert len(train_lines) >= 1000, f"Expected >= 1000 train lines, got {len(train_lines)}"
    assert len(val_lines) >= 100, f"Expected >= 100 val lines, got {len(val_lines)}"
    assert len(bench_data) == 50, f"Expected 50 held-out prompts, got {len(bench_data)}"


def test_is_synthetic_provenance():
    for p in ["data/processed/sft_train.jsonl", "data/processed/sft_val.jsonl"]:
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                data = json.loads(line)
                assert data.get("is_synthetic") is True, f"Missing is_synthetic=True at {p}:{line_idx}"
                assert "messages" in data and len(data["messages"]) >= 2, f"Invalid messages at {p}:{line_idx}"


def test_strict_train_val_zero_prompt_leakage():
    with open("data/processed/sft_train.jsonl", "r", encoding="utf-8") as f:
        train_prompts = {normalize_text(json.loads(l)["messages"][0]["content"]) for l in f if l.strip()}

    with open("data/processed/sft_val.jsonl", "r", encoding="utf-8") as f:
        val_prompts = {normalize_text(json.loads(l)["messages"][0]["content"]) for l in f if l.strip()}

    leakage = train_prompts.intersection(val_prompts)
    assert len(leakage) == 0, f"Found {len(leakage)} leaked prompts between SFT train and val: {leakage}"


def test_strict_held_out_benchmark_zero_leakage():
    with open("data/held_out_benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    held_out_prompts = {normalize_text(item["prompt"]) for item in benchmark}

    with open("data/processed/sft_train.jsonl", "r", encoding="utf-8") as f:
        train_prompts = {normalize_text(json.loads(l)["messages"][0]["content"]) for l in f if l.strip()}

    with open("data/processed/sft_val.jsonl", "r", encoding="utf-8") as f:
        val_prompts = {normalize_text(json.loads(l)["messages"][0]["content"]) for l in f if l.strip()}

    train_leakage = train_prompts.intersection(held_out_prompts)
    assert len(train_leakage) == 0, f"Found {len(train_leakage)} prompts in train that match held-out benchmark: {train_leakage}"

    val_leakage = val_prompts.intersection(held_out_prompts)
    assert len(val_leakage) == 0, f"Found {len(val_leakage)} prompts in val that match held-out benchmark: {val_leakage}"


def test_split_stratification_balance():
    with open("data/processed/sft_train.jsonl", "r", encoding="utf-8") as f:
        train_msgs = [json.loads(l)["messages"][0]["content"].lower() for l in f if l.strip()]

    with open("data/processed/sft_val.jsonl", "r", encoding="utf-8") as f:
        val_msgs = [json.loads(l)["messages"][0]["content"].lower() for l in f if l.strip()]

    hello_train = sum("hello" in m for m in train_msgs)
    hello_val = sum("hello" in m for m in val_msgs)
    assert hello_train > 0, "Train set has zero 'hello' prompts (severe stratification skew)!"
    assert hello_val > 0, "Val set has zero 'hello' prompts (severe stratification skew)!"

    noise_tokens = {"d", "c", "x", "z", "q", "m", "p", "b", "j", "v", "asdf", "qwerty", "zxcv", "idk", "blah", "???", "...", "!", "??", "!?"}
    noise_train = sum(normalize_text(m) in noise_tokens or m in noise_tokens for m in train_msgs)
    noise_val = sum(normalize_text(m) in noise_tokens or m in noise_tokens for m in val_msgs)
    assert noise_train > 0, "Train set has zero noise fallback prompts!"
    assert noise_val > 0, "Val set has zero noise fallback prompts!"


def test_manifest_checksums_match_disk():
    with open("data/manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for artifact_name in ["sft_train.jsonl", "sft_val.jsonl", "held_out_benchmark.json"]:
        assert artifact_name in manifest["artifacts"], f"{artifact_name} not found in manifest artifacts!"
        art_info = manifest["artifacts"][artifact_name]
        filepath = art_info["path"]
        assert os.path.exists(filepath), f"File {filepath} not found on disk"

        with open(filepath, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()
        assert actual_sha == art_info["sha256"], f"SHA256 mismatch for {artifact_name}: manifest {art_info['sha256']} != actual {actual_sha}"
