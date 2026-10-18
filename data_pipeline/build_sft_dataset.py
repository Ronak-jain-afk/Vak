import hashlib
import json
import os
import random
import re
from typing import Dict, List, Set, Tuple

from data_pipeline.sft_seeds import (
    BACKWARD_RANGES,
    CAPITALS,
    COMPARISONS,
    COUNTING_RANGES,
    DIALOGUES,
    DIVISION_PAIRS,
    EVEN_ODD_RANGES,
    FALLBACK_REPLIES,
    GREETINGS,
    IDENTITY_TEMPLATES,
    ITEMS,
    NAMES,
    NOISE_INPUTS,
    OPPOSITES_SYNONYMS,
    POEMS,
    PYTHON_OPS,
    PYTHON_QA,
    RESUMPTIONS,
    RHYMES,
    SCIENCE_FACTS,
    SKIP_SEQS,
    TEXT_TRANSFORMS,
)

random.seed(42)

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "held_out_benchmark.json")


def load_held_out_benchmark() -> List[Dict]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(t: str) -> str:
    """Normalize text for leakage checking while preserving punctuation-only prompts."""
    t_clean = re.sub(r"\s+", " ", t).strip().lower()
    alphanumeric = re.sub(r"[^\w\s]", "", t_clean).strip()
    if alphanumeric:
        return re.sub(r"\s+", " ", alphanumeric).strip()
    return t_clean


def generate_arithmetic(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []

    # Addition
    for a in range(0, 31):
        for b in range(0, 31):
            if (a + b) % 2 == 0 or a < 12 or b < 5:
                res = a + b
                for p, ans in [
                    (f"what is {a} + {b}?", f"{a} + {b} is {res}."),
                    (f"what is {a} plus {b}?", f"{a} plus {b} is {res}."),
                    (f"calculate {a} + {b}", f"{a} + {b} is {res}."),
                    (f"whats {a}+{b}", f"{a} + {b} is {res}."),
                ]:
                    if normalize_text(p) not in held_out_set:
                        items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Subtraction
    for a in range(1, 36):
        for b in range(0, a + 1):
            if (a * b) % 2 == 0 or b < 6 or (a - b) < 5:
                res = a - b
                for p, ans in [
                    (f"what is {a} - {b}?", f"{a} - {b} is {res}."),
                    (f"what is {a} minus {b}?", f"{a} minus {b} is {res}."),
                    (f"calculate {a} - {b}", f"{a} - {b} is {res}."),
                ]:
                    if normalize_text(p) not in held_out_set:
                        items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Multiplication
    for a in range(1, 16):
        for b in range(1, 16):
            res = a * b
            for p, ans in [
                (f"what is {a} * {b}?", f"{a} * {b} is {res}."),
                (f"what is {a} times {b}?", f"{a} times {b} is {res}."),
                (f"calculate {a} * {b}", f"{a} * {b} is {res}."),
            ]:
                if normalize_text(p) not in held_out_set:
                    items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Division
    for num, den in DIVISION_PAIRS:
        res = num // den
        for p, ans in [
            (f"what is {num} / {den}?", f"{num} divided by {den} is {res}."),
            (f"what is {num} divided by {den}?", f"{num} divided by {den} is {res}."),
            (f"calculate {num} / {den}", f"{num} / {den} = {res}."),
            (f"divide {num} by {den}", f"{num} divided by {den} equals {res}."),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Word problems
    for i in range(len(NAMES)):
        n1 = NAMES[i]
        n2 = NAMES[(i + 1) % len(NAMES)]
        item = ITEMS[i % len(ITEMS)]
        for count1 in [5, 6, 8, 10, 12, 15, 20]:
            for count2 in [1, 2, 3, 4, 5]:
                if count1 > count2:
                    rem = count1 - count2
                    tot = count1 + count2
                    cases = [
                        (f"If {n1} has {count1} {item} and gives {count2} to {n2}, how many {item} does {n1} have left?",
                         f"{n1} has {rem} {item} left ({count1} - {count2} = {rem})."),
                        (f"If {n1} has {count1} {item} and {n2} gives {n1} {count2} more, how many {item} does {n1} have in total?",
                         f"{n1} has {tot} {item} in total ({count1} + {count2} = {tot})."),
                        (f"Calculate the total if {n1} has {count1} {item} and receives {count2} more.",
                         f"The total is {tot} {item} ({count1} + {count2} = {tot})."),
                        (f"Who has more {item} if {n1} has {count1} and {n2} has {count2}?",
                         f"{n1} has more {item} ({count1} is greater than {count2})."),
                    ]
                    for p, ans in cases:
                        if normalize_text(p) not in held_out_set:
                            items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_counting(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for start, end in COUNTING_RANGES:
        nums_str = ", ".join(str(n) for n in range(start, end + 1)) + "."
        for p, ans in [
            (f"count from {start} to {end}", nums_str),
            (f"count from {start} to {end} please", f"Here you go: {nums_str}"),
            (f"can you count from {start} to {end}?", f"Sure! {nums_str}"),
            (f"please count from {start} to {end}", nums_str),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for start, end in BACKWARD_RANGES:
        nums_str = ", ".join(str(n) for n in range(start, end - 1, -1)) + "."
        for p, ans in [
            (f"count backwards from {start} to {end}", nums_str),
            (f"count down from {start} to {end}", nums_str),
            (f"count backwards from {start} down to {end}", nums_str),
            (f"can you count backwards from {start} to {end}?", f"Here is the countdown: {nums_str}"),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for kind, s, e, seq in EVEN_ODD_RANGES:
        s_str = ", ".join(str(n) for n in seq) + "."
        for p, ans in [
            (f"count {kind} numbers from {s} to {e}", s_str),
            (f"what are the {kind} numbers between {s} and {e}?", f"The {kind} numbers are: {s_str}"),
            (f"list the {kind} numbers from {s} to {e}", s_str),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for p, ans in SKIP_SEQS:
        for _ in range(3):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for n in range(1, 35):
        next_n, prev_n = n + 1, n - 1
        templates = [
            (f"what comes after {n}?", f"{next_n} comes after {n}."),
            (f"what comes after {n}", f"{next_n} comes after {n}."),
            (f"what number comes after {n}?", f"The number {next_n} comes after {n}."),
        ]
        if prev_n >= 0:
            templates.append((f"what comes before {n}?", f"{prev_n} comes before {n}."))
            templates.append((f"what comes before {n}", f"{prev_n} comes before {n}."))
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_facts(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for country, cap, cont in CAPITALS:
        for p, ans in [
            (f"what is the capital of {country}?", f"The capital of {country} is {cap}."),
            (f"what is the capital of {country}", f"The capital of {country} is {cap}."),
            (f"name the capital of {country}", f"{cap} is the capital of {country}."),
            (f"which city is the capital of {country}?", f"{cap} is the capital of {country}."),
            (f"what continent is {country} in?", f"{country} is in {cont}."),
            (f"on which continent is {country} located?", f"{country} is located in {cont}."),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for p, ans in SCIENCE_FACTS:
        for var_p, var_ans in [
            (p, ans),
            (p.rstrip("?"), ans),
            (f"tell me, {p.lower()}", ans),
            (f"can you tell me {p.lower()}?", ans),
            (f"do you know {p.lower()}?", f"Yes! {ans}"),
        ]:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    return items


def generate_coding(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for p, ans in PYTHON_QA:
        for var_p, var_ans in [
            (p, ans),
            (f"can you tell me {p.lower()}", ans),
            (f"please explain {p.lower()}", ans),
            (f"in python, {p.lower()}", ans),
            (f"how to {p.lower().replace('how do you ', '').replace('how to ', '')}", ans),
        ]:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    for expr, out in PYTHON_OPS:
        for p, ans in [
            (f"what is the output of `{expr}` in Python?", f"The output of `{expr}` is `{out}`."),
            (f"what does `{expr}` evaluate to in Python?", f"`{expr}` evaluates to `{out}`."),
            (f"in Python, what does `{expr}` return?", f"It returns `{out}`."),
            (f"calculate `{expr}` in python", f"`{expr}` evaluates to `{out}`."),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_reasoning(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for word, rh_words in RHYMES:
        for p, ans in [
            (f"give me words that rhyme with '{word}'", f"Words that rhyme with '{word}' include: {rh_words}."),
            (f"what words rhyme with '{word}'?", f"Some words that rhyme with '{word}' are: {rh_words}."),
            (f"name three words that rhyme with {word}",
             f"Three words that rhyme with '{word}' are {rh_words.split(', ')[0]}, {rh_words.split(', ')[1]}, and {rh_words.split(', ')[2]}."),
            (f"can you find a rhyme for '{word}'?", f"Yes! '{rh_words.split(', ')[0]}' rhymes with '{word}'."),
        ]:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for p, ans in COMPARISONS:
        for _ in range(3):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    for p, ans in TEXT_TRANSFORMS:
        for var_p, var_ans in [
            (p, ans),
            (f"please {p.lower()}", ans),
            (f"can you {p.lower()}?", ans),
            (f"instruction: {p.lower()}", ans),
        ]:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    for p, ans in OPPOSITES_SYNONYMS:
        for var_p, var_ans in [
            (p, ans),
            (f"tell me {p.lower()}", ans),
            (f"can you tell me {p.lower()}?", ans),
            (f"please answer: {p.lower()}", ans),
        ]:
            if normalize_text(var_p) not in held_out_set:
                items.append([{"role": "user", "content": var_p}, {"role": "assistant", "content": var_ans}])

    for p, ans in POEMS:
        for _ in range(4):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_continuations(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for prefix, user_next, asst_next in RESUMPTIONS:
        for _ in range(6):
            thread = list(prefix) + [
                {"role": "user", "content": user_next},
                {"role": "assistant", "content": asst_next}
            ]
            items.append(thread)

    for d in DIALOGUES:
        for _ in range(12):
            items.append(d)

    return items


def generate_identity(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for p, ans in IDENTITY_TEMPLATES:
        for _ in range(6):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_greetings(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for p, ans in GREETINGS:
        for _ in range(5):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_fallbacks(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    items = []
    for noise in NOISE_INPUTS:
        for _ in range(5):
            reply = random.choice(FALLBACK_REPLIES)
            if normalize_text(noise) not in held_out_set:
                items.append([{"role": "user", "content": noise}, {"role": "assistant", "content": reply}])
    return items


def split_category_stratified(
    category_samples: List[List[Dict[str, str]]],
    val_ratio: float = 0.15
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    clusters: Dict[str, List[List[Dict[str, str]]]] = {}
    for sample in category_samples:
        prompt_norm = normalize_text(sample[0]["content"])
        clusters.setdefault(prompt_norm, []).append(sample)

    cluster_keys = list(clusters.keys())
    random.seed(42)
    random.shuffle(cluster_keys)

    n_val_clusters = max(1, int(len(cluster_keys) * val_ratio))
    val_keys = set(cluster_keys[:n_val_clusters])
    train_keys = set(cluster_keys[n_val_clusters:])

    train_out = [sample for k in train_keys for sample in clusters[k]]
    val_out = [sample for k in val_keys for sample in clusters[k]]

    return train_out, val_out


def main():
    held_out_benchmark = load_held_out_benchmark()
    held_out_set = {normalize_text(item["prompt"]) for item in held_out_benchmark}

    categories = {
        "arithmetic": generate_arithmetic(held_out_set),
        "counting": generate_counting(held_out_set),
        "continuations": generate_continuations(held_out_set),
        "greetings": generate_greetings(held_out_set),
        "identity": generate_identity(held_out_set),
        "facts": generate_facts(held_out_set),
        "coding": generate_coding(held_out_set),
        "reasoning": generate_reasoning(held_out_set),
        "fallbacks": generate_fallbacks(held_out_set),
    }

    train_all = []
    val_all = []

    for cat_name, cat_samples in categories.items():
        tr, va = split_category_stratified(cat_samples, val_ratio=0.15)
        train_all.extend(tr)
        val_all.extend(va)

    random.seed(42)
    random.shuffle(train_all)
    random.shuffle(val_all)

    train_prompts_norm = {normalize_text(s[0]["content"]) for s in train_all}
    val_prompts_norm = {normalize_text(s[0]["content"]) for s in val_all}

    assert len(train_prompts_norm.intersection(val_prompts_norm)) == 0, "Train and val prompts overlap"
    assert len(train_prompts_norm.intersection(held_out_set)) == 0, "Train prompts overlap with benchmark"
    assert len(val_prompts_norm.intersection(held_out_set)) == 0, "Val prompts overlap with benchmark"

    os.makedirs("data/processed", exist_ok=True)
    train_path = "data/processed/sft_train.jsonl"
    val_path = "data/processed/sft_val.jsonl"
    conv_path = "data/processed/sft_conversations.jsonl"

    for path, data in [(train_path, train_all), (val_path, val_all), (conv_path, train_all)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps({"messages": s, "is_synthetic": True}) + "\n")

    def get_file_info(p: str):
        with open(p, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest(), len(data)

    train_sha, train_size = get_file_info(train_path)
    val_sha, val_size = get_file_info(val_path)
    conv_sha, conv_size = get_file_info(conv_path)
    bench_sha, bench_size = get_file_info("data/held_out_benchmark.json")

    manifest_path = "data/manifest.json"
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        manifest["artifacts"]["sft_train.jsonl"] = {
            "path": train_path,
            "description": "Supervised fine-tuning training sequences (stratified, is_synthetic=True)",
            "sha256": train_sha,
            "size_bytes": train_size,
            "sequence_count": len(train_all),
            "unique_prompts": len(train_prompts_norm),
        }
        manifest["artifacts"]["sft_val.jsonl"] = {
            "path": val_path,
            "description": "Supervised fine-tuning held-out validation split (disjoint from train)",
            "sha256": val_sha,
            "size_bytes": val_size,
            "sequence_count": len(val_all),
            "unique_prompts": len(val_prompts_norm),
        }
        manifest["artifacts"]["sft_conversations.jsonl"] = {
            "path": conv_path,
            "description": "Supervised fine-tuning training sequences (canonical training mirror)",
            "sha256": conv_sha,
            "size_bytes": conv_size,
            "sequence_count": len(train_all),
        }
        manifest["artifacts"]["held_out_benchmark.json"] = {
            "path": "data/held_out_benchmark.json",
            "description": "Canonical 50-prompt held-out generalization benchmark with grading criteria",
            "sha256": bench_sha,
            "size_bytes": bench_size,
            "prompt_count": len(held_out_benchmark),
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print(f"Generated {len(train_all):,} train and {len(val_all):,} val sequences.")


if __name__ == "__main__":
    main()
