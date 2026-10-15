import hashlib
import json
import os
import random
import re
from typing import Dict, List, Set, Tuple

# Set random seed for reproducibility
random.seed(42)

BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "held_out_benchmark.json")

def load_held_out_benchmark() -> List[Dict]:
    """Loads the canonical single source of truth for held-out evaluation prompts."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(t: str) -> str:
    """Normalize text for strict leakage checking.
    
    Strips peripheral punctuation for alphanumeric phrases, but preserves punctuation-only
    prompts (e.g. '...', '???', '!?') so distinct noise prompts are not collapsed to empty string.
    """
    t_clean = re.sub(r"\s+", " ", t).strip().lower()
    alphanumeric = re.sub(r"[^\w\s]", "", t_clean).strip()
    if alphanumeric:
        return re.sub(r"\s+", " ", alphanumeric).strip()
    return t_clean


def generate_arithmetic(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates varied arithmetic Q&A across spacing, contractions, and operations."""
    items = []
    
    # Addition up to 20
    for a in range(0, 16):
        for b in range(0, 16):
            res = a + b
            templates = [
                (f"what is {a} + {b}?", f"{a} + {b} is {res}."),
                (f"what is {a} + {b}", f"{a} + {b} is {res}."),
                (f"whats {a}+{b}", f"{a} + {b} is {res}."),
                (f"what is {a}+{b}", f"{a} + {b} is {res}."),
                (f"what is {a} plus {b}?", f"{a} plus {b} is {res}."),
                (f"can you add {a} and {b}?", f"{a} + {b} is {res}."),
                (f"calculate {a} + {b}", f"{a} + {b} is {res}."),
            ]
            for p, ans in templates:
                if normalize_text(p) not in held_out_set:
                    items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    
    # Subtraction up to 20
    for a in range(1, 21):
        for b in range(0, a + 1):
            res = a - b
            templates = [
                (f"what is {a} - {b}?", f"{a} - {b} is {res}."),
                (f"what is {a} - {b}", f"{a} - {b} is {res}."),
                (f"whats {a}-{b}", f"{a} - {b} is {res}."),
                (f"what is {a}-{b}", f"{a} - {b} is {res}."),
                (f"what is {a} minus {b}?", f"{a} minus {b} is {res}."),
            ]
            for p, ans in templates:
                if normalize_text(p) not in held_out_set:
                    items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    
    # Multiplication small tables (up to 5x5)
    for a in range(1, 6):
        for b in range(1, 6):
            res = a * b
            templates = [
                (f"what is {a} * {b}?", f"{a} * {b} is {res}."),
                (f"what is {a} times {b}?", f"{a} times {b} is {res}."),
            ]
            for p, ans in templates:
                if normalize_text(p) not in held_out_set:
                    items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_counting(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates counting in various forward and backward ranges."""
    items = []
    
    ranges = [
        (1, 5), (1, 10), (1, 15), (1, 20),
        (5, 10), (5, 15), (10, 15), (11, 20), (15, 20),
        (2, 6), (3, 8), (4, 9), (6, 11), (8, 14), (20, 25), (10, 20)
    ]
    
    for start, end in ranges:
        nums_str = ", ".join(str(n) for n in range(start, end + 1)) + "."
        templates = [
            (f"count from {start} to {end}", nums_str),
            (f"count from {start} to {end} please", f"Here you go: {nums_str}"),
            (f"can you count from {start} to {end}?", f"Sure! {nums_str}"),
            (f"please count from {start} to {end}", nums_str),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Backwards counting
    back_ranges = [(5, 1), (10, 1), (10, 5), (6, 2), (8, 3), (7, 1), (9, 4)]
    for start, end in back_ranges:
        nums_str = ", ".join(str(n) for n in range(start, end - 1, -1)) + "."
        templates = [
            (f"count backwards from {start} to {end}", nums_str),
            (f"count down from {start} to {end}", nums_str),
            (f"count backwards from {start} down to {end}", nums_str),
        ]
        for p, ans in templates:
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Next / Previous number sequences
    for n in range(1, 20):
        next_n = n + 1
        prev_n = n - 1
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


def generate_continuations(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Multi-turn threads including conversational ellipsis and counting continuations."""
    items = []

    resumptions = [
        (
            [{"role": "user", "content": "count from 1 to 10"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."}],
            "continue to 20",
            "11, 12, 13, 14, 15, 16, 17, 18, 19, 20."
        ),
        (
            [{"role": "user", "content": "count from 1 to 10"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5, 6, 7, 8, 9, 10."}],
            "keep going to 20",
            "11, 12, 13, 14, 15, 16, 17, 18, 19, 20."
        ),
        (
            [{"role": "user", "content": "count from 1 to 5"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5."}],
            "continue to 10",
            "6, 7, 8, 9, 10."
        ),
        (
            [{"role": "user", "content": "count from 1 to 5"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5."}],
            "keep going",
            "6, 7, 8, 9, 10."
        ),
        (
            [{"role": "user", "content": "count from 11 to 15"},
             {"role": "assistant", "content": "11, 12, 13, 14, 15."}],
            "continue to 20",
            "16, 17, 18, 19, 20."
        ),
        (
            [{"role": "user", "content": "count from 5 to 10"},
             {"role": "assistant", "content": "5, 6, 7, 8, 9, 10."}],
            "continue to 15",
            "11, 12, 13, 14, 15."
        ),
        (
            [{"role": "user", "content": "count from 1 to 5"},
             {"role": "assistant", "content": "1, 2, 3, 4, 5."}],
            "now backwards",
            "5, 4, 3, 2, 1."
        ),
    ]

    for prefix, user_next, asst_next in resumptions:
        for _ in range(4):
            thread = list(prefix) + [
                {"role": "user", "content": user_next},
                {"role": "assistant", "content": asst_next}
            ]
            items.append(thread)

    dialogues = [
        [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "My name is Vak. I am a compact conversational model."},
            {"role": "user", "content": "Nice to meet you Vak!"},
            {"role": "assistant", "content": "Nice to meet you too! What would you like to explore today?"},
        ],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How are you doing today?"},
            {"role": "user", "content": "I am doing well, thanks. What is 5 + 5?"},
            {"role": "assistant", "content": "5 + 5 is 10."},
            {"role": "user", "content": "And what is 10 + 10?"},
            {"role": "assistant", "content": "10 + 10 is 20."},
        ],
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And what is the capital of Italy?"},
            {"role": "assistant", "content": "The capital of Italy is Rome."},
            {"role": "user", "content": "And what is the capital of Japan?"},
            {"role": "assistant", "content": "The capital of Japan is Tokyo."},
        ],
        [
            {"role": "user", "content": "Can you count to 5?"},
            {"role": "assistant", "content": "Sure! 1, 2, 3, 4, 5."},
            {"role": "user", "content": "Thank you!"},
            {"role": "assistant", "content": "You are very welcome!"},
        ],
        [
            {"role": "user", "content": "Tell me a short story."},
            {"role": "assistant", "content": "Once upon a time, a curious bird found a shiny golden key beneath a tall oak tree. When it unlocked the garden gate, all the flowers bloomed with vibrant colors."},
            {"role": "user", "content": "That was very peaceful."},
            {"role": "assistant", "content": "I am glad you enjoyed it! Stories can bring a moment of calm."},
        ],
    ]

    for d in dialogues:
        for _ in range(4):
            items.append(d)

    return items


def generate_identity(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates identity and capability queries."""
    items = []
    identity_templates = [
        ("what are you?", "I am Vak, a compact conversational language model."),
        ("what are you", "I am Vak, a compact conversational language model."),
        ("who are you?", "I am Vak, an open-source compact language model trained from scratch."),
        ("who are you", "I am Vak, an open-source compact language model."),
        ("what is your name?", "My name is Vak."),
        ("what is your name", "My name is Vak."),
        ("whats your name", "My name is Vak."),
        ("what's your name?", "My name is Vak."),
        ("who created you?", "I am Vak, designed and trained as an open-source compact language model."),
        ("who made you?", "I was trained as Vak, a parameter-efficient language model."),
        ("are you a human?", "No, I am Vak, an artificial intelligence language model."),
        ("what can you do?", "I can answer questions, count numbers, do simple math, chat, and help with conversational tasks."),
        ("what can you do", "I can answer questions, count numbers, do simple math, and engage in simple conversations."),
    ]
    for p, ans in identity_templates:
        for _ in range(3):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_greetings(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates varied greetings and conversational pleasantries."""
    items = []
    greetings = [
        ("hi", "Hello! How can I help you today?"),
        ("hi!", "Hi there! How can I help you today?"),
        ("hi there", "Hi there! How are you doing?"),
        ("hi assistant", "Hello! How can I help you today?"),
        ("hello", "Hello! How can I assist you today?"),
        ("hello!", "Hello! It is great to chat with you. How can I help?"),
        ("hello there", "Hello there! How can I help you?"),
        ("hello friend", "Hello! Wonderful to chat with you. How can I assist?"),
        ("hello assistant", "Hello! How can I assist you today?"),
        ("hello vak", "Hello! How can I help you today?"),
        ("hello to you", "Hello! Wonderful to meet you. What can I do for you?"),
        ("say hello", "Hello! It is great to chat with you."),
        ("hey", "Hey! What can I do for you today?"),
        ("hey there", "Hey there! How is your day going?"),
        ("hey friend", "Hey! How can I assist you today?"),
        ("good morning", "Good morning! How can I help you today?"),
        ("good morning!", "Good morning! I hope you have a great day ahead. How can I assist?"),
        ("good morning to you", "Good morning! I hope you have a wonderful day."),
        ("good evening", "Good evening! How can I help you tonight?"),
        ("good evening!", "Good evening! How can I assist you?"),
        ("how are you?", "I am doing well, thank you! How are you doing?"),
        ("how are you today?", "I'm doing great! How has your day been?"),
        ("how are you doing?", "I am doing well, thanks! How can I help?"),
        ("nice to meet you", "Nice to meet you too! How can I help you today?"),
        ("nice to meet you.", "Nice to meet you too! How can I help you today?"),
        ("thank you", "You're welcome! Let me know if you need anything else."),
        ("thanks!", "Happy to help! Let me know if you have more questions."),
        ("goodbye", "Goodbye! Have a wonderful day!"),
        ("bye!", "Bye! Take care!"),
        ("greetings", "Greetings! How may I help you today?"),
        ("greetings!", "Greetings! What can I assist you with today?"),
    ]
    for p, ans in greetings:
        for _ in range(3):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])
    return items


def generate_facts(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates general facts, science, geography, and commonsense Q&A."""
    items = []

    facts = [
        ("what is the capital of France?", "The capital of France is Paris."),
        ("what is the capital of France", "The capital of France is Paris, France."),
        ("what is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("what is the capital of Germany?", "The capital of Germany is Berlin."),
        ("what is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
        ("what is the capital of the United States?", "The capital of the United States is Washington, D.C."),
        ("what is the capital of Italy?", "The capital of Italy is Rome."),
        ("what is the capital of Spain?", "The capital of Spain is Madrid."),
        ("what is the capital of Canada?", "The capital of Canada is Ottawa."),
        ("what is the capital of Australia?", "The capital of Australia is Canberra."),
        ("what color is the sky on a sunny day?", "The sky is blue on a sunny day."),
        ("what color is grass?", "Grass is green."),
        ("what color is a ripe banana?", "A ripe banana is yellow."),
        ("what color is snow?", "Snow is white."),
        ("what color is a lemon?", "A lemon is yellow."),
        ("what color are strawberries?", "Strawberries are red."),
        ("what do bees make?", "Bees make honey."),
        ("how many days are in a week?", "There are 7 days in a week."),
        ("how many months are in a year?", "There are 12 months in a year."),
        ("how many hours are in a day?", "There are 24 hours in a day."),
        ("how many minutes are in an hour?", "There are 60 minutes in an hour."),
        ("how many legs does a spider have?", "A spider has eight legs."),
        ("how many legs does a dog have?", "A dog has four legs."),
        ("how many legs does a cat have?", "A cat has four legs."),
        ("how many legs does a bird have?", "A bird has two legs."),
        ("which is bigger, an elephant or a mouse?", "An elephant is much bigger than a mouse."),
        ("which is faster: a car or a bicycle?", "A car is faster than a bicycle."),
        ("can birds fly in the sky?", "Yes, most birds can fly in the sky."),
        ("can fish breathe underwater?", "Yes, fish breathe underwater using gills."),
        ("what do people use an umbrella for?", "People use an umbrella to stay dry in the rain or to shield from the sun."),
        ("why do people sleep at night?", "People sleep to rest, restore their energy, and stay healthy."),
        ("is ice hot or cold?", "Ice is cold."),
        ("is fire hot or cold?", "Fire is hot."),
        ("what season comes after winter?", "Spring comes after winter."),
        ("what season comes after summer?", "Autumn (or fall) comes after summer."),
        ("what planet do we live on?", "We live on Earth."),
        ("which planet is closest to the Sun?", "Mercury is closest to the Sun."),
        ("what is the opposite of hot?", "The opposite of hot is cold."),
        ("what is the opposite of up?", "The opposite of up is down."),
        ("what is the opposite of fast?", "The opposite of fast is slow."),
        ("what is the opposite of day?", "The opposite of day is night."),
        ("what is water made of?", "Water is made of hydrogen and oxygen (H2O)."),
        ("what shines brightly in the sky during the day?", "The Sun shines brightly during the day."),
    ]

    for p, ans in facts:
        for _ in range(2):
            if normalize_text(p) not in held_out_set:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_fallbacks(held_out_set: Set[str]) -> List[List[Dict[str, str]]]:
    """Generates polite clarification responses for typos, single characters, and noise."""
    items = []
    
    noise_inputs = [
        "d", "c", "x", "z", "q", "m", "p", "b", "j", "v",
        "D", "C", "X", "Z", "Q",
        "asdf", "qwerty", "zxcv", "idk", "blah",
        "???", "...", "!", "??", "!?",
        "test", "testing 123", "hello???",
    ]

    replies = [
        "I didn't quite catch that. Feel free to ask a question or tell me what you need!",
        "I'm not sure what you mean by that. Could you please clarify or ask a question?",
        "Could you please elaborate? How can I help you?",
        "What does that refer to? I'm happy to help if you have a specific question.",
    ]

    for noise in noise_inputs:
        for _ in range(2):
            reply = random.choice(replies)
            if normalize_text(noise) not in held_out_set:
                items.append([{"role": "user", "content": noise}, {"role": "assistant", "content": reply}])

    return items


def split_category_stratified(
    category_samples: List[List[Dict[str, str]]],
    val_ratio: float = 0.10
) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """
    Groups samples by normalized prompt cluster and splits clusters between Train and Val.
    Guarantees that:
    1) Every category is represented in both Train and Val.
    2) Train and Val have ZERO prompt overlap.
    """
    clusters: Dict[str, List[List[Dict[str, str]]]] = {}
    for sample in category_samples:
        prompt_norm = normalize_text(sample[0]["content"])
        if prompt_norm not in clusters:
            clusters[prompt_norm] = []
        clusters[prompt_norm].append(sample)

    cluster_keys = list(clusters.keys())
    random.seed(42)
    random.shuffle(cluster_keys)

    n_val_clusters = max(1, int(len(cluster_keys) * val_ratio))
    val_keys = set(cluster_keys[:n_val_clusters])
    train_keys = set(cluster_keys[n_val_clusters:])

    train_out = []
    val_out = []

    for k in train_keys:
        train_out.extend(clusters[k])
    for k in val_keys:
        val_out.extend(clusters[k])

    return train_out, val_out


def main():
    print("=" * 60)
    print("Building Stratified, Non-Overfitting, Leakage-Free Vak SFT Dataset")
    print("=" * 60)

    # 1. Load canonical held-out benchmark
    held_out_benchmark = load_held_out_benchmark()
    held_out_set = {normalize_text(item["prompt"]) for item in held_out_benchmark}
    print(f"Loaded {len(held_out_benchmark)} canonical held-out evaluation prompts from data/held_out_benchmark.json")

    # 2. Generate samples per category
    categories = {
        "arithmetic": generate_arithmetic(held_out_set),
        "counting": generate_counting(held_out_set),
        "continuations": generate_continuations(held_out_set),
        "greetings": generate_greetings(held_out_set),
        "identity": generate_identity(held_out_set),
        "facts": generate_facts(held_out_set),
        "fallbacks": generate_fallbacks(held_out_set),
    }

    train_all = []
    val_all = []

    print("\nStratified Category Partitioning (Cluster-based, 0 prompt leakage):")
    for cat_name, cat_samples in categories.items():
        tr, va = split_category_stratified(cat_samples, val_ratio=0.15)
        train_all.extend(tr)
        val_all.extend(va)
        print(f"  • {cat_name:15s}: {len(tr):4d} train, {len(va):3d} val")

    # Shuffle splits deterministically
    random.seed(42)
    random.shuffle(train_all)
    random.shuffle(val_all)

    # 3. Verify zero leakage and schema integrity
    train_prompts_norm = {normalize_text(s[0]["content"]) for s in train_all}
    val_prompts_norm = {normalize_text(s[0]["content"]) for s in val_all}

    inter = train_prompts_norm.intersection(val_prompts_norm)
    assert len(inter) == 0, f"Critical error: {len(inter)} prompts leaked between train and val: {inter}"

    inter_held_train = train_prompts_norm.intersection(held_out_set)
    assert len(inter_held_train) == 0, f"Critical error: {len(inter_held_train)} prompts leaked into held-out suite: {inter_held_train}"

    inter_held_val = val_prompts_norm.intersection(held_out_set)
    assert len(inter_held_val) == 0, f"Critical error: {len(inter_held_val)} val prompts leaked into held-out suite: {inter_held_val}"

    # Verify both splits contain core conversational vocabulary (e.g. hello, greetings, arithmetic, noise)
    hello_train = sum("hello" in s[0]["content"].lower() for s in train_all)
    hello_val = sum("hello" in s[0]["content"].lower() for s in val_all)
    
    noise_tokens = {"d", "c", "x", "z", "q", "m", "p", "b", "j", "v", "asdf", "qwerty", "zxcv", "idk", "blah", "???", "...", "!", "??", "!?"}
    noise_train = sum(normalize_text(s[0]["content"]) in noise_tokens or s[0]["content"] in noise_tokens for s in train_all)
    noise_val = sum(normalize_text(s[0]["content"]) in noise_tokens or s[0]["content"] in noise_tokens for s in val_all)

    assert hello_train > 0 and hello_val > 0, f"Stratification failed: hello_train={hello_train}, hello_val={hello_val}"
    assert noise_train > 0 and noise_val > 0, f"Stratification failed: noise_train={noise_train}, noise_val={noise_val}"

    print("\nSplit Verification:")
    print(f"  • Train total: {len(train_all):,} sequences ({len(train_prompts_norm):,} unique prompts)")
    print(f"  • Val total:   {len(val_all):,} sequences ({len(val_prompts_norm):,} unique prompts)")
    print(f"  • Train/Val prompt overlap: 0 (Strictly Disjoint)")
    print(f"  • Overlap with Held-Out 50 Benchmark: 0 (Strictly Disjoint)")
    print(f"  • 'hello' distribution: {hello_train} in train, {hello_val} in val (Balanced)")
    print(f"  • noise distribution:   {noise_train} in train, {noise_val} in val (Balanced)")

    # 4. Save to files with is_synthetic: true provenance
    os.makedirs("data/processed", exist_ok=True)
    
    train_path = "data/processed/sft_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_all:
            record = {"messages": s, "is_synthetic": True}
            f.write(json.dumps(record) + "\n")

    val_path = "data/processed/sft_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_all:
            record = {"messages": s, "is_synthetic": True}
            f.write(json.dumps(record) + "\n")

    # Backward compatibility
    conv_path = "data/processed/sft_conversations.jsonl"
    with open(conv_path, "w", encoding="utf-8") as f:
        for s in train_all:
            record = {"messages": s, "is_synthetic": True}
            f.write(json.dumps(record) + "\n")

    # 5. Compute SHA-256 and file sizes to update manifest
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
            "unique_prompts": len(train_prompts_norm)
        }
        manifest["artifacts"]["sft_val.jsonl"] = {
            "path": val_path,
            "description": "Supervised fine-tuning held-out validation split (disjoint from train)",
            "sha256": val_sha,
            "size_bytes": val_size,
            "sequence_count": len(val_all),
            "unique_prompts": len(val_prompts_norm)
        }
        manifest["artifacts"]["sft_conversations.jsonl"] = {
            "path": conv_path,
            "description": "Supervised fine-tuning training sequences (canonical training mirror)",
            "sha256": conv_sha,
            "size_bytes": conv_size,
            "sequence_count": len(train_all)
        }
        manifest["artifacts"]["held_out_benchmark.json"] = {
            "path": "data/held_out_benchmark.json",
            "description": "Canonical 50-prompt held-out generalization benchmark with grading criteria",
            "sha256": bench_sha,
            "size_bytes": bench_size,
            "prompt_count": len(held_out_benchmark)
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Updated data/manifest.json with new SFT artifacts, exact checksums, and provenance.")

    print(f"\nFiles generated successfully:")
    print(f"  • {train_path} ({len(train_all):,} sequences, SHA: {train_sha[:8]}...)")
    print(f"  • {val_path} ({len(val_all):,} sequences, SHA: {val_sha[:8]}...)")


if __name__ == "__main__":
    main()
