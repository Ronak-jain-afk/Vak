import hashlib
import json
import os
import random
import re
from typing import Dict, List, Set, Tuple

# Set seed for deterministic generation
random.seed(42)

# =====================================================================
# HELD-OUT GENERALIZATION BENCHMARK (50 STRICTLY UNSEEN PROMPTS)
# NONE OF THESE PROMPTS MAY APPEAR IN SFT TRAINING OR VALIDATION DATA!
# =====================================================================
HELD_OUT_50_PROMPTS = [
    # 1. Unseen Greetings & Colloquial Phrasing
    "Howdy there!",
    "Good afternoon!",
    "What's up?",
    "How are things going with you?",
    "Hope you are having a wonderful afternoon.",
    "Hiya, nice to see you!",
    "Long time no see, how have you been?",
    "Good day to you.",
    "How is your week turning out?",
    "Greetings from a friend!",
    # 2. Unseen Identity & Meta Queries
    "What model are you based on?",
    "Who programmed your model architecture?",
    "Are you an artificial intelligence or a human?",
    "Could you describe your core capabilities?",
    "What was the main purpose behind creating you?",
    "Tell me about what you can do for me.",
    "Where do you run your computations?",
    "Do you have a nickname?",
    "Can you explain your background?",
    "What kind of conversations do you specialize in?",
    # 3. Unseen Facts & Paraphrased Knowledge
    "What city serves as the capital of France?",
    "Which European city is the capital of Germany?",
    "What color is an emerald gemstone?",
    "What liquid do cows produce for human consumption?",
    "At what Celsius temperature does water freeze?",
    "What is the largest hot desert on Earth?",
    "What gas do humans need to inhale to survive?",
    "Which bird is well known for being unable to fly?",
    "What color is the inside of a ripe watermelon?",
    "What instrument has black and white keys?",
    # 4. Unseen Arithmetic & Number Queries (including contractions & spacing variants)
    "whats 3+3",
    "what is 6 + 7?",
    "whats 5+5",
    "what is 9 - 4?",
    "what is 8 - 3?",
    "what is 7 + 8?",
    "calculate 4 + 7",
    "what is 10 - 6?",
    "what is 3 * 4?",
    "what number comes after 15?",
    # 5. Unseen Instructions & Sequences
    "Count backwards from 4 to 1.",
    "Count from 7 to 12.",
    "Recite numbers 1 through 5.",
    "Count odd numbers from 11 to 17.",
    "Say hello in Spanish.",
    # 6. Reasoning & Commonsense
    "If you have 5 oranges and give away 3, how many remain?",
    "Is an ocean deeper than a swimming pool?",
    "Can an airplane fly faster than a person can sprint?",
    "Do heavy rocks float when dropped into water?",
    "Why do people wear warm coats during the winter?",
]

def normalize_text(t: str) -> str:
    """Normalize text for strict leakage checking (lowercase, stripped punctuation & whitespace)."""
    t = t.lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

HELD_OUT_NORMALIZED = {normalize_text(p) for p in HELD_OUT_50_PROMPTS}


def generate_arithmetic() -> List[List[Dict[str, str]]]:
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
                if normalize_text(p) not in HELD_OUT_NORMALIZED:
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
                if normalize_text(p) not in HELD_OUT_NORMALIZED:
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
                if normalize_text(p) not in HELD_OUT_NORMALIZED:
                    items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_counting() -> List[List[Dict[str, str]]]:
    """Generates counting in various forward and backward ranges."""
    items = []
    
    ranges = [
        (1, 5), (1, 10), (1, 15), (1, 20),
        (5, 10), (5, 15), (10, 15), (11, 20), (15, 20),
        (2, 6), (3, 8), (4, 9), (6, 11), (8, 14)
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
            if normalize_text(p) not in HELD_OUT_NORMALIZED:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    # Backwards counting
    back_ranges = [(5, 1), (10, 1), (10, 5), (6, 2), (8, 3)]
    for start, end in back_ranges:
        nums_str = ", ".join(str(n) for n in range(start, end - 1, -1)) + "."
        templates = [
            (f"count backwards from {start} to {end}", nums_str),
            (f"count down from {start} to {end}", nums_str),
            (f"count backwards from {start} down to {end}", nums_str),
        ]
        for p, ans in templates:
            if normalize_text(p) not in HELD_OUT_NORMALIZED:
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
            if normalize_text(p) not in HELD_OUT_NORMALIZED:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_continuations() -> List[List[Dict[str, str]]]:
    """Multi-turn threads including conversational ellipsis and counting continuations."""
    items = []

    # 1. Counting continuations (1..10 -> continue to 20; 1..5 -> continue to 10; etc.)
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
        for _ in range(30):
            thread = list(prefix) + [
                {"role": "user", "content": user_next},
                {"role": "assistant", "content": asst_next}
            ]
            items.append(thread)

    # 2. Conversational multi-turn threads
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
        for _ in range(40):
            items.append(d)

    return items


def generate_identity_and_chat() -> List[List[Dict[str, str]]]:
    """Generates identity, greetings, and conversational pleasantries."""
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
        for _ in range(25):
            if normalize_text(p) not in HELD_OUT_NORMALIZED:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    greetings = [
        ("hi", "Hello! How can I help you today?"),
        ("hi!", "Hi there! How can I help you today?"),
        ("hello", "Hello! How can I assist you today?"),
        ("hello!", "Hello! It is great to chat with you. How can I help?"),
        ("hey", "Hey! What can I do for you today?"),
        ("hey there", "Hey there! How is your day going?"),
        ("good morning", "Good morning! How can I help you today?"),
        ("good morning!", "Good morning! I hope you have a great day ahead. How can I assist?"),
        ("good evening", "Good evening! How can I help you tonight?"),
        ("how are you?", "I am doing well, thank you! How are you doing?"),
        ("how are you today?", "I'm doing great! How has your day been?"),
        ("nice to meet you", "Nice to meet you too! How can I help you today?"),
        ("nice to meet you.", "Nice to meet you too! How can I help you today?"),
        ("thank you", "You're welcome! Let me know if you need anything else."),
        ("thanks!", "Happy to help! Let me know if you have more questions."),
        ("goodbye", "Goodbye! Have a wonderful day!"),
        ("bye!", "Bye! Take care!"),
    ]

    for p, ans in greetings:
        for _ in range(25):
            if normalize_text(p) not in HELD_OUT_NORMALIZED:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_facts() -> List[List[Dict[str, str]]]:
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
        for _ in range(15):
            if normalize_text(p) not in HELD_OUT_NORMALIZED:
                items.append([{"role": "user", "content": p}, {"role": "assistant", "content": ans}])

    return items


def generate_fallbacks() -> List[List[Dict[str, str]]]:
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
        for _ in range(20):
            reply = random.choice(replies)
            if normalize_text(noise) not in HELD_OUT_NORMALIZED:
                items.append([{"role": "user", "content": noise}, {"role": "assistant", "content": reply}])

    return items


def main():
    print("Building Diverse, Leakage-Free Vak SFT Dataset...")
    
    all_samples = []
    all_samples.extend(generate_arithmetic())
    all_samples.extend(generate_counting())
    all_samples.extend(generate_continuations())
    all_samples.extend(generate_identity_and_chat())
    all_samples.extend(generate_facts())
    all_samples.extend(generate_fallbacks())

    print(f"Total candidate conversations generated: {len(all_samples):,}")

    # Deduplicate and verify strict zero-leakage against Held-out 50 prompts
    filtered_samples = []
    leakage_count = 0
    for sample in all_samples:
        first_user_prompt = sample[0]["content"]
        norm_prompt = normalize_text(first_user_prompt)
        if norm_prompt in HELD_OUT_NORMALIZED:
            leakage_count += 1
            continue
        filtered_samples.append(sample)

    print(f"Filtered out {leakage_count} samples matching held-out evaluation prompts.")
    print(f"Preserved {len(filtered_samples):,} clean conversation threads.")

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(filtered_samples)

    # Cryptographic SHA-256 partition on user prompt into Train (90%) and Val (10%)
    train_samples = []
    val_samples = []
    train_prompts_norm: Set[str] = set()
    val_prompts_norm: Set[str] = set()

    for sample in filtered_samples:
        prompt_key = sample[0]["content"].strip().lower()
        h = int(hashlib.sha256(prompt_key.encode("utf-8")).hexdigest(), 16)
        norm_p = normalize_text(sample[0]["content"])
        
        # 10% held-out SFT validation split
        if (h % 10) == 0:
            # Check to ensure disjoint prompts
            if norm_p not in train_prompts_norm:
                val_samples.append(sample)
                val_prompts_norm.add(norm_p)
            else:
                train_samples.append(sample)
        else:
            if norm_p not in val_prompts_norm:
                train_samples.append(sample)
                train_prompts_norm.add(norm_p)
            else:
                val_samples.append(sample)

    # Strict assertion checks
    prompt_leakage = train_prompts_norm.intersection(val_prompts_norm)
    assert len(prompt_leakage) == 0, f"Error: {len(prompt_leakage)} prompts leaked between train and val!"
    
    heldout_train_leakage = train_prompts_norm.intersection(HELD_OUT_NORMALIZED)
    assert len(heldout_train_leakage) == 0, f"Error: {len(heldout_train_leakage)} prompts leaked into held-out suite!"

    print("\nDataset Partition Verification:")
    print(f"  • Train sequences: {len(train_samples):,} (Unique prompts: {len(train_prompts_norm):,})")
    print(f"  • Held-out SFT Val sequences: {len(val_samples):,} (Unique prompts: {len(val_prompts_norm):,})")
    print(f"  • Train/Val Prompt Overlap: 0 (Strictly Disjoint)")
    print(f"  • Overlap with Held-Out 50 Benchmark: 0 (Strictly Disjoint)")

    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    
    train_file = "data/processed/sft_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps({"messages": s}) + "\n")
            
    val_file = "data/processed/sft_val.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps({"messages": s}) + "\n")

    # For backward compatibility, sft_conversations.jsonl is the train split
    conv_file = "data/processed/sft_conversations.jsonl"
    with open(conv_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps({"messages": s}) + "\n")

    print(f"\nFiles saved cleanly:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {conv_file}")


if __name__ == "__main__":
    main()
