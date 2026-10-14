import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple


def assign_split(doc_id: str, salt: str = "vak_split_v1") -> str:
    # Deterministic salted hash partition prevents data leakage between splits
    h = hashlib.sha256(f"{salt}_{doc_id}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 90:
        return "train"
    elif bucket < 95:
        return "val"
    else:
        return "test"


def split_documents(
    documents: List[Dict],
    output_dir: str = "data/processed",
) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "train": os.path.join(output_dir, "train.jsonl"),
        "val": os.path.join(output_dir, "val.jsonl"),
        "test": os.path.join(output_dir, "test.jsonl"),
    }

    files = {k: open(v, "w", encoding="utf-8") for k, v in paths.items()}
    counts = {"train": 0, "val": 0, "test": 0}

    try:
        for doc in documents:
            split = assign_split(doc["id"])
            files[split].write(json.dumps(doc, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in files.values():
            f.close()

    return counts
