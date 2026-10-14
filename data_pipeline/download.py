import argparse
import json
import os
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm

from data_pipeline.clean import is_quality_document, normalize_text
from data_pipeline.deduplicate import Deduplicator
from data_pipeline.manifest import ManifestManager
from data_pipeline.split import assign_split


def stream_source_to_splits(
    dataset_iter,
    source_name: str,
    prefix: str,
    license_name: str,
    is_synthetic: bool,
    max_docs: int,
    dedup: Deduplicator,
    split_files: Dict[str, any],
    counts: Dict[str, int],
    raw_sample_file=None,
    max_raw_samples: int = 500,
    min_chars: int = 100,
):
    valid_count = 0
    raw_saved = 0

    pbar = tqdm(total=max_docs, desc=source_name)
    for row in dataset_iter:
        text = row.get("text", "")
        if not text:
            continue

        if raw_sample_file is not None and raw_saved < max_raw_samples:
            raw_sample_file.write(json.dumps({"text": text, "source": source_name}, ensure_ascii=False) + "\n")
            raw_saved += 1

        norm = normalize_text(text)
        ok, _ = is_quality_document(norm, min_chars=min_chars)
        if not ok:
            continue

        doc_id = f"{prefix}_{valid_count}"
        is_dupe, _ = dedup.is_duplicate(norm, doc_id=doc_id)
        if is_dupe:
            continue

        split = assign_split(doc_id)
        record = {
            "id": doc_id,
            "text": norm,
            "source": source_name,
            "license": license_name,
            "is_synthetic": is_synthetic,
        }
        split_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] += 1
        valid_count += 1
        pbar.update(1)

        if valid_count >= max_docs:
            break

    pbar.close()
    return valid_count


def process_pretrain_sources(
    manifest: ManifestManager,
    max_fineweb_docs: int = 100000,
    max_tinystories_docs: int = 120000,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> Dict[str, int]:
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    dedup = Deduplicator(threshold=0.70)
    split_paths = {
        "train": os.path.join(processed_dir, "train.jsonl"),
        "val": os.path.join(processed_dir, "val.jsonl"),
        "test": os.path.join(processed_dir, "test.jsonl"),
    }
    split_files = {k: open(v, "w", encoding="utf-8") for k, v in split_paths.items()}
    counts = {"train": 0, "val": 0, "test": 0}

    raw_sample_path = os.path.join(raw_dir, "raw_samples.jsonl")
    raw_sample_f = open(raw_sample_path, "w", encoding="utf-8")

    try:
        # 1. FineWeb-Edu
        print(f"\n[1/3] Streaming FineWeb-Edu sample-10BT (Target: {max_fineweb_docs:,} documents)...")
        manifest.record_source(
            name="fineweb-edu-sample-10bt",
            source_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            license_name="ODC-By 1.0",
            permitted_uses="Commercial and research use with attribution",
            language="en",
            domain="educational_web_text",
            acquisition_method="huggingface_streaming_api",
            is_synthetic=False,
            raw_document_count=max_fineweb_docs,
        )
        fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        stream_source_to_splits(
            fw_ds,
            source_name="fineweb-edu",
            prefix="fw",
            license_name="ODC-By 1.0",
            is_synthetic=False,
            max_docs=max_fineweb_docs,
            dedup=dedup,
            split_files=split_files,
            counts=counts,
            raw_sample_file=raw_sample_f,
            min_chars=100,
        )

        # 2. TinyStories
        print(f"\n[2/3] Streaming TinyStories (Target: {max_tinystories_docs:,} documents)...")
        manifest.record_source(
            name="tinystories",
            source_url="https://huggingface.co/datasets/roneneldan/TinyStories",
            license_name="CDLA-Permissive-1.0",
            permitted_uses="Permissive data reuse and research",
            language="en",
            domain="children_stories_and_dialogue",
            acquisition_method="huggingface_streaming_api",
            is_synthetic=True,
            raw_document_count=max_tinystories_docs,
        )
        ts_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        stream_source_to_splits(
            ts_ds,
            source_name="tinystories",
            prefix="ts",
            license_name="CDLA-Permissive-1.0",
            is_synthetic=True,
            max_docs=max_tinystories_docs,
            dedup=dedup,
            split_files=split_files,
            counts=counts,
            raw_sample_file=raw_sample_f,
            min_chars=50,
        )

    finally:
        for f in split_files.values():
            f.close()
        raw_sample_f.close()

    stats = dedup.get_stats()
    manifest.record_processing("deduplication", stats)
    manifest.record_splits(counts)
    print(f"\nDeduplication summary: {stats}")
    print(f"Splits summary: {counts}")
    return counts


def process_sft_source(
    manifest: ManifestManager,
    processed_dir: str = "data/processed",
) -> int:
    print("\n[3/3] Checking SmolTalk everyday-conversations for SFT...")
    manifest.record_source(
        name="smoltalk-everyday-conversations",
        source_url="https://huggingface.co/datasets/HuggingFaceTB/smoltalk",
        license_name="Apache-2.0",
        permitted_uses="Commercial and non-commercial research use",
        language="en",
        domain="conversational_chat_and_greetings",
        acquisition_method="huggingface_api",
        is_synthetic=True,
        raw_document_count=2260,
    )

    out_path = os.path.join(processed_dir, "sft_conversations.jsonl")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        with open(out_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
        print(f"SFT conversations already exist ({line_count} threads at {out_path})")
        return line_count

    ds = load_dataset("HuggingFaceTB/smoltalk", name="everyday-conversations", split="train")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            msgs = row["messages"]
            valid_msgs = []
            for m in msgs:
                content = normalize_text(m["content"])
                if content:
                    valid_msgs.append({"role": m["role"], "content": content})
            if len(valid_msgs) >= 2:
                f.write(json.dumps({"messages": valid_msgs, "is_synthetic": True}, ensure_ascii=False) + "\n")
                count += 1

    print(f"Saved {count} SFT conversation threads to {out_path}")
    manifest.record_artifact(out_path, "Supervised fine-tuning conversation threads")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Vak corpora at scale")
    parser.add_argument("--fineweb_docs", type=int, default=100000)
    parser.add_argument("--tinystories_docs", type=int, default=120000)
    args = parser.parse_args()

    manifest = ManifestManager("data/manifest.json")
    process_pretrain_sources(
        manifest,
        max_fineweb_docs=args.fineweb_docs,
        max_tinystories_docs=args.tinystories_docs,
    )
    process_sft_source(manifest)
    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()
