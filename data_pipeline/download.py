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
    domain: str = "text",
    preserve_indentation: bool = False,
    min_score: Optional[float] = None,
) -> int:
    valid_count = 0
    raw_saved = 0

    pbar = tqdm(total=max_docs, desc=source_name)
    for row in dataset_iter:
        if min_score is not None and row.get("score", 0.0) < min_score:
            continue

        text = row.get("text", "") or row.get("content", "")
        if not text:
            continue

        if raw_sample_file is not None and raw_saved < max_raw_samples:
            raw_sample_file.write(json.dumps({"text": text, "source": source_name}, ensure_ascii=False) + "\n")
            raw_saved += 1

        norm = normalize_text(text, preserve_indentation=preserve_indentation)
        ok, _ = is_quality_document(norm, domain=domain, min_chars=min_chars)
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
    max_fineweb_docs: int = 715000,
    max_cosmopedia_docs: int = 310000,
    max_python_docs: int = 240000,
    max_math_docs: int = 125000,
    max_tinystories_docs: int = 225000,
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
        # 1. FineWeb-Edu (50% backbone)
        if max_fineweb_docs > 0:
            print(f"\n[1/5] Streaming FineWeb-Edu sample-10BT (Target: {max_fineweb_docs:,} documents)...")
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
                domain="text",
                preserve_indentation=False,
            )

        # 2. Cosmopedia v2 (25% synthetic textbooks and how-to guides)
        if max_cosmopedia_docs > 0:
            print(f"\n[2/5] Streaming Cosmopedia v2 (Target: {max_cosmopedia_docs:,} documents)...")
            manifest.record_source(
                name="cosmopedia-v2",
                source_url="https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-v2",
                license_name="ODC-By 1.0",
                permitted_uses="Commercial and research use with attribution",
                language="en",
                domain="synthetic_textbooks_and_howtos",
                acquisition_method="huggingface_streaming_api",
                is_synthetic=True,
                raw_document_count=max_cosmopedia_docs,
            )
            cosmo_ds = load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True)
            stream_source_to_splits(
                cosmo_ds,
                source_name="cosmopedia-v2",
                prefix="cosmo",
                license_name="ODC-By 1.0",
                is_synthetic=True,
                max_docs=max_cosmopedia_docs,
                dedup=dedup,
                split_files=split_files,
                counts=counts,
                raw_sample_file=raw_sample_f,
                min_chars=100,
                domain="text",
                preserve_indentation=False,
            )

        # 3. Python-Edu (12% educational Python code & logic)
        if max_python_docs > 0:
            print(f"\n[3/5] Streaming Python-Edu (Target: {max_python_docs:,} documents)...")
            manifest.record_source(
                name="python-edu",
                source_url="https://huggingface.co/datasets/Avelina/python-edu",
                license_name="ODC-By 1.0 / MIT",
                permitted_uses="Permissive research and code modeling",
                language="en",
                domain="educational_python_code",
                acquisition_method="huggingface_streaming_api",
                is_synthetic=False,
                raw_document_count=max_python_docs,
            )
            py_ds = load_dataset("Avelina/python-edu", split="train", streaming=True)
            stream_source_to_splits(
                py_ds,
                source_name="python-edu",
                prefix="py",
                license_name="ODC-By 1.0 / MIT",
                is_synthetic=False,
                max_docs=max_python_docs,
                dedup=dedup,
                split_files=split_files,
                counts=counts,
                raw_sample_file=raw_sample_f,
                min_chars=50,
                domain="code",
                preserve_indentation=True,
                min_score=3.8,
            )

        # 4. OpenWebMath (8% formal mathematical reasoning & LaTeX)
        if max_math_docs > 0:
            print(f"\n[4/5] Streaming OpenWebMath (Target: {max_math_docs:,} documents)...")
            manifest.record_source(
                name="open-web-math",
                source_url="https://huggingface.co/datasets/open-web-math/open-web-math",
                license_name="ODC-By 1.0",
                permitted_uses="Commercial and research use with attribution",
                language="en",
                domain="mathematical_reasoning_and_latex",
                acquisition_method="huggingface_streaming_api",
                is_synthetic=False,
                raw_document_count=max_math_docs,
            )
            math_ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
            stream_source_to_splits(
                math_ds,
                source_name="open-web-math",
                prefix="math",
                license_name="ODC-By 1.0",
                is_synthetic=False,
                max_docs=max_math_docs,
                dedup=dedup,
                split_files=split_files,
                counts=counts,
                raw_sample_file=raw_sample_f,
                min_chars=80,
                domain="math",
                preserve_indentation=False,
            )

        # 5. TinyStories (5% fluency regularizer)
        if max_tinystories_docs > 0:
            print(f"\n[5/5] Streaming TinyStories (Target: {max_tinystories_docs:,} documents)...")
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
                domain="text",
                preserve_indentation=False,
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
    print("\nChecking SmolTalk everyday-conversations for SFT...")
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
    parser = argparse.ArgumentParser(description="Download and preprocess Vak 1.0B corpora across 5 domains")
    parser.add_argument("--fineweb_docs", type=int, default=715000, help="FineWeb-Edu document quota (50%)")
    parser.add_argument("--cosmopedia_docs", type=int, default=310000, help="Cosmopedia v2 document quota (25%)")
    parser.add_argument("--python_docs", type=int, default=240000, help="Python-Edu document quota (12%)")
    parser.add_argument("--math_docs", type=int, default=125000, help="OpenWebMath document quota (8%)")
    parser.add_argument("--tinystories_docs", type=int, default=225000, help="TinyStories document quota (5%)")
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Uniform scaling factor for test/dry runs")
    args = parser.parse_args()

    scale = args.scale_factor
    fineweb = int(args.fineweb_docs * scale)
    cosmopedia = int(args.cosmopedia_docs * scale)
    python_docs = int(args.python_docs * scale)
    math_docs = int(args.math_docs * scale)
    tinystories = int(args.tinystories_docs * scale)

    manifest = ManifestManager("data/manifest.json")
    process_pretrain_sources(
        manifest,
        max_fineweb_docs=fineweb,
        max_cosmopedia_docs=cosmopedia,
        max_python_docs=python_docs,
        max_math_docs=math_docs,
        max_tinystories_docs=tinystories,
    )
    process_sft_source(manifest)
    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()
