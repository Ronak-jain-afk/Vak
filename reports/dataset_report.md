# Vak Dataset Report

## 1. Executive Summary
The Vak corpus was designed to pretrain a ~29M decoder-only language model and subsequently instruction-tune it for simple conversational interactions on consumer laptop hardware (NVIDIA RTX 4050 6GB VRAM, 16 GB RAM). The pipeline prioritizes high-quality, legally reusable, and permissive open datasets over raw volume.

---

## 2. Source Provenance & Licensing Analysis

| Source | Domain / Purpose | License | Reuse Permissions | Content Type | Acquisition Method |
|---|---|---|---|---|---|
| **HuggingFaceFW/fineweb-edu** (sample-10BT) | High-quality educational web text | ODC-By 1.0 | Free for commercial & research use with attribution | Human-written (filtered by LLama-3-70B educational classifier) | Hugging Face streaming API |
| **HuggingFaceTB/smollm-corpus** (cosmopedia-v2) | Synthetic textbooks and how-to guides | ODC-By 1.0 | Free for commercial & research use with attribution | Synthetic (Mixtral-generated) | Hugging Face streaming API |
| **Avelina/python-edu** | Educational Python code | ODC-By 1.0 | Permissive code modeling and research | Human-written (curated educational code) | Hugging Face streaming API |
| **open-web-math/open-web-math** | Mathematical reasoning and LaTeX | ODC-By 1.0 | Free for commercial & research use with attribution | Human-written (filtered web math) | Hugging Face streaming API |
| **roneneldan/TinyStories** | Fluent narrative prose with basic vocabulary | CDLA-Permissive-1.0 | Open data sharing and permissive reuse | Synthetic (GPT-3.5/GPT-4 generated) | Hugging Face streaming API |
| **HuggingFaceTB/smoltalk** (everyday-conversations) | Greetings, casual chat, small talk | Apache-2.0 | Permissive commercial and research use | Synthetic / Curated | Hugging Face datasets API |

### License Compliance Decisions:
- **ODC-By 1.0 (FineWeb-Edu, Cosmopedia v2, Python-Edu, OpenWebMath):** Fully compliant with local research and model distribution; requires attribution in model card and artifacts.
- **CDLA-Permissive-1.0 (TinyStories):** Grants royalty-free permission to use, copy, modify, and distribute the data and downstream representations.
- **Apache-2.0 (SmolTalk):** Standard permissive open-source license; explicitly permits distribution and fine-tuning.
- **Separation of Concerns:** Human-written web data and synthetic story/chat data are tracked with explicit `is_synthetic` provenance flags in `manifest.json`. SFT conversation data is strictly isolated from pretraining data.

---

## 3. Preprocessing & Quality Pipeline

1. **Unicode & Whitespace Normalization (`clean.py`):**
   - Normalized text to Unicode Normalization Form C (`NFC`).
   - Standardized CRLF to standard `\n` linebreaks.
   - Removed ASCII control characters (`\x00-\x1f`) while preserving tabs and linebreaks.
   - Collapsed consecutive whitespace runs and excessive blank lines.

2. **Heuristic & Spam Filtering (`clean.py`):**
   - Minimum length: 100 characters for web documents; 50 characters for short stories.
   - Maximum length: 40,000 characters to prevent pathological sequence outliers.
   - Line repetition threshold: rejected documents with >25% repetitive lines (navigation menus, repeated links).
   - Alphabetic character ratio: enforced >= 45% alphabetic content to reject binary dumps and character spam.

3. **Deduplication (`deduplicate.py`):**
   - **Exact Deduplication:** SHA-256 hash tracking of normalized text strings.
   - **Near-Duplicate Detection:** 128-permutation MinHash with 3-word shingles and Locality-Sensitive Hashing (LSH) at Jaccard similarity threshold 0.70.

4. **Document-Level Splitting (`split.py`):**
   - Split partitioned strictly by document ID using salted cryptographic hashing (`vak_split_v1_{doc_id}`).
   - Partitions: 90% Train, 5% Validation, 5% Test.
   - Strictly disjoint partitions guarantee **zero data leakage** between pretraining and evaluation sets.

5. **Tokenization & Sharding (`tokenize_shards.py`):**
   - Byte-level BPE tokenizer with 8,192 vocabulary trained across pretraining and chat corpora.
   - Fixed special tokens: `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|end|>`.
   - Tokenized into binary shards (`.bin`) using `uint16` encoding (2 bytes per token), reducing storage footprint by 50% compared to `uint32`.

---

## 4. Current Corpus Statistics

- **Raw Documents Processed:** 1,455,460 raw documents.
- **Exact & Near Duplicates Filtered:** 4,642 exact + 818 near-duplicates filtered (1,450,000 unique documents preserved).
- **Document Splits:**
  - `train`: 1,304,645 documents (**1,057,535,818 tokens** across 106 binary shards)
  - `val`: 72,731 documents (**59,026,719 tokens** across 6 binary shards)
  - `test`: 72,624 documents (**59,186,744 tokens** across 6 binary shards)
  - `sft_conversations`: 2,959 conversation threads (SmolTalk everyday-conversations)
- **Total Processed Tokens:** **1,175,749,281 tokens** across 118 binary `uint16` shards totaling **≈2.2 GiB**.
- **Measured Domain Mix (train split, by tokens):** fineweb-edu 50.2%, cosmopedia-v2 23.5%, python-edu 12.0%, open-web-math 8.9%, tinystories 5.3% (target 50/25/12/8/5; document quotas in `data_pipeline/download.py` are tuned to the measured per-source document lengths, which differ by up to ~3.6×).
- **Manifest:** Full provenance, file hashes (SHA-256), and timestamps are recorded in `data/manifest.json`.

---

## 5. Storage & Full Scale Projections

- **Disk Budget Constraint:** Total storage available on WSL filesystem is bounded; target budget is < 100 GB for all raw, processed, and tokenized data combined.
- **Hardware Measured Throughput:** 24,419 tokens/second (bf16, micro-batch 8 × 1,024 tokens, optimizer included) on NVIDIA RTX 4050 Laptop GPU.
- **Recommended 36-Hour Pretraining Target:** 3.00B tokens (3,000,000,000 tokens).
- **Tokenized Storage Required for 3.0B Tokens:**
  - At 2 bytes per token (`uint16`), 3.0B tokens require **6.0 GB** of binary shard storage.
  - Raw + Processed text: ~12-15 GB.
  - Total required disk space: **< 25 GB**, well within the 100 GB limit.
