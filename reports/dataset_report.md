# Vak Dataset Report

## 1. Executive Summary
The Vak corpus was designed to pretrain a ~29M decoder-only language model and subsequently instruction-tune it for simple conversational interactions on consumer laptop hardware (NVIDIA RTX 4050 6GB VRAM, 16 GB RAM). The pipeline prioritizes high-quality, legally reusable, and permissive open datasets over raw volume.

---

## 2. Source Provenance & Licensing Analysis

| Source | Domain / Purpose | License | Reuse Permissions | Content Type | Acquisition Method |
|---|---|---|---|---|---|
| **HuggingFaceFW/fineweb-edu** (sample-10BT) | High-quality educational web text | ODC-By 1.0 | Free for commercial & research use with attribution | Human-written (filtered by LLama-3-70B educational classifier) | Hugging Face streaming API |
| **roneneldan/TinyStories** | Fluent narrative prose with basic vocabulary | CDLA-Permissive-1.0 | Open data sharing and permissive reuse | Synthetic (GPT-3.5/GPT-4 generated) | Hugging Face streaming API |
| **HuggingFaceTB/smoltalk** (everyday-conversations) | Greetings, casual chat, small talk | Apache-2.0 | Permissive commercial and research use | Synthetic / Curated | Hugging Face datasets API |

### License Compliance Decisions:
- **ODC-By 1.0 (FineWeb-Edu):** Fully compliant with local research and model distribution; requires attribution in model card and artifacts.
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

- **Raw Documents Processed:** 867,077 raw documents.
- **Exact & Near Duplicates Filtered:** 16,576 exact + 501 near-duplicates filtered (850,000 unique documents preserved).
- **Document Splits:**
  - `train`: 765,193 documents (**496,103,579 tokens** across 50 binary shards)
  - `val`: 42,382 documents (**27,423,378 tokens** across 28 binary shards)
  - `test`: 42,425 documents
  - `sft_conversations`: 11,650 conversation threads (balanced alignment mixture across counting, math, identity, greetings, multi-turn, fallbacks)
- **Total Processed Tokens:** **523,526,957 tokens** across 78 binary `uint16` shards totaling **999 MB**.
- **Manifest:** Full provenance, file hashes (SHA-256), and timestamps are recorded in `data/manifest.json`.

---

## 5. Storage & Full Scale Projections

- **Disk Budget Constraint:** Total storage available on WSL filesystem is bounded; target budget is < 100 GB for all raw, processed, and tokenized data combined.
- **Hardware Measured Throughput:** 30,038 tokens/second on NVIDIA RTX 4050 Laptop GPU.
- **Recommended 36-Hour Pretraining Target:** 3.00B tokens (3,000,000,000 tokens).
- **Tokenized Storage Required for 3.0B Tokens:**
  - At 2 bytes per token (`uint16`), 3.0B tokens require **6.0 GB** of binary shard storage.
  - Raw + Processed text: ~12-15 GB.
  - Total required disk space: **< 25 GB**, well within the 100 GB limit.
