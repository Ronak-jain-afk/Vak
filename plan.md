You are a senior AI researcher, developer, and AI data trainer.

You are tasked to train a small language model from scratch. The hard ceiling is **30.0M parameters**, the target is **~29M**, and the accepted range is **28.0M-30.0M**, counted from the instantiated model (not estimated).

---

# Overview
- **Project name:** Vak
- **One-line goal:** Train a ~29M-parameter decoder-only LLM from scratch that holds simple conversations.
- **Primary audience:** Developers reviewing the code (portfolio / learning project)
- **Non-goals:** Broad factual knowledge, reasoning, long contexts, multilingual support

---

# Hardware specs
The model will be trained on the following hardware:
- GPU + VRAM: Nvidia RTX 4050 laptop GPU with 6 GB VRAM.
- CPU: 13th Gen Intel Core i5-13420H.
- RAM: 16 GB
- Storage: 512 GB SSD (only 172 GB available). Keep raw data, processed text, and tokenized shards combined under ~100 GB.

With 6 GB of VRAM, use a small micro-batch and gradient accumulation to reach the target batch size.

---

# Technical Environment

1. Language: Python, CUDA (if needed)
2. Run everything inside **WSL2 Ubuntu**. `torch.compile` needs Triton, which does not work on native Windows. Keep the code, venv, and data on the WSL (Linux) filesystem, not under `/mnt/c`, for I/O speed.
3. A Python virtual environment (already created). If it was created on the Windows side, recreate it inside WSL.
4. PyTorch >= 2.9 with a CUDA build (includes `torch.optim.Muon`). If Muon is unavailable, implement it by hand.
5. Libraries: PyTorch, NumPy, tqdm, tokenizers, pyyaml, datasets, huggingface_hub, tensorboard, lm-eval, matplotlib, pytest, datasketch (near-duplicate detection)

---

# Requirements

- The model needs to be efficient, and have at least basic conversation capabilities.
> For example, it should be able to exchange greetings and have casual talk, I know there are limitations in a 29M model, but still try your best.

- Write the code the way a careful human developer would. Concretely:
  - Plain, descriptive names; no decorative banners, emoji, or ASCII art in code or comments.
  - Comments only where the reasoning is non-obvious. No docstrings on trivial functions.
  - Each non-obvious architectural choice (RoPE, GQA, QK-norm, tied embeddings, scaled init) gets a one- or two-line comment explaining why, not a paragraph.
  - No unnecessary classes, abstractions, or defensive try/except blocks.
  - One clear model file; small modules elsewhere; settings come from a single config file.

- The present working directory is completely empty. Your task is not only to write efficient and modern code, but also to research best-fit datasets on the internet and download them. If a downloaded dataset is raw, pre-process it too.

---

# Dataset
You are also the AI data trainer. You are tasked to research, collect, and create a proper, advanced dataset for this project, and apply preprocessing where needed. You have internet access via the Hugging Face MCP, Fetch MCP, Playwright MCP, and GitHub MCP.

## Objectives

- Create a diverse corpus suitable for general language modeling and eventual conversational instruction tuning.
- Prioritize high-quality, legally reusable data over raw dataset size.
- Prefer free datasets and local processing. Do not require paid APIs or cloud GPUs.
- Use Hugging Face datasets and metadata as the primary discovery mechanism. Use Playwright and Fetch only for targeted source discovery and inspection.

## Dataset stages
1. Discover candidate datasets covering general prose, educational explanations, stories, technical documentation, and natural conversation.
2. Record each dataset's name, source URL, version or revision, license, permitted uses, language, domain, size, acquisition method, and whether the content is human-written or synthetic (LLM-generated).
3. Reject sources with unclear reuse rights, incompatible licenses, access restrictions, or prohibited scraping terms. Never assume publicly accessible means freely reusable.
4. Download only approved sources using their official dataset releases, APIs, or documented download mechanisms.
5. Normalize Unicode and whitespace; remove boilerplate, empty records, malformed text, and excessively repetitive content.
6. Deduplicate exact matches and near-duplicates using hashes and a scalable similarity method.
7. Filter spam, SEO pages, navigation text, low-information content, and unwanted sensitive or personal information.
8. Preserve source provenance and license metadata throughout processing.
9. Split by source/document before chunking into train, validation, and test sets to reduce leakage.
10. Generate tokenizer training input, tokenized shards, dataset statistics, and a reproducible manifest.

## Deliverables
- A configurable Python pipeline with separate discovery, download, cleaning, deduplication, splitting, and tokenization modules.
- A manifest containing source provenance, licensing, processing decisions, document counts, token counts, and hashes.
- Automated tests for deduplication, leakage, malformed records, split integrity, and reproducibility.
- A report describing dataset composition, quality limitations, license concerns, and estimated token counts.
- A small end-to-end sample run before any large download.

## Constraints
- Optimize for Windows + WSL Ubuntu, 16 GB system RAM, and an RTX 4050 Laptop GPU with 6 GB VRAM.
- Keep raw data, processed text, and tokenized shards in separate directories.
- Do not download massive corpora without first reporting estimated disk usage and asking for approval.
- Do not generate conversations yourself and present them as human-written data. Existing datasets that are synthetic (LLM-generated) are allowed if their license permits reuse and the manifest labels them `synthetic`. Human-written and synthetic sources must never be mixed without that label.
- Keep pretraining text separate from supervised conversation examples.
- Do not begin model training until the corpus passes validation.

---

# Training
- **Optimizer:** Muon (2D hidden weights) + AdamW (embeddings, norms, 1D parameters). Starting learning rates, to be tuned: ~0.02 for Muon, 1e-3 to 3e-3 for AdamW.
- **Schedule:** short linear warmup, then WSD (warmup-stable-decay) or cosine decay to ~10% of peak.
- **Precision:** bf16 autocast, `torch.compile`, gradient clipping at 1.0.
- **Regularization:** weight decay ~0.1 on matrices only, none on norms or embeddings. No dropout.
- **Batch:** ~0.5M tokens per optimizer step via gradient accumulation; micro-batch sized to fit 6 GB VRAM.
- **Hardware:** the RTX 4050 laptop GPU above, inside WSL2 Ubuntu.
- **Time budget:** pretraining <= 36 h, SFT <= 4 h, ablations <= 8 h of short runs (~100M tokens each).
- **Token budget:** run a throughput benchmark first (tokens/sec on the real model and batch setup). Then set the pretraining target to `min(3B, measured tokens/sec x pretraining time budget)`. Do not assume a token count before measuring.
- **Resumability:** save a checkpoint about every 30 minutes containing model, optimizer, scheduler, RNG state, and data position. A `--resume` flag must continue the run without breaking the loss curve.
- **Thermals:** log tokens/sec and GPU temperature. A sustained throughput drop signals laptop throttling and must be reported.

---

# Architecture Design

A decoder-only transformer using the modern Llama-style recipe, scaled down and tuned for ~29M parameters. Use only components with proven wins at small scale, spend parameters where they count (depth over width, small vocabulary, tied embeddings), and make every choice testable through ablations.

## 1. Overview

```
Token IDs
   │
Token Embedding (vocab × d_model)  ◄─── weights tied with output head
   │
   ▼
┌─────────────────────────────────────┐
│  Transformer Block  × 16            │
│                                     │
│   x ──► RMSNorm                     │
│         │                           │
│         ▼                           │
│   Grouped-Query Attention           │
│     • RoPE on Q and K               │
│     • QK-norm                       │
│     • causal, flash kernel          │
│         │                           │
│   x = x + attn_out    (residual)    │
│                                     │
│   x ──► RMSNorm                     │
│         │                           │
│         ▼                           │
│   SwiGLU Feed-Forward               │
│         │                           │
│   x = x + ffn_out     (residual)    │
└─────────────────────────────────────┘
   │
Final RMSNorm
   │
Output Head (tied embedding transpose)
   │
Logits over vocab
```

## 2. Configuration

| Setting | Value |
|---|---|
| Vocabulary | 8,192 (special tokens included) |
| d_model | 384 |
| Layers | 16 |
| Attention heads | 6 (head_dim = 64) |
| KV heads | 2 (GQA, 3 query heads per KV head) |
| FFN hidden dim | 1,024 (SwiGLU) |
| Context length | 1,024 |
| RoPE base (theta) | 10,000 |
| Biases | None |
| Embeddings | Tied |
| Dropout | 0 |

## 3. Parameter Budget

| Component | Calculation | Params |
|---|---|---|
| Q projection | 384 × 384 | 147,456 |
| K projection | 384 × 128 | 49,152 |
| V projection | 384 × 128 | 49,152 |
| Output projection | 384 × 384 | 147,456 |
| SwiGLU (gate, up, down) | 3 × 384 × 1,024 | 1,179,648 |
| Norms (incl. QK-norm) | small | ~900 |
| **Per layer** | | **~1.57M** |
| **16 layers** | | **~25.18M** |
| Token embedding (tied) | 8,192 × 384 | 3.15M |
| **Total** | | **~28.3M** |

The default is 16 layers (~28.3M). A 17th layer (~29.9M) is allowed only if the exact counted total stays at or below 30.0M. Report the exact count from the model, not this estimate.

## 4. Components

- **Pre-norm RMSNorm** before each sub-layer and before the output head.
- **RoPE** on Q and K (theta 10,000). No learned position embeddings.
- **GQA:** 6 query heads sharing 2 key/value heads.
- **QK-norm:** RMSNorm on queries and keys, per head, before the dot product.
- **SwiGLU FFN:** gate, up, and down matrices, hidden dim 1,024.
- **Tied embeddings:** the input embedding is reused, transposed, as the output projection.
- **No biases** in any linear layer.
- **Attention kernel:** `F.scaled_dot_product_attention` with the causal flag; no hand-written attention loop and no explicit mask tensor.
- **Initialization:** normal, std 0.02. Scale the residual output projections (attention output, FFN down) by 1/√(2 × layers).
- **Depth over width:** 16 layers at d=384 rather than 8 layers at d=512, following MobileLLM's findings for sub-billion models.

## 5. Tokenizer

- **Byte-level BPE**, vocab size **8,192**, including special tokens.
- Special tokens: `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|end|>`.
- Train the tokenizer on a mix representative of both pretraining and chat data.

## 6. Data Pipeline

### Stage 1: Pretraining
- Clean, educational web text (FineWeb-Edu or similar high-quality subset).
- Optionally mix in a small share of simple narrative or dialogue-style text for fluency.

### Stage 2: Supervised Fine-Tuning (Chat)
- A small, clean instruction/chat dataset (e.g., SmolTalk or similar), subject to the license and provenance rules above.
- Fixed chat template with explicit role markers.
- **Compute loss on assistant tokens only**; mask the user prompt.
- A few epochs at a much lower learning rate than pretraining.

### Optional Stage 3
- Light preference tuning (DPO). Skip unless time allows.

## 7. Inference

- **KV cache** so each new token costs one forward step instead of reprocessing the sequence.
- Sampling: temperature ~0.7, top-p ~0.9, mild repetition penalty.
- Stop on the `<|end|>` token or at max new tokens.

## 8. Evaluation Plan

1. **Validation loss / perplexity** on held-out pretraining data, tracked throughout.
2. **Benchmarks** suited to small models: HellaSwag, ARC-Easy, PIQA, LAMBADA. `lm-eval` needs a thin custom wrapper to load this model. Expect near-random on harder tasks.
3. **Qualitative chat samples** on a fixed prompt set, saved at each checkpoint so progress is visible.

### Ablation Table

Train short runs (~100M tokens each, same seed and token budget), changing one component at a time, and report validation loss:

| Variant | Val loss |
|---|---|
| Full model | baseline |
| LayerNorm instead of RMSNorm | |
| GELU MLP instead of SwiGLU | |
| Learned positions instead of RoPE | |
| MHA instead of GQA | |
| Without QK-norm | |
| Untied embeddings | |
| AdamW only instead of Muon + AdamW | |
| 8 layers wide vs 16 layers thin (at roughly equal params) | |

If a component shows no gain, report that honestly.

## 9. Realistic Expectations

A 29M model should produce fluent, grammatical English and follow the simple chat format for short exchanges. It will not hold broad factual knowledge, reason reliably, or sustain long coherent conversations. State these limits plainly in the README.

---

# Acceptance Criteria & Verification

Split into hard pass/fail checks (correctness) and calibrated targets (quality). The quality numbers are **provisional estimates, not measured values**. Lock the exact thresholds after the first baseline run.

## 1. Correctness (hard pass/fail)

| Check | Pass condition |
|---|---|
| Parameter count | Between 28.0M and 30.0M, counted from the model |
| Initial loss | About ln(8192) ≈ 9.01 at init |
| Causality | Changing a future token leaves all earlier logits unchanged |
| KV cache equivalence | Cached and full-forward logits match within bf16 tolerance |
| Weight tying | Input embedding and output head are the same tensor |
| Overfit test | A single tiny batch drives loss to near zero |
| Save/load roundtrip | Identical logits after reload |
| Resume | A resumed run continues the loss curve without a jump |
| Determinism | Same seed gives the same loss curve for the first N steps |
| Numerical health | No NaN/Inf in bf16 over a full run |

## 2. Training stability

- No loss spikes that fail to recover within about 200 steps.
- Gradient norm stays bounded with no sustained upward drift.
- Train/val gap is small (single-epoch pretraining should show almost none).
- Across 3 seeds, final val loss varies by roughly ±0.02 or less.

## 3. Pretraining quality (provisional)

Report **bits-per-byte** in addition to loss, since loss isn't comparable across tokenizers.

| Benchmark | Random | Provisional target |
|---|---|---|
| HellaSwag (acc_norm) | 25% | Roughly 27-30% |
| ARC-Easy | 25% | Roughly 35-42% |
| PIQA | 50% | Roughly 57-61% |
| LAMBADA | ~0% | Low, so track perplexity, not accuracy |

Criteria: beat random with a clear margin on at least three of these, and show monotonic improvement across checkpoints. Do not expect ARC-Challenge or MMLU to move.

## 4. Architecture justification

- Full model beats a plain GPT-2-style baseline on val loss at equal params and tokens.
- Each ablation row is filled in. A component that shows no gain is flagged honestly, not hidden.

## 5. Chat behavior (after SFT)

- **Template:** a test confirms loss is computed only on assistant tokens.
- **Termination:** at least 95% of responses end with `<|end|>` within the 256-token limit.
- **Fixed prompt set:** 50 prompts (greetings, simple Q&A, short instructions, small talk), scored on a rubric for grammaticality, relevance, and format-following. Target is a majority passing grammar and relevance; low factual accuracy is acceptable.
- **Repetition:** under 5% of responses contain degenerate loops at the default sampling settings.
- **Stability:** no crashes or garbage output on empty, very long, or non-English input, following the edge-case rules below.

## 6. Inference

- KV cache gives a measurable tokens/sec speedup over uncached generation.
- Report throughput and peak memory on the stated hardware.
- Behavior at the 1,024-token limit follows the edge-case rules below.

## 7. Repo and reproducibility

- One command trains from config, and one command runs eval.
- README includes the exact param count, hardware, training hours, and token count, plus loss curves, the ablation table, sample conversations (including failures), and the stated limitations.
- Fresh-clone test: a clean environment reproduces the smoke run.

---

# Input / Output format

### Input (chat template)
- **Special tokens:** `<|bos|>`, `<|user|>`, `<|assistant|>`, `<|end|>`
- **Conversation layout:** `<|bos|>` followed by role-tagged turns, each closed with `<|end|>`, and ending with the `<|assistant|>` tag so the model continues from there
- **Max input length:** 768 tokens (leaves room for up to 256 new tokens within the 1,024 total)

### Output
- **Type:** Plain text, generated token by token
- **Termination:** Stops on `<|end|>` or max new tokens
- **Max new tokens:** 256
- **Default sampling:** temperature ~0.7, top-p ~0.9, mild repetition penalty
- **Streaming:** Yes; the CLI prints tokens as they are generated

### Error / edge behavior
- **Empty input:** re-prompt the user; do not run generation and do not crash.
- **Input exceeding context:** drop the oldest turns whole until the history fits in 768 tokens, always keeping the latest user turn. If the latest user turn alone exceeds 768 tokens, return an error message and do not generate.
- **Non-English input:** no support is promised; generate normally without crashing. Not evaluated.
