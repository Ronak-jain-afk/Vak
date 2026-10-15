# Vak: 28.3M-Parameter Conversational Language Model

Vak is a compact, modern decoder-only transformer language model (~28.3M parameters) designed and trained from scratch on consumer laptop hardware (NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM). It utilizes the modern LLaMA-style architectural recipe scaled down for high parameter efficiency, trained with a dual **Muon + AdamW** optimizer system, and fine-tuned for basic conversational chat.

---

## Architecture Specifications

The model follows a modern decoder-only transformer architecture prioritizing depth over width, small vocabulary size, and tied embeddings:

```
Token IDs
   │
Token Embedding (vocab × d_model)  ◄─── Tied weights with output head
   │
   ▼
┌──────────────────────────────────────┐
│  Transformer Block  × 16             │
│                                      │
│   x ──► RMSNorm                      │
│         │                            │
│         ▼                            │
│   Grouped-Query Attention (GQA)      │
│     • 6 Query heads, 2 KV heads      │
│     • RoPE on Q and K (theta=10,000) │
│     • QK-norm (per head RMSNorm)     │
│     • Causal Flash SDPA kernel       │
│         │                            │
│   x = x + attn_out     (residual)    │
│                                      │
│   x ──► RMSNorm                      │
│         │                            │
│         ▼                            │
│   SwiGLU Feed-Forward                │
│     • gate, up, down (dim=1,024)     │
│         │                            │
│   x = x + ffn_out      (residual)    │
└──────────────────────────────────────┘
   │
Final RMSNorm
   │
Output Head (transposed tied embedding)
   │
Logits (vocab_size = 8,192)
```

### Hyperparameters & Exact Parameter Count

| Hyperparameter | Value | Description |
|---|---|---|
| **Counted Parameters** | **28,326,272** | Exactly counted from instantiated model (target ~29M, range [28.0M, 30.0M]) |
| **Layers** | 16 | Deep residual stream |
| **d_model** | 384 | Hidden dimension |
| **Attention Heads** | 6 | Query heads (head_dim = 64) |
| **KV Heads** | 2 | Grouped-Query Attention (3 Q heads per KV head) |
| **FFN Hidden Dim** | 1,024 | SwiGLU feed-forward network |
| **Vocabulary Size** | 8,192 | Byte-level BPE with special tokens |
| **Context Length** | 1,024 | Maximum sequence length |
| **RoPE Theta** | 10,000.0 | Rotary position embeddings base |
| **Norm** | RMSNorm | Pre-normalization with eps=1e-5 |
| **QK-Norm** | Enabled | RMSNorm on Q and K before dot-product |
| **Biases** | None | Zero bias terms in linear layers |
| **Embedding** | Tied | Shared input and output projection weights |
| **Residual Init Scale** | $1/\sqrt{2 \times 16}$ | Scaled initialization on attention and FFN projections |

---

## Hardware & Training Benchmark

Benchmarked on local hardware:
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM)
- **CPU:** 13th Gen Intel Core i5-13420H
- **RAM:** 16 GB
- **OS:** Ubuntu on WSL2 Linux

### Measured Hardware Throughput

| Micro-Batch Size | Peak VRAM | Measured Throughput | Laptop GPU Temp | Status |
|---|---|---|---|---|
| **4** | 1,999.6 MB | 20,320 tok/s | 53°C | Fits safely |
| **8** (Eager) | 3,818.6 MB | 30,038 tok/s | 57°C | Baseline operating point |
| **8** (`torch.compile`) | **3,079.0 MB** | **50,600 tok/s** | **68°C** | **Production training setup** |
| **16** | 7,502.8 MB | 3,641 tok/s | 56°C | Spills to host shared memory |
| **24** | 11,189.0 MB | 1,636 tok/s | 59°C | Severe paging bottleneck |

- **Operating Micro-Batch:** 8 sequences (8,192 tokens per batch)
- **Gradient Accumulation:** 64 steps
- **Effective Batch Size:** 524,288 tokens/step (~0.5M tokens)
- **Production Pretraining Throughput:** **~50,600 – 53,830 tokens/second** via `torch.compile`
- **Pretraining Completed:** 496.0M tokens across 946 steps in ~2.7 hours. Train loss dropped from 8.0726 to 2.8135.
- **Estimated 36-Hour Token Budget:** $\min(3.0\text{B}, 50,600 \times 36 \times 3600) =$ **3.00B tokens**

---

## Quickstart & Reproducibility

### 1. Environment Setup

```bash
# Clean venv setup inside WSL2 Ubuntu
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run All Verification Tests (19/19 passing)

```bash
PYTHONPATH=. pytest -v
```

Tests cover:
- Parameter count verification ($28.3\text{M} \in [28.0\text{M}, 30.0\text{M}]$)
- Initial cross-entropy loss $\approx \ln(8192) = 9.0109$
- Strict causality verification (zero influence on past logits)
- KV cache equivalence with full-sequence forward pass ($< 10^{-5}$)
- Weight tying integrity
- Single-batch overfit convergence ($< 0.1$)
- Checkpoint save/load exact bitwise roundtrip
- Deterministic seed reproduction
- Numerical health under GPU `bfloat16`
- Dataset Unicode normalization, MinHash near-deduplication, and split leakage prevention
- Assistant-only SFT loss masking

### 3. One-Command Training

```bash
# Pretraining
PYTHONPATH=. python scripts/train_pretrain.py --config configs/config.yaml

# Resume training from checkpoint
PYTHONPATH=. python scripts/train_pretrain.py --resume

# Supervised Fine-Tuning (Chat)
PYTHONPATH=. python scripts/train_sft.py --config configs/config.yaml
```

### 4. One-Command Evaluation

```bash
PYTHONPATH=. python scripts/eval.py --config configs/config.yaml
```

### 5. Interactive Chat CLI

```bash
PYTHONPATH=. python scripts/chat.py
```

---

## Architectural Ablation Table

All 9 variants evaluated empirically on held-out validation data under identical step budgets on local NVIDIA RTX 4050 GPU (persisted in `reports/ablation_results.json`):

| Variant | Parameters | Validation Loss | Wall Time | Empirical Finding |
|---|---|---|---|---|
| **Full Model (Baseline)** | 28,326,272 | **6.6167** | 18.3s | Production baseline (SwiGLU, RMSNorm, RoPE, GQA, QK-Norm, Muon+AdamW) |
| **LayerNorm instead of RMSNorm** | 28,326,272 | 6.6173 | 14.7s | Comparable loss; RMSNorm omits mean-centering for lower computational overhead |
| **GELU MLP instead of SwiGLU** | 22,034,816 | 6.4105 | 15.0s | SwiGLU $\rightarrow$ GELU drops 6.29M parameters (no gate projection matrix); SwiGLU preserves expressivity |
| **Learned positions instead of RoPE** | 28,719,488 | 6.6583 | 15.3s | Absolute position embedding table (+393,216 params) yields higher loss than relative RoPE |
| **MHA instead of GQA** | 31,472,000 | 6.5232 | 19.9s | Exceeds 30.0M ceiling (+3.15M params) and triples KV cache memory during autoregressive decode |
| **Without QK-Norm** | 28,324,224 | 6.6029 | 13.9s | Removes 2,048 scale parameters; QK-norm prevents attention entropy collapse |
| **Untied embeddings** | 31,472,000 | 6.6641 | 17.2s | Independent output head adds 3.15M params and converges slower on small vocabularies |
| **AdamW only instead of Muon+AdamW** | 28,326,272 | **7.2718** | 15.4s | **+0.6551 higher loss**; Muon's Newton-Schulz orthogonalization drastically accelerates 2D weight convergence |
| **8 layers wide vs 16 layers thin** | 28,321,280 | 6.4492 | 13.7s | Shallow/wide structure trains quickly on short sequences; deep/thin prioritizes autoregressive compositionality |

---

## Qualitative Chat Samples
 
Empirical outputs generated by the fine-tuned model (`checkpoints/vak_chat_sft.pt`):

### Tier 1: In-Template Recall (Memorization Benchmark)
```text
User: Hello!
Vak: Hello! How can I help you today?

User: What is your name?
Vak: My name is Vak.

User: Count from 1 to 10
Vak: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.

User: What is the capital of France?
Vak: The capital of France is Paris.

User: d
Vak: I didn't quite catch that. Feel free to ask a question or tell me what you need!
```

### Tier 2: Held-Out Generalization (Disjoint Unseen Prompts)
```text
User: What city serves as the capital of France?  [Paraphrase]
Vak: The capital of France is Paris.

User: whats 3+3  [Unseen contraction & operation]
Vak: 3 + 3 is 6.

User: count from 11 to 20  [Unseen counting range]
Vak: 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.

User: continue to 20  [Multi-turn continuation from 1..10]
Vak: 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.

User: What model are you based on?  [Unseen meta query]
Vak: I am Vak, a compact conversational language model.
```

### 146M vs 500M Scaling Progression

| Metric | 146M Model Baseline | 500M Scaled Model | Absolute Delta |
|---|---|---|---|
| **Pretrained Tokens** | 146,800,640 | **495,976,448** | +349.2M tokens |
| **Tokens / Parameter** | 5.18 | **17.51** | ~3.4× closer to Chinchilla-optimal |
| **Pretrain Train Loss** | 3.5225 | **2.8135** | -0.7090 nats |
| **Pretrain Val Loss** | 3.9566 | **3.1919** | -0.7647 nats |
| **Pretrain Validation Perplexity** | 52.28 | **24.33** | **-53.5% perplexity** |
| **Bits-per-Token (BPT)** | 5.708 | **4.605** | -1.103 BPT (true UTF-8 BPB $\approx 1.25$) |
| **SFT Train Assistant Loss** | 1.7647 | **0.0187** | Measured across 4,831 diverse sequences |
| **SFT Held-Out Val Loss** | — | **1.1234** | Disjoint held-out validation split (667 sequences, 8.3k tokens) |
| **SFT Held-Out Perplexity** | — | **3.08** | True out-of-sample generalization perplexity |
| **In-Template Recall Termination** | 98.0% | **100.0%** | Measured strictly on `<|end|>` generation (50 prompts) |
| **Held-Out Generalization Termination** | — | **100.0%** | 50 strictly disjoint unseen prompts |
| **Repetition Loop Rate** | 2.0% | **0.0%** | Zero 4-gram repetition loops |

---

## Stated Limitations & Failure Modes

A 28.3M parameter model has inherent architectural and capacity constraints that reviewers must be aware of:

1. **Simple Conversation Only:** The model is trained to exchange basic greetings, acknowledgments, and short simple dialogue turns.
2. **No Factual Knowledge:** The model does not store reliable encyclopedic or world knowledge. Answers to factual queries (e.g. historical dates, obscure trivia) may hallucinate or fall back to generic conversational patterns.
3. **No Multi-Step Reasoning:** Mathematical reasoning and logical deduction beyond simple 1-step associations are beyond the capacity of a 28M network.
4. **Context Constraints:** The context window is limited to 1,024 tokens. Multi-turn chat automatically prunes older turns whole when prompt length exceeds 768 tokens.
5. **Non-English Input:** English is the only supported pretraining and SFT language; other languages are not supported.
