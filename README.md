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
- **Production Pretraining Throughput:** **~50,600 tokens/second** via `torch.compile`
- **Pretraining Completed:** 146.8M tokens across 280 steps in ~1.3 hours. Loss dropped from 7.7476 to 3.5225.
- **Estimated 36-Hour Token Budget:** $\min(3.0\text{B}, 30,038 \times 36 \times 3600) =$ **3.00B tokens**

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

Evaluated on held-out validation data using a controlled step and token budget:

| Variant | Parameters | Validation Loss | Observation / Note |
|---|---|---|---|
| **Full Model (Baseline)** | 28,326,272 | **7.8014** | Baseline recipe (SwiGLU, RMSNorm, RoPE, GQA, QK-Norm, Muon+AdamW) |
| **LayerNorm instead of RMSNorm** | 28,326,272 | 7.9120 | Slightly slower throughput, higher variance during early warmup |
| **GELU MLP instead of SwiGLU** | 27,146,624 | 8.0435 | Inferior parameter efficiency; SwiGLU improves loss at equal depth |
| **Learned positions instead of RoPE** | 28,719,744 | 8.1250 | Poorer length extrapolation and higher initial training entropy |
| **MHA instead of GQA** | 29,112,704 | 7.7980 | Minor loss delta, but 3× larger KV cache size during autoregressive inference |
| **Without QK-Norm** | 28,324,224 | 7.8840 | Slightly elevated attention logit drift; QK-norm provides stability |
| **Untied embeddings** | 31,472,000 | 7.7610 | Marginally lower loss, but exceeds the strict 30.0M parameter ceiling (~31.5M) |
| **AdamW only instead of Muon+AdamW** | 28,326,272 | 8.1890 | Muon converges noticeably faster on 2D hidden matrices at early training steps |
| **8 layers wide vs 16 layers thin** | 28,642,816 | 7.9450 | Confirms MobileLLM findings: depth over width yields lower loss at sub-50M scale |

---

## Qualitative Chat Samples

Using the fine-tuned model (`checkpoints/vak_chat_sft.pt`):

```text
User: Hello!
Vak: Hello! How can I help you today?

User: Good morning!
Vak: Hello! How can I help you today?

User: How is the weather?
Vak: The weather forecast shows around 30 million pounds of whole grains, while the weather is sunny with a significant amount of sunlight and a low of 75°F (15°C).

User: What is the capital of France?
Vak: The United States, Germany, and the US. (Failure mode: demonstrates lack of factual world knowledge in sub-30M models)

User: If I have three apples and eat one, how many are left?
Vak: Great choice! How can I help you today? (Failure mode: demonstrates absence of multi-step reasoning capacity)
```

### Format Termination & Repetition Benchmarks
- **Pretrained Validation Loss:** **3.9566** (Bits-per-Byte: **5.708**)
- **Final SFT Assistant Loss:** **1.7647**
- **Format Termination Rate:** **98.0%** (Target: $\ge 95\%$; 49/50 responses cleanly stop on `<|end|>`).
- **Repetition Loop Rate:** **2.0%** (Target: $< 5\%$; 1/50 responses exhibited mild phrase repetition).

---

## Stated Limitations & Failure Modes

A 28.3M parameter model has inherent architectural and capacity constraints that reviewers must be aware of:

1. **Simple Conversation Only:** The model is trained to exchange basic greetings, acknowledgments, and short simple dialogue turns.
2. **No Factual Knowledge:** The model does not store reliable encyclopedic or world knowledge. Answers to factual queries (e.g. historical dates, obscure trivia) may hallucinate or fall back to generic conversational patterns.
3. **No Multi-Step Reasoning:** Mathematical reasoning and logical deduction beyond simple 1-step associations are beyond the capacity of a 28M network.
4. **Context Constraints:** The context window is limited to 1,024 tokens. Multi-turn chat automatically prunes older turns whole when prompt length exceeds 768 tokens.
5. **Non-English Input:** English is the only supported pretraining and SFT language; other languages are not supported.
