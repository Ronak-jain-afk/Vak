# Vak

Vak is a 54.9M parameter decoder-only language model trained from scratch on consumer laptop hardware (NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM). It uses a modern LLaMA-style architecture with tied embeddings, trained on a 1.0 billion token multi-source corpus using a dual **Muon + AdamW** optimizer stack and aligned for conversational chat.

## Architecture

| Parameter | Value | Details |
|---|---|---|
| **Parameters** | 54,940,416 | Exactly counted from instantiated model |
| **Layers** | 18 | Transformer decoder blocks |
| **d_model** | 512 | Hidden representation dimension |
| **Attention Heads** | 8 | Head dimension 64 |
| **KV Heads** | 2 | Grouped-Query Attention (4:1 ratio) |
| **FFN Dimension** | 1,408 | SwiGLU feed-forward network |
| **Vocabulary Size** | 8,192 | Byte-level BPE with special tokens |
| **Context Length** | 1,024 | Maximum sequence length |
| **Positional Embedding**| RoPE | Rotary position embeddings ($\theta = 10,000$) |
| **Normalization** | RMSNorm | Pre-norm with $\epsilon = 10^{-5}$ and QK-norm |
| **Weight Tying** | Enabled | Input embedding tied to output projection head |

## Training & Scaling

The model was pretrained on **1,000,341,504 tokens** across a curated multi-source mixture:
* **FineWeb-Edu** (50%)
* **Cosmopedia v2** (25%)
* **Python-Edu** (12%)
* **OpenWebMath** (8%)
* **TinyStories** (5%)

Pretraining used quintic Newton-Schulz **Muon** on 2D weight matrices ($\text{lr} = 0.02$) and **AdamW** on 1D vectors, norms, and embeddings ($\text{lr} = 0.002$), reaching ~31.8k tokens/second under PyTorch `torch.compile` and BF16 SDPA.

### Scaling Study (28.3M vs 54.9M)
Holding the token budget (1.0B tokens), tokenizer, optimizer recipe, and dataset mixture constant, the architecture was scaled from 28.3M to 54.9M parameters:
* **Held-out validation loss:** 2.4570 $\to$ **2.3248** nats ($-12.4\%$ perplexity reduction: 11.67 $\to$ **10.22**).
* **Bits-per-byte:** 0.936 $\to$ **0.885 BPB**.
* Detailed comparison and telemetry: [`reports/scaling_55m_report.md`](reports/scaling_55m_report.md).

### SFT Conversational Alignment
The foundation model was fine-tuned on an expanded 10k-sequence multi-turn dialogue corpus (8,490 train, 1,509 validation) with zero prompt contamination against held-out benchmarks:
* **Format Termination:** 100.0% clean `<|end|>` stopping.
* **Repetition Rate:** 0.0% degenerate looping.
* **Held-out Semantic Accuracy:** 64.0% (32/50 on unseen benchmark prompts).
* Detailed alignment report: [`reports/sft_expansion_report.md`](reports/sft_expansion_report.md).

## Quickstart

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Tests
```bash
PYTHONPATH=. pytest -v
```

### Interactive Chat CLI
```bash
PYTHONPATH=. python scripts/chat.py --checkpoint checkpoints/run_55m/vak_chat_sft.pt
```

### Pretraining & Fine-Tuning
```bash
# Pretraining
PYTHONPATH=. python scripts/train_pretrain.py --config configs/config.yaml

# Resume pretraining
PYTHONPATH=. python scripts/train_pretrain.py --resume

# Supervised fine-tuning
PYTHONPATH=. python scripts/train_sft.py --config configs/config.yaml
```

### Evaluation
```bash
# Two-tier conversational and validation evaluation
PYTHONPATH=. python scripts/eval.py --checkpoint checkpoints/run_55m/vak_chat_sft.pt

# Zero-shot lm-evaluation-harness
PYTHONPATH=. python scripts/benchmark_lmeval.py --checkpoint checkpoints/run_55m/pretrain_step_001908.pt --tasks hellaswag,arc_easy,piqa --limit 100
```

## Known Limitations

At 55M parameters, the model demonstrates coherent conversational formatting, short Python syntax generation, and basic single-step arithmetic, but operates within clear capacity constraints:
1. **Parametric Memory:** It cannot store broad encyclopedic world knowledge. Unseen factual trivia will hallucinate or fall back to generic answers.
2. **Context Inertia:** Small models can experience attention carryover across rapid, unrelated multi-turn topic switches. Typing `clear` in the chat CLI resets the conversation buffer.
3. **Multi-Digit Arithmetic:** Without chain-of-thought scratchpad tokens, arithmetic beyond single-digit addition relies on surface pattern matching.

## License

Code and weights are released under the [Apache License 2.0](LICENSE).
Pretraining data provenance is documented in [`reports/dataset_report.md`](reports/dataset_report.md) and [`data/manifest.json`](data/manifest.json).
