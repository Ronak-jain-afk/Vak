# Vak — Phase 2 Plan: Model Scaling to ~55M Parameters

**Current state (Phase 1, complete):** a 28,326,272-parameter decoder-only model, pretrained on 1,000,341,504 tokens, SFT-aligned and evaluated, merged via PR #8. The phase's artifacts are the comparison baseline: `checkpoints/pretrain_step_001908.pt`, `checkpoints/vak_chat_sft.pt`, `logs/pretrain_1b.log`, `reports/scaling_1b_report.md`, `reports/eval_pretrain_1b_base.json`, `reports/lmeval_pretrain_1b_base.json`, `reports/eval_results.json`.

**This phase in one line:** keep everything that made the 1B run comparable — same corpus (1.0B tokens), same tokenizer, same training recipe, same evaluation protocol, same token budget — and scale only the model from ~28.3M to ~54.9M parameters (~1.94x). This is the project's first clean model-size comparison at fixed data.

**Scope note:** this plan supersedes the Phase 1 brief's 30M parameter ceiling, per explicit decision. All Phase 1 constraints not restated here (hardware envelope, dataset provenance rules, engineering style, thermals logging rule) still apply.

---

# 1. Why scale now, and why ~55M

- The 28.3M model trained at **35.3 tokens per parameter** — deep past the ~20 tokens/parameter Chinchilla balance. The model, not the data, is the scarce resource.
- With the corpus fixed at 1.0B tokens, the balanced model size is ~50M. Past ~70M the corpus itself becomes the bottleneck (fewer than ~14 tokens/parameter), so ~55M is the sweet spot for this phase.
- Token scaling has been done three times (146M → 500M → 1.0B tokens on the same backbone); model scaling has never been tested. This phase adds that axis.
- Expected result: a real but modest quality step — **ballpark 5–15% lower held-out perplexity**. This is an extrapolation, not a promise; whatever the result, it is reported honestly.

# 2. Target configuration (old → new)

| Setting | Phase 1 | Phase 2 (target) | Rationale |
|---|---|---|---|
| Vocabulary | 8,192 | 8,192 | unchanged — tokenizer and corpus stay valid |
| d_model | 384 | 512 | wider residual stream |
| Layers | 16 | 18 | modestly deeper; keeps the deep-narrow bias |
| Attention heads | 6 | 8 (head_dim 64) | head_dim held constant |
| KV heads (GQA) | 2 (3:1) | 2 (4:1) | KV cache stays small |
| FFN hidden dim (SwiGLU) | 1,024 | 1,408 (~2.75x) | standard SwiGLU ratio |
| Context length | 1,024 | 1,024 | unchanged |
| RoPE theta | 10,000 | 10,000 | unchanged |
| Embeddings | tied | tied | unchanged; still only ~7.6% of params |
| QK-norm | on | on | unchanged |
| Exact parameter count | 28,326,272 | 54,940,416 (verify from the instantiated model) | |

Expected implementation surface: the model section of `configs/config.yaml` (five values) plus a new checkpoint directory. No source changes are expected; if any script turns out to hardcode a dimension, the fix is minimal and belongs to this phase.

# 3. Parameter budget (target, exact by construction)

| Component | Calculation | Params |
|---|---|---|
| Q projection | 512 x 512 | 262,144 |
| K projection | 512 x 128 | 65,536 |
| V projection | 512 x 128 | 65,536 |
| Output projection | 512 x 512 | 262,144 |
| SwiGLU (gate, up, down) | 3 x 512 x 1,408 | 2,162,688 |
| Norms (incl. QK-norm) | small | 1,152 |
| **Per layer** | | **2,819,200** |
| **18 layers** | | **50,745,600** |
| Token embedding (tied) | 8,192 x 512 | 4,194,304 |
| Final norm | | 512 |
| **Total** | | **54,940,416** |

The exact count from `count_parameters()` at instantiation is authoritative. Phase tolerance: 54.0M–56.0M.

# 4. Fixed for comparability

Only the model changes. Everything else stays byte-identical to Phase 1:

- **Corpus:** the same tokenized shards, same train/val/test split, same 1.0B-token budget. No re-tokenization.
- **Batch shape:** micro-batch 8 x seq 1,024 x 64 accumulation steps = 524,288 tokens per optimizer step.
- **Step budget:** 1,908 steps = 1,000,341,504 tokens — the same token count Phase 1 trained on.
- **Recipe:** Muon lr 0.02, AdamW lr 0.002, cosine decay to 10%, 100-step warmup, grad clip 1.0, weight decay 0.1, bf16 + `torch.compile`, no dropout.
- **Checkpointing:** every 100 steps plus wall-clock saves; `--resume` with the 100-step rewarm (already tested).
- **Evaluation protocol:** same val split; `scripts/eval.py` (with `--output`) for loss/perplexity/BPT/BPB; `scripts/benchmark_lmeval.py --limit 100 --num_fewshot 0` on HellaSwag / ARC-Easy / PIQA; the same held-out chat/loss prompts; the existing chat CLI and its default sampling settings.
- **SFT:** same chat data, same recipe (3 epochs, assistant-only loss, per-epoch saves).

# 5. Execution phases

**Phase 0 — Preflight (~100–150 steps, ~35–55 min projected)**
- Purpose: prove stability and measure real throughput, VRAM, and temperature at the new size before committing to the full run.
- Gates: initial loss near ln(8192) ≈ 9.01; no NaN/Inf and no unrecovered spikes; measured throughput within ±20% of the ~26–27k tok/s projection; VRAM well under 2 GB; temperatures stable; loss curve smooth.
- If instability appears (spikes, grad-norm drift), run a short Muon lr check (0.015 / 0.02 / 0.025) before continuing — no lr change is planned by default.
- Also confirms all scripts consume the new config (no hardcoded dimensions).
- Deliverable: measured numbers that lock the Phase 1 time estimate.

**Phase 1 — Pretraining**

Full 1,908 steps. Projected **~11–13 h wall clock** (Phase 1 took ~6.1 h; time is not a constraint). Uses the existing checkpoint/resume machinery. Log kept at `logs/pretrain_55m.log`.

**Phase 2 — Base evaluation (identical protocol)**

- Deterministic metrics from the final checkpoint (val loss, perplexity, BPT, BPB) — minutes.
- lm-eval (HellaSwag / ARC-Easy / PIQA, limit 100, 0-shot) — minutes.
- Generation-based sections of the base eval are slow in the current path (~3 tok/s observed; roughly 90 s per sample at 28.3M, so budget **~3–6 h** at 55M). Acceptable for this phase; inference-path speed is a separate future item.

**Phase 3 — SFT**

Same data and recipe; 3 epochs; assistant-only loss masking; per-epoch saves. Reads the Phase-2 base checkpoint from the new run directory (config-driven; Phase-1 files untouched). Projected ~1–2 h.

**Phase 4 — Chat evaluation**

The fixed 50-prompt chat eval to a JSON artifact, plus a live smoke test on the standard prompts (greeting, identity, capital, bees, counting, story). Verify termination / repetition / semantic rates the same way Phase 1's were checked.

**Phase 5 — Report, README, PR**

- New report: `reports/scaling_55m_report.md` — "model scaling at fixed 1.0B tokens": Phase 1 vs Phase 2 table (loss/ppl/BPT/BPB, lm-eval, SFT and chat metrics), exact parameter counts, wall clock and throughput, full artifact traceability, honest uncertainty notes (generation metrics are stochastic; ±5 pp on 100-sample benchmarks).
- Refresh the README front page to the final model (it is still 500M-era; flagged in the PR #8 review).
- Ship as a PR with the same review discipline as #8: every number traceable to a JSON or log artifact, no unsupported claims, corrections applied before merge.

# 6. Artifacts and naming

| Artifact | Path |
|---|---|
| Pretrain checkpoints | `checkpoints/run_55m/` (new run directory) |
| SFT checkpoint | `checkpoints/run_55m/vak_chat_sft.pt` |
| Training log | `logs/pretrain_55m.log` |
| Base eval JSON | `reports/eval_pretrain_55m_base.json` |
| lm-eval JSON | `reports/lmeval_pretrain_55m_base.json` |
| SFT/chat eval JSON | `reports/eval_results_55m.json` |
| Report | `reports/scaling_55m_report.md` |

Rule: no Phase-2 run may overwrite a Phase-1 artifact — Phase 1 is the comparison baseline and stays intact (`checkpoints/` top level keeps the 28.3M checkpoints).

# 7. Baseline to beat (Phase 1 artifacts)

| Metric | Phase 1 value | Source |
|---|---|---|
| Held-out val loss / ppl | 2.4570 / 11.67 | `reports/eval_pretrain_1b_base.json` |
| BPT / BPB | 3.545 / 0.936 | same |
| HellaSwag / ARC-Easy / PIQA (acc_norm, n=100) | 44% / 39% / 56% | `reports/lmeval_pretrain_1b_base.json` |
| SFT val assistant loss | 0.1128 | `reports/eval_results.json` |
| Chat rates (termination / repetition / semantic) | 100% / 0% / 50% | same (stochastic, n=50) |

Success on deterministic metrics means strictly lower loss/perplexity on the identical split. Generation metrics are compared loosely and reported as stochastic.

# 8. Risks and mitigations

- **Throughput worse than projected:** Phase 0 measures it first; worst case ~14 h is still acceptable.
- **Muon lr not ideal at width 512:** Muon's lr is width-robust (no change planned); a short 3-point sweep is the fallback if the preflight shows instability.
- **Long-run interruption:** resume path is already tested; 15-minute checkpoint cadence limits loss.
- **Thermal drift over a much longer run:** tokens/sec and temperature are logged; any sustained throughput drop must be reported (Phase 1 rule).
- **Smaller quality gain than the ballpark:** reported as-is. The next lever would be growing the corpus, which is out of scope here.

# 9. Out of scope

Corpus growth beyond 1.0B tokens; tokenizer or vocabulary changes; context beyond 1,024; architecture experiments (MoE, alternative attention, untied embeddings); inference-path optimization; scaling past ~55M.

# 10. Definition of done

- Exact parameter count reported from the instantiated model (target 54,940,416; tolerance 54.0M–56.0M).
- Pretraining completed at 1,908 steps with a healthy, resumable run.
- Deterministic eval metrics improved vs Phase 1 on the identical split (goal; honest reporting if not).
- lm-eval completed on the identical protocol; any regression surfaced, not hidden.
- SFT/chat criteria hold: assistant-only masking verified, ≥95% termination, <5% degenerate loops.
- Report and README updated; every number traceable to artifacts; merged via a reviewed PR.
