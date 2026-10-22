# Spark-X2.6-4B on 8 GB VRAM — a 4B agentic model with a cheap long context

**Date:** 2026-10-21 · **Branch:** `spark_on_8gb_vram` (from `main`) · **Status:** works — 50 tok/s, 108K context, tool calls + pi; real-task trial pending
**Companion:** [`qwen-bakeoff.md`](qwen-bakeoff.md), [`dory-on-8gb.md`](dory-on-8gb.md) — same machine, same scripts.

## Introduction

[XHToken/Spark-X2.6-4B](https://huggingface.co/XHToken/Spark-X2.6-4B) is a dense 4.1B text
model (Apache-2.0) aimed at on-device agents. It is a post-training refresh of Spark-X2.5-4B
with the same architecture, `Spark2_5ForCausalLM`:

- 36 layers, hidden 2560, GQA 16 query / 4 KV heads, head_dim 256, GeGLU MLP, tied 131k vocab.
- Three sliding-window layers (512 tokens, full RoPE, theta 10k) per full-attention layer
  (partial RoPE 25 %, theta 5M). Only those **9 full-attention layers keep a long KV cache**.
- A per-head sigmoid gate on the attention output.
- Thinking on by default, with a `reasoning_effort` of `high` or `low` in the chat template
  (`enable_thinking: false` turns it off). The card claims a 1M context.

Support for `spark2_5` landed in upstream llama.cpp on 2026-09-06 (ggml-org/llama.cpp#27868).
`vendor/dory-llama.cpp` is upstream master of 2026-10-15 plus the Dory patch, so the existing
Dory build runs it unchanged. `vendor/llama.cpp` (June) is too old.

## Setup

```bash
bash scripts/download-qwen-bakeoff.sh spark   # official Q8_0 GGUF, 4.07 GiB, pinned + sha256-checked
bash scripts/start_spark.sh                   # server (Q8_0, 108K, q8_0 KV) + pi
```

Wired in as `QMODEL=spark` in `run-qwen.sh` / `start-qwen.sh`, like dory and qwen3vl.
Sampling follows the card: temp 1.0, top-p 0.95, top-k off.

## Why 108K and not 128K

The plan was Q8_0 weights + 128K at a q8_0 KV. llama.cpp's own breakdown at 128K:

| | MiB |
|---|---|
| weights (Q8_0, all 37 layers on the GPU) | 4167 |
| KV, 9 full layers × 128K, q8_0 | 2448 |
| KV, 27 SWA layers × 1024 cells | 57 |
| compute buffer (ubatch 512) | 697 |
| **total** | **7370** — free next to the desktop: ~7205 → OOM |

The compute buffer grows with the context: 218 MiB at 32K, 617 at 112K, 697 at 128K, a
linear fit of ~58 MiB + **~5 KiB per context token**. That is one layer's K+V in f16
(1024 × 2 B × 2): with a quantized KV, flash attention works on an f16 copy of a layer's
cache. So a q8_0 KV really costs ~24 KiB/token here, not 19. `UBATCH=256` saves only 92 MiB
and still OOMs at 128K.

Solved for ~0.3 GB of headroom, that gives **110592 (108K)** as the default: 7.4 GB used on the
8 GB card, with no OOM through a 105K-token prompt. Full 128K needs smaller weights (official
Q4_K_M, 3.3 GiB, saves ~0.8 GB; not tested).

## Measurements (RTX 2070 Max-Q, single runs)

| test | result |
|---|---|
| generation, short coding prompt, thinking on (`qwen-bench.py`) | **50.4 tok/s** |
| tool call (`read` offered) | well-formed call, `finish=tool_calls` |
| prefill, 16.8K tokens | 1619 tok/s (10.4 s) |
| needle at 25 % depth in a **104,905-token** prompt, thinking off | **found** (`7413-KESTREL`) |
| ↳ prefill / generation at that depth | 791 tok/s (133 s wall) / 19.1 tok/s |
| short question, `reasoning_effort: low` | `finish=stop`, "Hefei", 427 reasoning chars |
| pi one-shot via `start_spark.sh` (read tool + answer) | correct |

Reasoning comes back in `reasoning_content`, separate from the answer. The server log line
`added <｜end▁of▁sentence｜> logit bias = -inf` concerns warmup only; real requests stop normally.

## Open

- A real agentic task trial in pi, as for the other models.
- More than one needle depth; one depth at 105K is a smoke test, not a recall curve.
- Q4_K_M at full 128K (and its quality vs Q8_0).
