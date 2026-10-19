# Nemotron-Labs Dory1.0-4B on 8 GB VRAM — a looped model you can turn up or down

**Date:** 2026-10-18 · **Branch:** `dory_on_8gb_vram` (from `qwen3vl_on_8gb_vram`) · **Status:** works — 29 tok/s at 2 loops, 64K context, tool calls + pi; real-task trial pending
**Companion:** [`qwen-bakeoff.md`](qwen-bakeoff.md), [`bonsai2-on-8gb.md`](bonsai2-on-8gb.md), [`qwen3vl-on-8gb.md`](qwen3vl-on-8gb.md) — same machine, same scripts.

## Introduction

[NVIDIA-Nemotron-Labs-Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)
is a 4B text model NVIDIA released on 2026-10-16 under the OpenMDW-1.1 license. It is a
*looped* (latent-recurrent) normalized-GPT: its stack is split into an input block (8 layers),
a recurrent block (20 layers) and an output block (8 layers), and the recurrent block is run
several times per token, with the input re-injected into every loop. Attention alternates
between sliding-window (1024 tokens) and full-attention layers, so most layers keep only a
short KV cache. It was pre-trained on 6T tokens (most of them natively in NVFP4) and supports
up to 256K context.

What is special is that one checkpoint serves several compute budgets. The loop count (1–4) is
a runtime setting, not a different model: more loops spend more compute per token and, per
the card, raise accuracy and reduce how many tokens the model needs to solve a task. NVIDIA's
own numbers put 2 loops at SWE-bench Verified 59.5 and 4 loops at 61.8, against 54.7 for
Qwen3.5-4B. The card also offers a *last-loop* KV-cache mode that keeps one cache per recurrent
layer instead of one per loop, so more loops don't cost more KV memory.

Size is not the problem: the Q5_K_M GGUF is 3.67 GB. The problems are that no released
llama.cpp can load it yet, and that the obvious launch settings waste most of the card. It
needs the open llama.cpp PR (ggml-org/llama.cpp#30306), which this repo ships as a patch
(`patches/llama.cpp-dory-pr30306.patch`) and builds with CUDA. The card's
example (a Mac) turns flash attention off, which costs a 1.1 GB compute buffer. And this
build counts every loop as extra layers, so the usual `-ngl 99` silently leaves weights on the
CPU: 3.8 tok/s at 4 loops instead of 19.

What made it fit well: `-ngl 999` (every layer on the GPU), flash attention on, and a q8_0 KV
cache. Measured against the card's settings, flash attention and q8_0 change the model's
predictions about 13× less than dropping from 2 loops to 1. That leaves room for 64K context
at 5.7 GB, or 128K with the last-loop cache.

The trade-offs: speed falls with every loop (43 / 31 / 22 / 19 tok/s for 1–4), it is
English-only and text-only, and the llama.cpp support is an unmerged PR. See
"Measurements" and "Quality check".

## TL;DR

- [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF),
  `Q5_K_M` (3.67 GB), on stock llama.cpp `6e66cb5` + the PR #30306 patch kept in this repo
  (`patches/`), built into `vendor/dory-llama.cpp`.
- **2 loops (default): 29.3 tok/s**, 15.8K-token prefill at 852 tok/s, 5.7 GB VRAM at 64K.
  **4 loops: 17.7 tok/s**, 492 tok/s prefill, 5.1 GB.
- Tool calls ✓ (qwen-bench and pi), thinking on by default (`reasoning_content`).
- **`-ngl 999`, not 99.** This build's layer count includes the loops; at 4 loops `-ngl 99` kept
  2.9 GB of weights and most of the KV cache on the CPU.
- **Flash attention on + q8_0 KV**, unlike the card's Mac example: compute buffer 1,140 → 93–364
  MiB, KL divergence 0.007 vs the card's settings.
- `LOOPS=3/4` switch to the last-loop KV cache automatically; per-loop caches for 4 loops don't
  fit 32K, let alone 64K.

## Run it

Same toolchain as the Qwen bake-off (see
[`qwen-bakeoff.md`](qwen-bakeoff.md#run-these-models-on-your-own-8-gb-machine)), plus a
patched llama.cpp:

```bash
# Stock llama.cpp at the PR's base commit + the Dory patch from this repo, built with the
# same locked CUDA 12.2 toolchain into vendor/dory-llama.cpp (~15 min)
CUDA_VER=12.2 LLAMA_REF=6e66cb5d4e9a2b4f02820a9f3053422b92ec4d51 \
  PATCH=patches/llama.cpp-dory-pr30306.patch SRC_DIR=$PWD/vendor/dory-llama.cpp \
  bash scripts/build-llama-cuda.sh

bash scripts/download-qwen-bakeoff.sh dory          # 3.67 GB, pinned revision + sha256
bash scripts/start_dory.sh                          # server + pi (2 loops, 64K)
LOOPS=4 bash scripts/start_dory.sh                  # more accurate, ~40% slower
bash scripts/start_dory.sh -p "explain @README.md" --no-session   # one-shot

QMODEL=dory bash scripts/run-qwen.sh                # server only, OpenAI API on :8080
```

`start_dory.sh` needs no environment variables. Like `start_bonsai2.sh` it hides
shepherd-pi's tools and refuses to start if another model is serving on the port.

Knobs (`run-qwen.sh`): `LOOPS` 1–4 (default 2), `LOOP_KV` `per_loop` | `last_loop` (default
per_loop for 1–2 loops, last_loop for 3–4), `FA` on | off, plus the usual `CTX`, `KVQUANT`, `NGL`.
`CTX=131072 LOOP_KV=last_loop` gives 128K context at 2 loops (6.1 GB, 30.1 tok/s).

## The llama.cpp patch

No llama.cpp release supports Dory yet. NVIDIA's pull request
[ggml-org/llama.cpp#30306](https://github.com/ggml-org/llama.cpp/pull/30306) adds it, and this
repo keeps that PR as one patch file, so the build doesn't depend on the PR or its author's
fork staying as they are:

- **`patches/llama.cpp-dory-pr30306.patch`** — `git diff` from stock llama.cpp `6e66cb5`
  (master, 2026-10-15, the commit the PR last merged) to the PR head `43e865e`. 17 files,
  +507/−9 lines, 48 KB. MIT-licensed like llama.cpp; the file's header carries the notice.
- `build-llama-cuda.sh` fetches stock llama.cpp at `LLAMA_REF` and `git apply`s `PATCH` on top
  (skipped if already applied; anything that doesn't apply cleanly stops the build).
- Checked: applied to `6e66cb5`, the patch gives a source tree identical to `43e865e`, and a
  from-scratch build this way serves the model at the same speed as the PR build (see
  "Measurements").

What the patch adds:

- **A `dory` architecture** (`src/models/dory.cpp`, ~200 lines, plus registration in
  `llama-arch`, `llama-model`, `models.h`). The GGUF stores the 72 *physical* entries of the
  stack — attention and MLP are separate entries, so 8 + 20 + 8 blocks are 16 + 40 + 16 — and
  the loader expands them into **16 + 40 × loops + 16 *virtual* layers** that point at the same
  weights: 72 at 1 loop, 112 at 2, 192 at 4. That expansion is why `-ngl` must cover the
  virtual count (`999`), not the physical one.
- **The loop and nGPT math in the graph:** the input of the recurrent block is added back in
  at the start of every loop, every sublayer output and residual is L2-normalized and mixed by
  a learned per-channel step (`alpha`), and attention has a learned sigmoid output gate and
  a learned per-channel query scale. Sliding-window and full-attention layers use different RoPE bases.
- **Loop count and KV-cache mode as GGUF metadata** (`dory.num_loops`, 1–4;
  `dory.recurrent_kv_cache_mode`, `per_loop` | `last_loop`), so they can be changed at launch
  with `--override-kv` — that is what `LOOPS` / `LOOP_KV` do. `last_loop` maps every loop's
  attention layer to one cache slot, so each loop overwrites the previous one's KV.
- Small changes elsewhere: a bigger graph-node budget for the expanded layer count
  (`llama-context.cpp`), context shifting disabled for Dory (`llama-kv-cache.cpp`), the
  quantizer counting physical rather than virtual FFNs (`llama-quant.cpp`), the HF → GGUF
  converter (`conversion/dory.py`, `gguf-py`), and tests.

Why the PR head and not the card's `98a74c9`: the head is the version under review, merged
with current llama.cpp. The card's older revision also changed two CUDA matmul paths to stay
in full FP32 instead of TF32; the PR later moved that to a separate change. TF32 exists only on
Ampere and newer, so on this Turing card it makes no difference.

To build from the PR itself instead (same source tree): GitHub serves the PR's commits from
the main repository, so `LLAMA_REF=43e865ec059a35d57f1167eaea76ca187077f063` with the default
`LLAMA_REPO` and no `PATCH` works too.

### Exact files

| File | Size | sha256 |
|---|---:|---|
| `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (repo rev `0963c80`) | 3,673,288,864 B | `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0` |

## Measurements

RTX 2070 Max-Q (8 GB, sm_75), CUDA 12.2. Generation speed from llama-server's own timings;
"VRAM" is `nvidia-smi` used memory (desktop included) or llama.cpp's own breakdown where
noted. `qwen-bench.py` = 1500-token coding prompt with thinking, one tool call, 15.8K-token
prefill.

### The `-ngl 99` trap

The first runs used `-ngl 99` like every other dense model here, at the card's settings
(flash attention off, f16 KV, per-loop cache, 32K):

| Loops | `-ngl 99` | `-ngl 999` |
|---|---:|---:|
| 1 | 36.0 tok/s | 42.9 tok/s |
| 2 | 18.1–22.3 tok/s | 28.7 tok/s |
| 3 | 5.9 tok/s | 21.9 tok/s |
| 4 | 3.8 tok/s | out of memory at 32K (per-loop KV) |

llama.cpp's memory breakdown at 4 loops with `-ngl 99`: 640 MiB of weights on the GPU, 2,855 MiB
on the host, and 1,624 MiB of the KV cache on the host. With `-ngl 999` only the 642 MiB token
embedding stays on the host (normal for llama.cpp). The VRAM going *down* as loops went up was
the clue.

### Flash attention, KV cache, context (all `-ngl 999`)

| Config | Context | Speed | GPU (model + KV + compute) |
|---|---:|---:|---:|
| 2 loops, FA off, f16, per_loop (card) | 32K | 28.7 tok/s | 2853 + 2044 + 1140 MiB |
| 2 loops, FA on, f16, per_loop | 32K | 31.3 tok/s | 2853 + 2044 + 93 MiB |
| 2 loops, FA on, f16, per_loop | 64K | 31.6 tok/s | 2853 + 3836 + 125 MiB (tight) |
| 2 loops, FA on, f16, last_loop | 64K | 31.3 tok/s | 2853 + 2466 + 125 MiB |
| **2 loops, FA on, q8_0, per_loop (default)** | **64K** | **29.9 tok/s** | **2853 + 2037 + 364 MiB** |
| 2 loops, FA on, q8_0, last_loop | 128K | 30.1 tok/s | 2853 + 2534 + 684 MiB |
| 2 loops, FA off, f16, per_loop | 64K | — | out of memory |
| 4 loops, FA off, f16, last_loop | 32K | 17.9 tok/s | 2853 + 1314 + 1140 MiB |
| 4 loops, FA on, f16, last_loop | 32K | 19.0 tok/s | 2853 + 1314 + 93 MiB |

(300-token story, thinking off, one run each.)

### Final defaults, `qwen-bench.py`

| Loops | Generation | Prefill 15.8K | Peak VRAM | Tool call |
|---|---:|---:|---:|---|
| 2 (per_loop, 64K) | 29.3 tok/s | 852 tok/s | 5.7 GB | ✓ `read` |
| 4 (last_loop, 64K) | 17.7 tok/s | 492 tok/s | 5.1 GB | ✓ `read` |

From pi (`start_dory.sh -p "Read scripts/stop-server.sh and tell me … how it finds the process
to stop"`): one `read` call, correct one-sentence answer.

## Quality check

Speed alone can't show whether flash attention, q8_0 or the last-loop cache change the
model. `llama-perplexity` on the first 6K tokens of `docs/TECHNICAL.md` (3 chunks × 2048),
with KL divergence against the card's settings (2 loops, per_loop, f16, FA off):

| Config | PPL | KLD vs card | Same top token |
|---|---:|---:|---:|
| card settings (2 loops) | 6.728 | — | — |
| FA on | 6.721 | 0.0070 | 96.4% |
| FA on, q8_0 | 6.699 | 0.0069 | 96.4% |
| FA on, last_loop | 6.661 | 0.0155 | 94.6% |
| FA on, last_loop, q8_0 | 6.674 | 0.0157 | 94.4% |
| 1 loop (FA on) — for scale | 7.138 | 0.0942 | 85.8% |
| 4 loops, last_loop (FA on) | 6.599 | 0.0371 | 91.3% |

- Flash attention costs a small, fixed drift (0.007); q8_0 adds nothing measurable on top.
- The last-loop cache doubles that drift but does not raise perplexity, matching the card's
  "close to the per-loop mode".
- Loops matter much more: 7.14 → 6.73 → 6.60 for 1 → 2 → 4.
- One text, 6K tokens: enough to rule out a broken setting, not a benchmark.

## Notes

- **Thinking** is on by default (the template opens `<think>`); llama-server splits it into
  `reasoning_content`. The card's sampling (temp 1.0, top-p 0.95, top-k off) is the default.
- The warning *"the V embeddings have different sizes across layers and FA is not enabled -
  padding V cache to 1024"* only appears with flash attention off.
- The PR's own stated limits: context shifting unsupported (the card adds `--no-context-shift`),
  no loop counts above 4, Metal untested, long-context qualification incomplete.
- `build-llama-cuda.sh` builds `llama-server llama-cli llama-bench`; the quality check also
  needs `cmake --build vendor/dory-llama.cpp/build --target llama-perplexity`.
- The measurements above were made on the PR build (`brianchmielNV/llama.cpp` @ `43e865e`);
  the patched build has the identical source tree; re-checked with `qwen-bench.py`: 28.8 tok/s
  generation (29.3 on the PR build), 852 tok/s prefill (same), tool call ✓.

## Files

- `patches/llama.cpp-dory-pr30306.patch` — Dory support for llama.cpp (PR #30306), MIT
- `scripts/build-llama-cuda.sh` — `PATCH=` applies it on top of `LLAMA_REF`
- `scripts/run-qwen.sh` — `QMODEL=dory`, knobs `LOOPS`, `LOOP_KV`, `FA`
- `scripts/start_dory.sh` = `QMODEL=dory start-qwen.sh`, with shepherd-pi's tools hidden
- `scripts/download-qwen-bakeoff.sh dory` — pinned revision + sha256
- `config/pi-provider.json` — `dory1-4b` model entry (text only, reasoning, 64K)

## Sources

- Model card — https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16
- GGUF — https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF
- llama.cpp PR — https://github.com/ggml-org/llama.cpp/pull/30306
