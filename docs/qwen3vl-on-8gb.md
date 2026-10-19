# Qwen3-VL-8B-Instruct on 8 GB VRAM — a vision model that fits entirely on the GPU

**Date:** 2026-10-18 · **Branch:** `qwen3vl_on_8gb_vram` (from `bonsai2_on_8gb_vram`) · **Status:** works — ~38 tok/s, vision + tool calls + pi, 16K context
**Companion:** [`qwen-bakeoff.md`](qwen-bakeoff.md), [`bonsai2-on-8gb.md`](bonsai2-on-8gb.md) — same machine, same scripts.

## Introduction

Qwen3-VL-8B-Instruct is the 8B member of Alibaba Qwen's Qwen3-VL vision-language family,
which comes in dense and MoE sizes, each with an Instruct and a reasoning ("Thinking")
edition; this is the Instruct one, so it does not think before answering. It is a 36-layer
transformer with standard attention in every layer (8 KV heads × 128), plus a vision encoder
that Qwen ships as a separate `mmproj` file. Per the card, the architecture adds
Interleaved-MRoPE positions and DeepStack, which fuses features from several levels of the
vision transformer, and the native context is 256K.

It was worth trying because it is small enough to keep everything, vision included, on the
GPU. The card claims OCR in 32 languages, GUI-agent use and code generation from images,
and Qwen publishes official GGUFs (`Q4_K_M` / `Q8_0` / FP16 weights, `Q8_0` / FP16
projectors) that run on stock llama.cpp.

The naive setup does not fit. `Q8_0` weights alone are 8.7 GB. With `Q4_K_M` (5.03 GB) and
the projector (0.75 GB), the KV cache becomes the limit: 2.25 GB at 16K in f16, and even at
q8_0, 32K (2.39 GB) is too much. The image encoder is the second problem: on the CPU a 4K
photo took 270 s, and on the GPU an uncapped 4K photo (4,096 tokens) runs it out of memory
and aborts the server.

What makes it work is plain configuration, with no fork and no rebuild: `Q4_K_M` weights
with all 36 layers on the GPU, a 16K q8_0 KV cache (1.20 GB), the projector on the GPU
(`MMPROJ_GPU=1`), and images capped at 1024 tokens (`IMAGE_MAX_TOKENS=1024`). That uses
~7.4 GB and encodes the same 4K photo in 5.4 s.

The trade-offs are the 16K context, downscaled images (2048 tokens fits but leaves less
headroom), and no thinking mode. pi also needs an image-capable model entry before it sends
images. See "Measurements" and "pi integration".

## TL;DR

- [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF), `Q4_K_M`
  (5.03 GB) + `mmproj Q8_0` (0.75 GB). **Stock llama.cpp** (`vendor/llama.cpp`, arch `qwen3vl`)
  — no fork, no rebuild.
- **Everything on the GPU:** 36 layers, the vision projector and a 16K q8_0 KV cache:
  **~7.4 GB used** of 8 GB.
- **~38 tok/s** generation (26 tok/s with the context full), **~830–900 tok/s** prefill.
- **Vision ✓** (shapes, colors, layout, OCR), **tool calls ✓**, **16K needle ✓**, **pi ✓**
  (images attached with `@file.png` reach the model).
- **Two settings make it practical**, both now the `qwen3vl` defaults:
  - **projector on the GPU** (`MMPROJ_GPU=1`): on the CPU a 4K photo took **270 s**;
  - **≤1024 tokens per image** (`IMAGE_MAX_TOKENS=1024`): uncapped, a 4K photo is
    4,096 tokens and the GPU encoder **OOMs and aborts the server**.
- Not a thinking model (`reasoning: false` in pi). No MTP head.

## Run it

```bash
bash scripts/download-qwen-bakeoff.sh qwen3vl       # 5.8 GB, pinned revision + sha256
bash scripts/start_qwen3vl.sh                       # server (with vision) + pi
bash scripts/start_qwen3vl.sh -p "describe @photo.jpg" --no-session   # one-shot

QMODEL=qwen3vl bash scripts/run-qwen.sh --image     # server only, OpenAI API on :8080
```

The download script needs an env with the `hf` CLI (`ENV_NAME=...`). If downloads stall
(broken IPv6 to Hugging Face), force IPv4 for the `hf` process and set `HF_HUB_DISABLE_XET=1`.
Both files verified against the pinned sha256.

## Measurements (RTX 2070 Max-Q 8 GB, 2026-10-18)

All with `-ngl 99 -c 16384 -ctk/-ctv q8_0 -fa on --parallel 1 -b 512 -ub 512`.

| Test | Prompt | Prefill | Generation | Result |
|---|---:|---:|---:|---|
| Text: Fibonacci function | 22 tok | 386 tok/s | 36.5 tok/s | correct |
| Synthetic 640×400 image, projector on CPU | 290 tok | 57 tok/s | 38.7 tok/s | all 3 shapes + colors + order, OCR `INVOICE #7319` exact |
| 3840×2160 photo, projector on CPU, uncapped | 4096 tok | 16 tok/s (**270 s**) | 31.8 tok/s | correct description |
| 3840×2160 photo, projector on GPU, uncapped | — | — | — | **CUDA OOM → server aborts** |
| 3840×2160 photo, GPU, `IMAGE_MAX_TOKENS=1024` | 1027 tok | 480 tok/s (**5.4 s** total) | 38.7 tok/s | correct description |
| 3840×2160 photo, GPU, `IMAGE_MAX_TOKENS=2048` | 1999 tok | 377 tok/s (8.8 s total) | 36.7 tok/s | correct, slightly more detail |
| Tool call (`get_weather`) | — | — | — | `finish_reason=tool_calls`, `{"city": "Lisbon"}` |
| Needle in 15,922-token filler | 15922 tok | 828 tok/s | 26.3 tok/s | `MAGENTA-4471` exact |

VRAM (nvidia-smi, desktop included): 6.35 GB with the projector on the CPU, **7.39 GB** with it on
the GPU (idle and with the context full), 7.54 GB peak during a 2048-token image. 2048 fits, but
1024 is the default because the remaining headroom depends on what the desktop is using.

KV-cache math (36 layers × 8 KV heads × 128 dims): 16K = 2.42 GB at f16, 1.28 GB at q8_0;
32K at q8_0 is 2.57 GB, which does not fit next to 5.03 GB of weights and the 0.75 GB projector
(8.35 GB total).

## pi integration

- `config/pi-provider.json` has a `qwen3-vl-8b` entry with `"input": ["text", "image"]`.
  **Without it pi does not send images**: an unknown model id falls back to a text-only custom
  model, and the model then replies that it "can't view the image".
- `start_qwen3vl.sh` = `QMODEL=qwen3vl start-qwen.sh --image`, with shepherd-pi's tools hidden
  (same reason as `start_bonsai2.sh`: ~2K of the 16K tokens). It refuses to reuse a server that
  is serving a different model.
- A non-interactive `-p` run hangs if stdin is left open (e.g. under a harness). Use `< /dev/null`.

## Not done yet

- No real-task trial (agentic coding in pi) and no comparison against Gemma 4's vision on the
  same images.
- `Q8_0` weights (8.7 GB) do not fit fully. Partial offload was not tried.
