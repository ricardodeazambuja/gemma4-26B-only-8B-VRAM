# Bonsai 2 27B on 8 GB VRAM — Qwen3.8-27B in ternary weights

**Date:** 2026-10-16 · **Branch:** `bonsai2_on_8gb_vram` (from `qwen_on_8gb_vram`) · **Status:** works — 27 tok/s fully on the GPU at 16K context; real-task trial pending
**Companion:** [`qwen-bakeoff.md`](qwen-bakeoff.md) — same machine, same scripts, same smoke tests.

## TL;DR

- **Qwen3.8-27B becomes usable on 8 GB: ~27 tok/s**, vs **2.5 tok/s** for the GSQ-RCO
  `IQ3_XXS` build of the same model (≈11×). The whole 5.95 GB model sits on the GPU.
- Tool calls ✓, 15.6K-token prefill at 269 tok/s, code check correct (0/500 wrong).
- **Needs the PrismML llama.cpp fork**, built here from source with the same locked CUDA 12.2
  toolchain as the Qwen work.
- **Turing gotcha:** the fork's flash-attention kernel for a **q8_0 KV cache has no sm_75
  code** → the server aborts on the first request. **Use an f16 KV cache** (now the default
  for `QMODEL=bonsai2`).
- **Context is capped at ~16K** on 8 GB (7.45 GB used with f16 KV); 32K does not fit.
  The 35B-A3B MoEs in [`qwen-bakeoff.md`](qwen-bakeoff.md) keep 128K.
- **From pi:** Bonsai's chat template rejects pi's `reasoning_effort: "high"` (HTTP 500); the
  pi model entry now maps pi's thinking levels onto the values it accepts. Use pi's
  **`medium`** thinking level — `xhigh` thinks for minutes at 27 tok/s.

## Why this model

In the Qwen bake-off, Qwen3.8-27B (ISTA-DASLab GSQ-RCO `IQ3_XXS`, 10.4 GB) only ran at
**2.5 tok/s**: half of it had to live in system RAM, and those i-quant layers are
CPU-compute-bound. [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)
(PrismML, 2026-09-16, Apache-2.0) is the **same Qwen3.8-27B** (architecture unchanged) with
ternary weights — every weight is −1, 0 or +1 with one FP16 scale per 128 weights, stored in a
Hadamard-rotated basis. The `PTQ1_0` pack is **5.95 GB**, so the whole model fits on the GPU.

Card claims (PrismML's own evaluation, EvalScope + vLLM on H100, thinking mode — *not
re-measured here*):

| Benchmark | Qwen3.8-27B FP16 | UD-Q4_K_XL | IQ2_XXS | Bonsai 2 (5.95 GB) |
|---|---:|---:|---:|---:|
| LiveCodeBench | 90.05 | 87.96 | 56.40 | 90.07 |
| AIME26 | 94.58 | 93.00 | 57.50 | 95.83 |
| MMLU-Redux | 91.46 | 93.35 | 88.93 | 89.09 |
| BFCL v3 (tool calling) | 76.74 | — | — | 74.92 |

No agentic-coding benchmark (SWE-bench / Terminal-Bench) is reported; BFCL v3 is the only
tool-use number.

## Requirements specific to this model

- **The PrismML llama.cpp fork.** Stock llama.cpp rejects `PTQ1_0` / `PQ2_0` as unknown types,
  and loads `Q2_0` but produces garbage (no Hadamard activation transform) — card,
  "These files need our llama.cpp build". Fork: https://github.com/PrismML-Eng/llama.cpp,
  tested at release tag `prism-b10777-e3de715` (= `e3de715222b2667c54a267441159a5729f11155e`).
- **Turing works in principle:** the fork's CUDA code has an explicit path for this pack on
  Turing and newer (`ggml/src/ggml-cuda/mmvq.cu`: `type == GGML_TYPE_PTQ1_0 && ... cc >= GGML_CUDA_CC_TURING`).
  PrismML's published speed table only covers Ampere and newer.
- **Context is the trade-off:** it is dense, so weights (5.95 GB) + KV cache + compute buffers
  must all fit in VRAM. KV at f16 is ~1.14 GB at 16K and ~2.3 GB at 32K (same architecture as
  Qwen3.8-27B: 16 of 64 layers use full attention, 4 KV heads × 256). A q8_0 KV would halve
  that, but crashes on Turing with this fork (see Measurements). Default here: **16K, f16 KV**.

## Run it

Same setup as the Qwen bake-off (toolchain lock, download env — see
[`qwen-bakeoff.md`](qwen-bakeoff.md#run-these-models-on-your-own-8-gb-machine)), plus the fork:

```bash
# Build the PrismML fork with the same locked CUDA 12.2 toolchain, into vendor/prism-llama.cpp
CUDA_VER=12.2 LLAMA_REPO=https://github.com/PrismML-Eng/llama.cpp \
  LLAMA_REF=prism-b10777-e3de715 SRC_DIR=$PWD/vendor/prism-llama.cpp \
  bash scripts/build-llama-cuda.sh

# Download + verify at the pinned revision (PTQ1_0 + vision projector, ~6.6 GB)
bash scripts/download-qwen-bakeoff.sh bonsai2

# Serve (all layers on GPU, 16K ctx, f16 KV, card sampling incl. min-p 0.05) and open pi
bash scripts/start_bonsai2.sh          # = QMODEL=bonsai2 bash scripts/start-qwen.sh --thinking medium -xt 'shepherd_*'
```

`start_bonsai2.sh` needs no environment variables. It checks that the model and the fork are
in place, refuses if another model is already serving on the port (`scripts/stop-server.sh`
first), and passes any pi flags through.

It also hides shepherd-pi's tools. With every extension in `~/.pi/agent/extensions` loaded,
pi's first request is ~5,400 tokens, a third of the 16K context, and shepherd-pi's 10 tool
definitions are ~2,000 of them; without them it is ~3,400 (measured by capturing pi's request
with each extension loaded alone). Your own `-xt`/`-t` replaces that default.

16K fills fast. One `fetch_page` of a JSON API (`pypi.org/pypi/bonsai/json`, 233 KB on one
line) once returned ~146K tokens because the page cap counted lines only; it now also cuts
long lines and caps each page at 6,000 characters. A session that overflowed stays broken
(each request resends the oversized turn): start a new one with `/new`.

`build-llama-cuda.sh` gained `LLAMA_REPO` for this, and now always checks out exactly
`LLAMA_REF` (tag, branch or commit) or fails — previously a failed `--branch` clone fell back
to the default branch's HEAD, and a failed fetch silently rebuilt the old checkout.

### Exact files

| File | Size | sha256 |
|---|---:|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5.95 GB | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 0.63 GB | `6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903` |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` (not used; 2-bit slots, 7.21 GB) | 7.21 GB | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |

Pinned repo revision: prism-ml/Ternary-Bonsai-2-27B-gguf `b072e1d3b35a0a630cece372c2127528e0994386`.

## Measurements

Same machine and smoke tests as [`qwen-bakeoff.md`](qwen-bakeoff.md) (RTX 2070 Max-Q 8 GB,
i7-8750H, 32 GB DDR4). Fork built from source at `e3de715` with the locked CUDA 12.2.2 /
gcc 12.4 toolchain (14 min, 525 steps). Single runs unless stated.

### Build note: the post-build smoke test "failed" — VRAM, not the build

`build-llama-cuda.sh` ran its Gemma smoke test while another server held 7.3 GB of VRAM;
`llama-bench` printed only `failed to load model`. With the GPU free, the same fork binary
runs that test fine (Gemma4-26B-A4B, `--n-cpu-moe 22`: pp8 42.8, tg8 25.8 tok/s).
Free the GPU before building.

### `llama-bench` (`-ngl 99 -t 6 -fa 1 -r 2`, default f16 KV)

| test | tok/s |
|---|---:|
| pp512 | **322.3 ± 2.9** |
| tg64 | **28.47 ± 0.08** |

### Server (`QMODEL=bonsai2`: `-ngl 99`, 16K ctx, `-fa on`)

| KV cache | VRAM | Gen tok/s | Tool call | Prefill (15.6K tok) | Notes |
|---|---:|---:|---|---:|---|
| q8_0 | 6.97 GB idle | — | — | — | **aborts on first request:** `CUDA kernel flash_attn_ext_f16 has no device code compatible with CUDA arch 750` (`fattn-mma-f16.cuh:1877`, head dim 256, KV type q8_0) |
| **f16** | **7.45 GB** | **27.0** | ✓ | **269 tok/s** (58 s) | 1,383-token answer, `finish=stop` |
| f16 (defaults, re-run) | — | 27.1 | ✓ | — | confirms the per-model default |

### Code check ([`qwen-codecheck.py`](../scripts/qwen-codecheck.py))

| | tokens | gen tok/s | own doctests | 500 random vs reference | time |
|---|---:|---:|---|---|---:|
| Bonsai 2 27B | 1,597 | 26.5 | pass | **0 wrong** | 60 s |

### Compared with the Qwen bake-off (same rig, same tests)

| Model | Size | Placement | Context | Gen tok/s | Prefill | Code check |
|---|---:|---|---:|---:|---:|---|
| Qwen3.8-27B GSQ-RCO IQ3_XXS | 10.4 GB | 30/64 layers GPU | 32K | 2.5 | 167 | — |
| **Bonsai 2 27B PTQ1_0** | **5.95 GB** | **all GPU** | **16K** | **27.0** | **269** | correct, 1,597 tok |
| Qwen3.8-35B-A3B distill | 19.6 GB | experts mostly RAM | 128K | 26.2 | 167 | correct, 557–784 tok |
| Qwen3.6-35B-A3B | 17.7 GB | experts mostly RAM | 128K | 22.8 | 221 | correct, 3,558 tok |

Bonsai is the only way here to run the actual Qwen3.8-27B at a usable speed. The trade is
context: 16K vs 128K for the MoEs — tight for long agent sessions with a large system
prompt (pi + extensions).

## Using it from pi

### `reasoning_effort` must be mapped

pi sends its thinking level as `reasoning_effort`. The mainline llama.cpp build used in the
Qwen bake-off ignored that field; the PrismML fork passes it into the chat template, and
Bonsai's template only accepts three values:

```
Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
```

Every request from pi at its `high` level failed with HTTP 500. Fix: a `thinkingLevelMap` on
the `bonsai2-27b` entry in [`config/pi-provider.json`](../config/pi-provider.json) (synced into
pi by `start-qwen.sh` / `configure-pi.sh`):

| pi thinking level | sent as `reasoning_effort` |
|---|---|
| minimal, low, medium | `medium` |
| high, xhigh | `xhigh` |

`low` is deliberately not used: the card says `low` "is not supported and … the model will
behave close to `xhigh`". (pi's `max` level is left unmapped — a value outside the template's
list fails the same way.) Checked directly against the server: `xhigh` and `medium` work,
`high` returns the error above.

### Prefer pi's `medium` thinking level

At `xhigh` (the template default, and what pi's `high` maps to), the model thinks at great
length; at 27 tok/s that is minutes per turn. Use `--thinking medium` (or switch inside pi).

### Open: slow pi side-requests

In one-shot mode (`pi -p`), the main answer arrived quickly — server log: pi's 5,468-token
prompt in 17.7 s (308 tok/s), the reply in 8.4 s — but pi then sent further small requests
(292- and 130-token prompts), one of which generated 1,255 tokens (48 s), and `pi -p` had not
exited after 400 s, at both `high` and `medium`. Runs with `--no-session` finished promptly.
Not yet confirmed which pi feature issues these requests (session-related side calls are the
working guess). The server has one slot (`--parallel 1`), so a long side-request delays the
next turn.

Debugging note: when a client gives up (e.g. a `timeout`), llama-server keeps generating the
abandoned request; later requests queue behind it on the single slot and look like hangs.
Check `curl -s http://127.0.0.1:8080/slots` (`is_processing`) before blaming the client.

## Next steps

1. Real-task trial through pi (`QMODEL=bonsai2 bash scripts/start-qwen.sh --thinking medium`),
   especially how far 16K context goes with pi's system prompt and tools.
2. Identify pi's slow side-requests (see "Open: slow pi side-requests") and give them short
   thinking.
3. Optional: `PQ2_0` (7.21 GB) is not an option on 8 GB with any useful context.
4. Optional: report the q8_0-KV flash-attention gap on Turing to PrismML.

## References

- Model card — https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
- Base model — https://huggingface.co/Qwen/Qwen3.8-27B
- PrismML llama.cpp fork — https://github.com/PrismML-Eng/llama.cpp (release `prism-b10777-e3de715`)
- PrismML run instructions ("source of truth" per the card) — https://github.com/PrismML-Eng/Bonsai-demo
