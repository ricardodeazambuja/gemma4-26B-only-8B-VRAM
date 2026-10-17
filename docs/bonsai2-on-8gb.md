# Bonsai 2 27B on 8 GB VRAM — Qwen3.8-27B in ternary weights

**Date:** 2026-10-16 · **Branch:** `bonsai2_on_8gb_vram` (from `qwen_on_8gb_vram`) · **Status:** in progress
**Companion:** [`qwen-bakeoff.md`](qwen-bakeoff.md) — same machine, same scripts, same smoke tests.

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
  must all fit in VRAM. KV at q8_0 is ~0.57 GB at 16K and ~1.14 GB at 32K (same architecture
  as Qwen3.8-27B: 16 of 64 layers use full attention, 4 KV heads × 256). Default here: **16K**.

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

# Serve (all layers on GPU, 16K ctx, card sampling incl. min-p 0.05) and open pi
QMODEL=bonsai2 bash scripts/start-qwen.sh
```

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

*(to be filled in)*

## References

- Model card — https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
- Base model — https://huggingface.co/Qwen/Qwen3.8-27B
- PrismML llama.cpp fork — https://github.com/PrismML-Eng/llama.cpp (release `prism-b10777-e3de715`)
- PrismML run instructions ("source of truth" per the card) — https://github.com/PrismML-Eng/Bonsai-demo
