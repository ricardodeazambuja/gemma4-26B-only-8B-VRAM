# Gemma 4 12B Coder (fable5-composer2.5) — quant evaluation for 8 GB VRAM + 31 GB RAM

Evaluation of [`yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF)
on this box (RTX 2070 Max-Q, **8 GB VRAM**; **31 GB** system RAM; CUDA backend built in `vendor/`).

**Status: IN PROGRESS** — see [Results](#results) for what has completed.

---

## TL;DR / recommendation

_(filled in once benchmarks complete)_

---

## 1. What this model is — and why the repo's headline trick does NOT apply

| | |
|---|---|
| Repo | `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF` |
| Architecture (GGUF metadata) | `gemma4` |
| Base model | `google/gemma-4-12B-it` |
| Params | ~11.9 B — **dense**, not MoE |
| Native context | 131072 |
| Chat template | custom, with thinking (`<\|channel>thought`) + tool-call channels (needs `--jinja`) |
| Likes / downloads | 615 / ~20k (as of 2026-06-15) |

**Key consequence:** this repo's whole reason for existing — llama.cpp's `--cpu-moe` /
`--n-cpu-moe` (`NCMOE`) flag, which pins the 26B-A4B's **expert** FFN weights to RAM while keeping
attention + KV on the GPU — **does not apply to a dense model.** There are no experts to offload.

For a *dense* 12B the levers that fit it into 8 GB VRAM are different:

1. **`-ngl N` — partial layer offload.** Put the first N transformer layers on the GPU and run the
   rest on the CPU (in RAM). This is the primary VRAM lever for a dense model. (`run-server.sh`
   hardwires `-ngl 99` = all layers on GPU, which is correct for the MoE-on-RAM design but will OOM
   a dense 12B that doesn't fully fit — see §4 for the dense launch recipe.)
2. **KV-cache quantization** (`KVQUANT=q8_0`) — shrinks the per-token context cost in VRAM, freeing
   room for more layers. Same lever as the MoE path; near-lossless at q8_0.
3. **Context sizing** (`-c`) — smaller context = smaller KV cache = more room for weights.
4. **Flash attention** (`-fa on`) — required for V-cache quant; also saves KV memory.

The available quants:

| Quant | File size | Fits fully in 8 GB VRAM? (weights only) |
|---|---|---|
| Q2_K | 4.83 GB | yes, with room for KV + context |
| Q4_K_M | 7.38 GB | no — needs partial `-ngl` offload |
| Q6_K | 9.79 GB | no — substantial RAM offload |
| Q8_0 | 12.67 GB | no — mostly RAM (near-lossless) |

All four fit in 31 GB RAM with room to spare.

---

## 2. Safety assessment

**Verdict: safe to download and run, with one ordinary caveat (review generated code).**

The user explicitly asked to confirm the model is safe. Findings:

- **File format is data-only.** GGUF is a tensor/metadata container parsed by llama.cpp. Unlike
  PyTorch `.bin`/`.pt`/`.ckpt` (Python *pickle* → arbitrary code execution on load), **loading a
  GGUF does not deserialize or execute code.** The classic "malicious weights run code when you load
  them" supply-chain vector does not apply to GGUF.
- **The repo contains only data.** Full file listing: 4× `.gguf`, `README.md`, `.gitattributes`.
  **No `.py`, `.sh`, no pickle, no executable** — nothing that runs on your machine.
- **The chat template is sandboxed.** The embedded Jinja template is rendered by llama.cpp's *minja*
  engine, a restricted template evaluator with no filesystem/network/`os` access. Reviewed the
  template: it is elaborate (tool-call + thinking-channel formatting) but does only string
  manipulation — no injection path to the host.
- **Checksums verified on download.** `hf download` validates each file's SHA-256 against the
  repo's LFS metadata; a corrupted/tampered transfer fails the download.
- **Parser-CVE exposure is low.** Early-2024 llama.cpp had heap-overflow CVEs in GGUF metadata
  parsing; those are mitigated by a recent build (this box's `llama-server` is a Jun-2026 build).
- **Uploader signal is positive-ish.** Public, non-gated, `license: gemma`, 615 likes / ~20k
  downloads. Not a guarantee, but not a throwaway account.

**Residual caveat (applies to *any* model):** a coder fine-tune could emit subtly insecure or
backdoored code suggestions. This is not detectable from the file — mitigate the normal way: read
the code it produces before running it. Nothing about *this* model's packaging raises that risk
above baseline.

---

## 3. Method

- Backend: locally-built **CUDA** binary (`vendor/llama.cpp/build/bin/`), per `.gemma4-menu`.
- Held constant across quants: **context = 32768**, **KV cache = q8_0**, **flash-attn on**.
- For each quant: find the **max `-ngl`** (layers on GPU) that loads inside 8 GB at that context,
  then measure prompt (pp) and generation (tg) tok/s with `llama-bench`, and record VRAM headroom.
- Coherence check: a real coding prompt through `llama-server` on the leading candidates (speed
  alone can't pick a winner — Q2_K may be fast but too degraded to code with).
- Hang-safety (running unattended): servers are backgrounded with timeouts; VRAM is verified freed
  (`nvidia-smi`) between quants; slow quants (Q6_K, Q8_0) run last.

---

## 4. Dense launch recipe (the deliverable — `models/` is gitignored)

_(filled in once the best `-ngl` per quant is known)_

---

## Results

_(appended incrementally as each quant completes, so a dropped session keeps finished work)_

| Quant | Size | Max `-ngl` @ 32k/q8_0 | VRAM used | pp tok/s | tg tok/s | Coherent? |
|---|---|---|---|---|---|---|
| Q2_K | 4.83 GB | 99 (fits fully) | 6.3 GB @ 4k ctx | 103 | 19.5 | **NO — degenerate** |
| Q4_K_M | 7.38 GB | _testing_ | | | | _pending_ |

### Diagnostics log (the hard part)

The model **loads** as arch `gemma4` and runs at full speed, but **Q2_K output is degenerate**
(greedy/temp-0 yields `ed.py.y…<channel|><channel|>…` / CJK noise). Ruled out, step by step:

- **Not the chat template / parser.** Raw `/completion` (no template, no parser) at temp 0 is also
  garbage. (Separately, the server's `peg-gemma4` output parser *crashes* on this model's
  `<\|channel>` output — a real llama.cpp robustness bug, but downstream of the garbage, not its
  cause.)
- **Not the vendor CUDA build, not CUDA, not flash-attn.** The conda-forge build (different commit
  `3c585b3`) on **CPU** garbles identically. Two independent builds, same failure → points at the
  file/config, not one binary.
- **Not a missing arch update.** `LLM_ARCH_GEMMA4` exists and `gemma4.cpp` even has a
  `gemma4_unified` logits-bias path. And **zero** model-execution commits exist between our build
  (88a3927, Jun 12) and today's master — so rebuilding to latest would not change model output.
- Load warnings are benign (token-type overrides for `</s>`, `<\|tool_response>`).

**Leading hypothesis:** the **Q2_K quant is over-degraded / bad** (most aggressive quant of a 12B
with unusual attention dims — `key_length=512`, per-layer MQA on global layers). The card only
*recommends* Q4_K_M; Q2_K is listed as "tiniest, runs almost anywhere" with no quality promise.
**Q4_K_M is the discriminator** — if it's coherent, Q2_K is simply unusable here; if it's also
garbage, the arch isn't correctly supported by any available llama.cpp and that's the finding.
