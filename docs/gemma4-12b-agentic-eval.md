# Gemma 4 12B v2 "agentic" (fable5-composer2.5) — TurboQuant eval for 8 GB VRAM

Evaluation of [`yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF`](https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF)
on this box (RTX 2070 Max-Q, **8 GB VRAM**, ~7.6 GB free; 31 GB RAM). The v2 **agentic**
successor to the v1 coder ([gemma4-12b-coder-eval.md](gemma4-12b-coder-eval.md)).

**Status: COMPLETE** (2026-06-21). Q4_K_M downloaded + sha256-verified + run via the
**TurboQuant fork**; full GPU offload achieved on 8 GB. Other quants not yet measured.

---

## TL;DR / recommendation

**Use `TURBO=1` — the TurboQuant `turbo3` V-cache lets Q4_K_M fully offload (`-ngl 99`) on 8 GB.**

Three ways to launch (server only → server+pi → zero-config preset):

```bash
# 1. Server only (run-12b-agentic.sh -h for all knobs)
TURBO=1 bash scripts/run-12b-agentic.sh             # -ngl 99, K=q8_0 V=turbo3, 16K ctx -> ~28 t/s
TURBO=1 CTX=25000 bash scripts/run-12b-agentic.sh   # author's exact recipe (marginal on 8 GB — see below)
bash scripts/run-12b-agentic.sh                     # stock build, partial offload -ngl 40 (no turbo3)

# 2. Server + pi + stop-on-exit offer (start.sh-style)
TURBO=1 bash scripts/start-agentic.sh

# 3. Zero-config: the tweet recipe baked in (just run it)  <-- the easy button
bash scripts/start-turbo.sh                         # = TURBO=1 QUANT=Q4_K_M CTX=16384, then pi
bash scripts/start-turbo.sh -p "explain @README.md" # one-shot
```

> **CTX note for 8 GB:** the tweet uses `-c 25000`, but on this RTX 2070 25k is on the
> knife-edge — it fit once and OOM'd once (a ~148 MiB compute-buffer alloc), depending on
> momentary desktop VRAM. **16384 is the reliable default** (what `start-turbo.sh` uses).
> For the literal 25k: `CTX=25000 bash scripts/start-turbo.sh` (close other GPU apps, or
> add `BATCH=256 UBATCH=256`).

| Mode (Q4_K_M) | Build | offload | KV (K/V) | ctx | VRAM | tok/s | Verdict |
|---|---|---|---|---|---|---|---|
| **TURBO=1** | turboquant fork | **-ngl 99 (all 48)** | q8_0 / **turbo3** | 16K | 7.76/8 GB | **~28.3** | ✅ **recommended** |
| stock | EAGLE3 / mainline | -ngl 40–42 (partial) | q8_0 / q8_0 | 32K | ~7.1 GB | ~16.5 | works without the fork |

Same DENSE 48-layer `gemma4_unified` arch as v1, so `-ngl` (not `--cpu-moe`) is the VRAM
lever. The new win is `turbo3`: a ~3.5-bit V cache that shrinks KV enough to keep **all**
layers on the GPU, ~1.7× the partial-offload speed. Matches the model author
(@analogalok)'s ~30 t/s on an RTX 4060.

⚠️ Model card: *"not safety-aligned, reduced refusals"* — task-focused agent; review its output.

---

## 1. What this model is

| | |
|---|---|
| Repo | `yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF` |
| Architecture (GGUF metadata) | `gemma4` (dense, 48 layers, head_dim 256), GGUF v3, 667 tensors |
| Base model | `google/gemma-4-12B-it` |
| Quants | Q3_K_M (5.7 GB) · **Q4_K_M (6.87 GB, recommended)** · Q6_K (9.11) · Q8_0 (11.8) — **no Q2_K** |
| Native context | 262144 |
| Chat template | custom thinking + native tool-call protocol (needs `--jinja`) |
| Focus | coding + agentic/terminal tool-use; claims ~55% vs base ~15% on tau2-bench telecom |
| Extras | `MTP/` speculative-decoding drafts; `TQ3_1S`/`TQ4_1S` **weight** quants (separate feature) |

Sampling (card): `temp 1.0, top_p 0.95, top_k 64`; `rep_pen 1.1` fixes a `0000…` garble mode
(the run script defaults to this). Greedy (`temp 0`) fine for coding.

## 2. TurboQuant — what it is and how it was built

`turbo3` is a **runtime KV-cache codec** (`--cache-type-v turbo3`), **not** a model format —
so it works on *any* gemma4 GGUF (v1, base, 26B-A4B), no special download. It's Walsh-Hadamard
rotated polar quantization (Google's TurboQuant, ICLR 2026) and is **not in mainline llama.cpp**;
it needs the [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant) fork.

Build (CUDA, sm_75) — done into a **separate** dir so the EAGLE3 `vendor/llama.cpp` build is untouched:

```bash
git clone --depth 1 --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant vendor/llama-cpp-turboquant
SRC_DIR=$PWD/vendor/llama-cpp-turboquant LLAMA_REF=feature/turboquant-kv-cache \
  ENV_NAME=llamacpp-cuda bash scripts/build-llama-cuda.sh
```

Binary lands at `vendor/llama-cpp-turboquant/build/bin/`; `TURBO=1` in the run script auto-selects it.

- CUDA flash-attn vec instances exist for head_dim **64/128/256** (gemma4 = 256), K∈{f16,q8_0,turbo2/3/4}
  × V∈same. Recipe `-ctk q8_0 -ctv turbo3` (K is sensitive, V tolerates aggression).
- The gemma4 `dk=512` FA kernels are **Metal-only** per the README; CUDA vec tops out at 256 —
  fine for the 12B text model, may matter for vision/`gemma4-assistant` paths.

## 3. Verification & results (RTX 2070 Max-Q, 8 GB)

- **File integrity:** `sha256sum` == HF `lfs.oid` `0b9506ca…c96791` ✅ (see lesson §4).
- **turbo3 gate (v1 coder, same arch):** `q8_0`-K + `turbo3`-V at `-ngl 42`, head_dim 256 →
  coherent (correct thinking trace + code). Proves the turbo3 CUDA kernel works on sm_75.
- **v2 full offload:** `TURBO=1`, `-ngl 99`, 16K ctx → **7.76/8 GB, ~28.3 t/s**, coherent on
  thinking + coding (merge_intervals, hash-map, string reversal all correct).
- **Memory gotcha:** `-ngl 99` OOMs with the default `n_parallel=4` + `n_batch=2048` (a ~0.5 GB
  CUDA compute buffer). `TURBO=1` defaults to `--parallel 1 -b 512 -ub 512` (override via
  `NP`/`BATCH`/`UBATCH`). 25K ctx is tighter than 16K here; 16K is the safe default.

## 4. Lesson learned — a corrupt GGUF looks like a kernel/model bug

The first download (aria2c `-x16` against HF's flaky Xet CDN, then I interrupted a stuck
phantom-retry) produced a file with the **right size and a valid GGUF header but wrong tensor
data**. It loaded fine and emitted pure `<unused32><unused32>…` garbage — which *looked* like a
turbo3 / full-offload failure. It wasn't: the v1 model on the identical code path was coherent
throughout, and `sha256sum` vs HF's `lfs.oid` exposed the mismatch.

**Always sha256-verify a GGUF before blaming the engine.** Robust download = a verify loop:
`aria2c -x8 --max-tries=10` (not `-x16`/`--max-tries=0`, which hang) → `sha256sum` → re-download
on mismatch; let aria2c exit 0 naturally, never reuse an interrupted partial. Debug order on
garbage from a new GGUF: (1) checksum, (2) only then suspect quant/kernel/arch.
