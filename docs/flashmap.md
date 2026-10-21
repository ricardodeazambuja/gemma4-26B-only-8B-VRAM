# FlashMap (flow-map speculative decoding) on 8 GB

Upstream: <https://github.com/al-jshen/flashmap> (MIT, created 2026-10-13), paper arXiv:2610.24365.
Branch: `flashmap_on_8gb_vram`. Measured 2026-10-20 on the RTX 2070 Max-Q 8 GB (sm_75), 31 GB RAM.

## TL;DR

- FlashMap had no llama.cpp support anywhere. This branch ports it to the vendored
  **buun-llama.cpp** fork as a variant of the fork's upstream-style DFlash drafter
  (`patches/buun-llama.cpp-flashmap.patch`, ~400 lines). It runs end to end with
  Qwen3-8B Q4_K_M + the FlashMap-Qwen3-8B drafter in 7.65 GB of VRAM.
- **The port drafts, but acceptance is about half of the paper's.** The mean acceptance
  length τ is ~3.2 on math (paper: 7.2–7.7) and ~1.5 on prose (paper: 3.1–3.3).
- **Net speed: no win yet.** With 16-token drafts: math +50% (59–63 vs 39–42 t/s),
  code ±0, prose −25–35%; aggregate 39.0 vs 39.8 t/s for the target alone.
- **The low τ is NOT target quantization.** A Q8_0 target gives the same τ as Q4_K_M
  (run C). It is also not the adaptive draft-depth controller (run A) nor the fused
  single-graph cycle (run B). **The prime suspect is a bug in the drafter graph port.**
  See [Open problem](#open-problem-and-next-step).

## What FlashMap is

A ~500M **drafter** for lossless speculative decoding (DFlash family). It proposes a block
of K=16 tokens in one forward pass. Instead of predicting each slot independently it applies
a *flow map* that jumps from Gaussian noise to the whole block in 1 (or 2–4) steps, so the
slots can depend on each other. Upstream serves it with HF transformers, SGLang (as
`--speculative-algorithm DSPARK`) and vLLM.

| Drafter | Target | Params | bf16 file |
|---|---|---|---|
| j-shen/FlashMap-Qwen3-8B | Qwen/Qwen3-8B | 500M | ~0.95 GB |
| j-shen/FlashMap-Qwen3-4B | Qwen/Qwen3-4B | 475M | ~0.95 GB |

Paper, Qwen3-8B, T=0, 1 step, H200, SGLang (τ / speedup):
GSM8K 7.69 / 6.14×, MATH 7.24 / 5.89×, MBPP 4.95 / 4.02×, HumanEval 5.82 / 4.73×,
MT-Bench 3.34 / 2.73×, Alpaca 3.14 / 2.56×. DFlash on the same table: τ 7.43 / 7.09 / … /
3.14 / 2.96. The Qwen3-4B tables are similar.

### Architecture (from `modeling_flashmap.py`, Qwen3-8B drafter)

- **Context.** Target hidden states at layers `[1, 9, 17, 25, 33]` (HF
  `hidden_states[i+1]`) are concatenated (5×4096), passed through `fc` and an RMSNorm, then
  through per-layer cross-attention `k_proj`/`v_proj` (k_norm, RoPE) into a cache. The
  anchor (last committed) token enters the same cache at its own position via the target
  embedding → `anchor_proj` → RMSNorm.
- **DiT body.** 6 layers, width 2048, 16 heads of 128. Each layer has AdaLN-zero
  self-attention over the 16 slots (non-causal), AdaLN cross-attention to the context, and
  an AdaLN GELU MLP (with biases). q/k RMSNorm uses eps 1e-5, the other norms 1e-6.
  RoPE is **interleaved** (pairs 2i, 2i+1), so ggml NORM mode, not NeoX.
- **Time conditioning.** Sinusoidal embeddings of the jump start `s` and end `t` produce
  `(scale, shift, gate)` for every norm. For a fixed grid these are constants; one step is
  `s=0 → t=1`.
- **One-step input.** `RMSNorm_state(uniform-posterior embedding) + gate·(t−s)·RMSNorm_noise(z @ Lᵀ)`
  with `z ~ N(0, I_4096)` and `L` the Cholesky factor of `EᵀE` (E = target embedding).
- **Output.** `head` (2048 → 4096), then the **target's** lm_head. Slot *j* sits at
  position anchor+1+*j* and predicts that token: 16 drafts per step.

## The port (buun-llama.cpp + patch)

Why the buun fork: it already has two DFlash pipelines. The **upstream-style** one
(`LLM_ARCH_DFLASH`, `src/models/dflash.cpp`, driven by `common_speculative_impl_draft_dflash`)
keeps the drafter's own KV cache and injects the target features into it. That is exactly
FlashMap's cached cross-attention context. Its **DSpark anchor-first** protocol
(`--spec-type draft-dspark`, `sample_from_anchor=true`) feeds `[anchor, mask×(n-1)]` at
positions P..P+n−1 and reads output row *i* as the draft for P+1+*i*. That is FlashMap's
block contract, so no driver changes were needed.

What the patch adds (all keyed off the GGUF key `dflash.flashmap.target_hidden_size`):

- 20 tensor types (`flashmap.*`, `blk.N.fm_*`) plus hparams `flashmap_n_embd_tgt` and
  `flashmap_qk_norm_eps`. The encoder input width is `n_target_layers × target n_embd`
  (DFlash assumed drafter width = target width).
- `build_flashmap_draft()` in `dflash.cpp`:
  - The context rows of the decode go to the KV cache via `build_attn(..., kv_only=true)`.
    These are the anchor embedding plus, under the fused single-graph cycle, the injected
    target rows.
  - Self-attention runs over 16 slots, and cross-attention reads the cached K/V with the
    **anchor row's KV mask repeated to every slot**. That mask covers context ≤ P in this
    sequence and hides the placeholder rows written at P+1…
  - A custom graph input carries the slot positions and the host-generated Gaussian noise
    (`GGML_FLASHMAP_SEED`).
  - The output goes through the target's lm_head, plus the fork's in-graph GPU top-K tail.
- The injection path uses the FlashMap q/k-norm eps. Rope type is NORM for FlashMap.
- Auto-detection: `common_speculative` and the server pick `draft-dspark` for a FlashMap
  GGUF, so plain `-md flashmap.gguf` is enough.

`scripts/convert-flashmap.py` writes the drafter GGUF and **bakes the one-step constants**
(num_steps=1 only):
- the AdaLN modulations (`mod = 1+scale`, `mod.bias = shift`, `gate`)
- `state_norm(mean row of E)`
- `noise_norm.weight × noise_gate × (t−s)`
- `L = chol(EᵀE)`, computed in float32 by row blocks with float64 Cholesky, as upstream does

It needs only the target's `embed_tokens` (fetched by HTTP range from shard 1) and copies
the tokenizer from the target GGUF. f16 output is 929 MiB; `--outtype q8_0` is also
available but untested.

Gotcha hit and fixed: worst-case graph reservation builds the drafter graph with a
512-token ubatch. The slot count is therefore `max(block_size, n_out)` (real drafts are
always one block).

## How to run

```bash
bash scripts/download-flashmap.sh        # Q4_K_M target, drafter, Qwen3-8B tokenizer + embed_tokens (sha256-checked)

# fork at the tested commit + nvcc-12.2 patch (build script), then the FlashMap patch
LLAMA_REPO=https://github.com/spiritbuun/buun-llama-cpp LLAMA_REF=34ef0ef7c6eaa58afd2011cf4385e8b2d76b225b \
  SRC_DIR=$PWD/vendor/buun-llama.cpp PATCH=patches/buun-llama.cpp-nvcc122-no-immintrin.patch \
  bash scripts/build-llama-cuda.sh
git -C vendor/buun-llama.cpp apply "$PWD/patches/buun-llama.cpp-flashmap.patch"
mamba run -n llamacpp-cuda cmake --build vendor/buun-llama.cpp/build -j 8 --target llama-server

mamba run -n local python scripts/convert-flashmap.py models/qwen3-8b/flashmap-hf \
  models/qwen3-8b/hf/embed_tokens.safetensors models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf \
  models/qwen3-8b/flashmap-qwen3-8b-f16.gguf

bash scripts/run-flashmap.sh                      # FlashMap, greedy (DRAFT=0 → target alone)
GGML_DFLASH_DRAFT_ADAPTIVE=0 bash scripts/run-flashmap.sh   # fixed 16-token drafts (best on math)
python3 scripts/flashmap-bench.py                 # 6 prompts, thinking off, temp 0, 512 tokens
```

## Measurements

`scripts/flashmap-bench.py`: 6 chat prompts, `enable_thinking=false`, temp 0, max 512 tokens,
ctx 8192, q8_0 target KV, `-np 1`. `τ` is the server's "mean len" (tokens per verify step).

| prompt | baseline t/s | default (adaptive depth) t/s / τ | **A**: 16-token drafts t/s / τ | **C**: Q8_0 target, 16-token, τ |
|---|---|---|---|---|
| math (divisors of 196) | 38.9 | 45.0 / 3.04 | **58.9** / 3.24 | 3.34 |
| gsm (Natalia's clips) | 42.4 | 49.4 / 3.23 | **63.0** / 3.38 | 2.67 |
| code (fib memo) | 41.7 | 41.8 / 2.61 | 47.9 / 2.64 | 2.63 |
| code2 (C binary search) | 39.7 | 38.5 / 2.41 | 42.6 / 2.34 | 2.41 |
| prose (story) | 38.9 | 30.5 / 1.59 | 25.7 / 1.49 | 1.47 |
| explain (hash table) | 39.2 | 33.8 / 1.91 | 30.6 / 1.70 | — (stopped) |
| **all** | **39.8** | 38.0 | 39.0 | — |

- **Default** = the fork's adaptive depth controller (`GGML_DFLASH_DRAFT_ADAPTIVE`, on). It
  shrinks drafts to 2–9 tokens after the first cycles. The first full-16 cycles accepted
  only 1–2 tokens.
- **A** = `GGML_DFLASH_DRAFT_ADAPTIVE=0`, so always 16 drafts. τ is unchanged, so the
  controller was not the limiter. With a fixed τ the deeper drafts pay off on math (each
  verify step is cheap relative to the extra accepted tokens) and cost more on prose.
- **B** = A + `GGML_DFLASH_ONEGRAPH=0`. Identical τ on math/gsm/code (the same 287/119/288
  accepted). The fused-inject path in the new graph is not the problem.
- **C** = Q8_0 target (`NGL=18 CTX=4096`, partly on the CPU, so speed is meaningless).
  The same τ as Q4_K_M, so **quantization of the target's features does not explain the gap**.
  Stopped after 5 prompts; the answer was already clear. Q8_0 + drafter does not fit 8 GB
  fully on the GPU (OOM at `-ngl 26`).
- VRAM with Q4_K_M + f16 drafter at ctx 8192: 7.65 GB used.
- Greedy outputs with the drafter differ from the baseline on 4/6 prompts (math and gsm
  match byte for byte). This is likely the usual batched-verify numerics on a quantized
  target (llama.cpp argmax flips with batch shape), not drafter-caused, but unverified.

### Already ruled out (by reading code, no run needed)

- **Target-layer convention.** The converter writes `target_layers = i+1`. The fork's staged
  capture copies `t_layer_inp[i+1]`, the input of layer i+1 = the output of layer i = HF
  `hidden_states[i+1]` (qwen3.cpp sets `t_layer_inp[il] = inpL`). Same as the fork's
  DFlash converter.
- **Context positions.** The injected rows sit at their target positions 0..P−1 and the
  anchor at P. Stale cells ≥ the anchor are trimmed (`seq_rm`) before each draft decode.
- **Noise mapping.** `ggml_mul_mat(L_gguf, z)` computes `z @ Lᵀ` with numpy-row-major L.

## Open problem and next step

τ is ~half of the paper's on every task type, independent of target quantization. Unless
upstream's numbers rely on something the DSpark protocol does differently, the draft graph
has a numerical bug.

**Next step: one instrumented comparison, not more benchmark runs.** Dump a single drafter
decode from llama.cpp:
- the injected context features
- the anchor token and its position
- the noise `z`
- the per-layer `fm_x_embed` / `fm_self_out` / `fm_cross_out` / `l_out` and the logits
  (they are named via `cb()`)

Then run the same inputs through a NumPy port of upstream `FlashMapDraftModel.draft()` (the
safetensors loader in `convert-flashmap.py` already reads the weights). The first diverging
tensor names the bug. Suspects in rough order:

1. **AdaLN modulation / time conditioning.** The baked `cond` uses float32 sinusoids, while
   upstream casts them to bf16 before `linear_1`. Also check that the chunk order
   `(scale, shift, gate)` matches the trained weights.
2. **The cross-attention mask row** (the anchor row's mask repeated). Verify that n_kv cells
   ≤ P are actually visible and that nothing else leaks in.
3. **Slot RoPE positions / Q-K norm application order** in the self-attention.
4. **The constant input** (`state_const`, the noise scale).

If the port turns out correct, the remaining explanation is the protocol (e.g. how context
features are refreshed after partial accepts) versus upstream's `spec_generate`.

## Files

- `patches/buun-llama.cpp-flashmap.patch` — the port, on buun-llama.cpp `34ef0ef` + the
  nvcc patch
- `scripts/download-flashmap.sh` — downloads (curl -4, sha256)
- `scripts/convert-flashmap.py` — drafter → GGUF with baked one-step constants
- `scripts/run-flashmap.sh` — server (DRAFT=0 baseline, DRAFT_MAX, env knobs)
- `scripts/flashmap-bench.py` — the benchmark above
