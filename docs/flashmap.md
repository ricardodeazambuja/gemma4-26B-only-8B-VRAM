# FlashMap (flow-map speculative decoding) on 8 GB — investigation log

Upstream: <https://github.com/al-jshen/flashmap> (MIT, created 2026-10-13), paper arXiv:2610.24365.
Branch: `flashmap_on_8gb_vram`.

## What it is

A **drafter** for lossless speculative decoding, not a standalone model. It proposes a
whole block of K=16 tokens in one forward pass (DFlash-style), but instead of predicting
each slot independently it applies a *flow map* that jumps from Gaussian noise to the block
in 1 (or 2–4) steps, so the slots can depend on each other. Claimed up to 6x over the target
alone on Qwen3-8B.

Released drafters (bf16, Apache-2.0):

| Drafter | Target | Params | File |
|---|---|---|---|
| j-shen/FlashMap-Qwen3-8B | Qwen/Qwen3-8B | 500M | ~0.95 GB |
| j-shen/FlashMap-Qwen3-4B | Qwen/Qwen3-4B | 475M | ~0.95 GB |

Serving upstream: HF `transformers` (`draft.spec_generate`), SGLang plugin
(`--speculative-algorithm DSPARK`), vLLM plugin. **No llama.cpp support anywhere.**

## Architecture (from `modeling_flashmap.py`, Qwen3-8B drafter config)

- Context path: target hidden states at layers `[1, 9, 17, 25, 33]` concatenated
  (5×4096) → `fc` → RMSNorm; the anchor token's target embedding → `anchor_proj` → RMSNorm.
  Both become per-layer cross-attention K/V (cached, RoPE'd). Same idea as DFlash.
- DiT body: 6 layers, dim 2048, 16 heads, each layer = AdaLN-zero self-attn over the 16
  block slots (non-causal) + AdaLN cross-attn to the context + AdaLN GELU MLP (biased).
- Time conditioning: sinusoidal embeddings of jump start `s` and end `t` → per-norm
  `(scale, shift, gate)`. For a fixed `num_steps` grid these are **constants** (one-step:
  `s=0, t=1`), so they can be precomputed and baked in as tensors.
- Input for a one-step draft: `RMSNorm(uniform-posterior embedding) + gate·RMSNorm(z @ Lᵀ)`
  with `z ~ N(0, I_H)` and `L` the Cholesky factor of `EᵀE` (E = target embedding). Also
  a precomputable constant plus an H-dim noise vector per slot.
- Output: `head` (2048 → 4096), then the **target's** `lm_head`. Uses the target's embedding
  and lm_head, so no own vocab matrices.

## Differences from what the vendored forks already have

`vendor/buun-llama.cpp` and `vendor/ik_llama.cpp` implement DFlash draft archs
(`LLM_ARCH_DFLASH_DRAFT`, `--spec-type draft-dflash`) including the target-layer hidden-state
capture (`dflash.target_layer_ids`). buun's `draft-dspark` is DeepSeek-V4 DSpark only, unrelated
to FlashMap despite SGLang reusing the DSPARK algorithm name.

A llama.cpp port would reuse the hidden-state capture/plumbing and need a new draft graph:
LayerNorm with baked modulations, gated residuals, a cross-attention to cached context K/V,
a biased GELU MLP, the head projection into the target's lm_head, and the noise input.

## 8 GB budget (RTX 2070 Max-Q, sm_75, no bf16 tensor cores)

- HF path: Qwen3-8B bf16 (~16 GB) and even Qwen3-4B bf16 (~8 GB) do not fit. Would need
  a 4-bit target (bitsandbytes) in fp16. The output stays lossless with respect to the
  *quantized* target. The drafter was trained on bf16 features, so acceptance will drop by
  an unknown amount.
- llama.cpp path: Qwen3-8B Q4_K_M (~5 GB) + drafter f16/q8_0 (~0.5–1 GB) + KV should fit.
- No ready Python env: `qwen-repro-hf` / `local` lack a working torch+transformers stack.

## Status

Investigation only; nothing measured yet.
