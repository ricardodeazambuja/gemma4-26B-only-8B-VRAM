#!/usr/bin/env bash
# Launch the diffusiongemma OpenAI shim (llama-diffusion-cli behind HTTP :8082).
#
# Prereqs: vendor/llama.cpp-diffusion built with the JSONL patch (see
# docs/DIFFUSION.md in this branch), and the Q4_K_M GGUF downloaded.
# Stop the regular llama-server first — 31 GB RAM does not fit both models.
#
# Overrides: DGEMMA_MODEL, DGEMMA_PORT, DGEMMA_ARGS, NCMOE, CTX
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BIN="${DGEMMA_BIN:-$ROOT/vendor/llama.cpp-diffusion/build/bin/llama-diffusion-cli}"
# Model lives in the MAIN checkout's models/ (shared across worktrees).
MODEL="${DGEMMA_MODEL:-$(ls "$ROOT"/../Gemma4/models/diffusiongemma-26b-a4b/*Q4_K_M*.gguf 2>/dev/null | head -1)}"
# 4096 default: this arch is MHA (16 heads x 512, no GQA), so full-attn KV is
# 0.92 MiB/token-layer — ctx 8192 could cost up to 7.5 GiB KV worst-case.
CTX="${CTX:-4096}"
# --cpu-moe = all experts in host RAM (safe default for 8 GB VRAM).
# Set NCMOE to a number to put (layers - NCMOE) expert layers on the GPU instead,
# mirroring run-server.sh tuning.
if [ -n "${NCMOE:-}" ]; then
  SPLIT_ARGS="--n-cpu-moe $NCMOE"
  echo ">> MoE: experts of first $NCMOE layers on CPU, the rest on GPU"
else
  SPLIT_ARGS="--cpu-moe"
  echo ">> MoE: all experts on CPU (RAM)"
fi

# --- Flash Attention and KV Cache Quantization ---
KVQUANT="${KVQUANT:-}"
FA="${FA:-on}"
KV_ARGS=()

if [ -n "$KVQUANT" ] && [ "$KVQUANT" != "f16" ]; then
  KV_ARGS=(-ctk "$KVQUANT" -ctv "$KVQUANT")
  FA="on"
  echo ">> KV-cache: K and V quantized to $KVQUANT"
fi

if [ "$FA" = "on" ]; then
  KV_ARGS+=(-fa on)
  echo ">> Attention: Flash Attention ENABLED"
else
  KV_ARGS+=(-fa auto)
fi

[ -x "$BIN" ]   || { echo "ERROR: $BIN missing — build llama-diffusion-cli first." >&2; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: diffusiongemma GGUF not found — download it first." >&2; exit 1; }

export DGEMMA_BIN="$BIN"
export DGEMMA_MODEL="$MODEL"

# -n 512: n_ubatch = n + 2048; at -n 1024 the compute buffer is 3.14 GB,
# which OOMs when combined with offloaded experts on an 8 GB card.
# Dropping to -n 512 halves the compute buffer and enables NCMOE offloading.
# Enable device-resident sampling and reduction for speed.
EXTRA_ARGS="${KV_ARGS[*]} --diffusion-gpu-sampling on --diffusion-gpu-sample-reduce on"
export DGEMMA_ARGS="${DGEMMA_ARGS:--ngl 99 $SPLIT_ARGS -c $CTX -n 512 --temp 0.0 $EXTRA_ARGS}"
export DGEMMA_PORT="${DGEMMA_PORT:-8082}"

echo ">> binary : $BIN"
echo ">> model  : $MODEL"
echo ">> args   : $DGEMMA_ARGS"
echo ">> port   : $DGEMMA_PORT"
exec node "$ROOT/scripts/diffusion-shim.mjs"
