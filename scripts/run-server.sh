#!/usr/bin/env bash
#
# run-server.sh — start llama.cpp's OpenAI-compatible server for Gemma 4 26B-A4B QAT.
#
# Memory split (the whole point of --cpu-moe / -cmoe):
#   * MoE expert FFN weights  -> system RAM  (the bulk of the 26B params)
#   * Attention + embeddings  -> GPU VRAM
#   * KV cache                -> GPU VRAM    (grows with context size, -c)
#
# By default it uses the **Vulkan** backend on the NVIDIA GPU. Why not CUDA:
# the conda-forge llama.cpp is compiled for a recent CUDA (e.g. 12.9); if your
# NVIDIA driver only supports an older CUDA (e.g. 12.2 on driver 535) the CUDA
# kernels fail with "device kernel image is invalid". Vulkan uses the driver's
# own ICD and sidesteps that entirely. --cpu-moe is backend-agnostic, so the
# RAM/VRAM split is identical either way.
#
# Config (override via env vars):
#   ENV_NAME   conda env name                 (default: llamacpp)
#   MODEL      path to the .gguf              (default: ./models/.../UD-Q4_K_XL.gguf)
#   BACKEND    vulkan | cuda                  (default: vulkan)
#   CTX        context window                 (default: 32768; ~max at NCMOE=22 on 8 GB)
#   NCMOE      if set to N, keep only the first N layers' experts on CPU and put
#              the rest on the GPU (faster, uses more VRAM). Empty = all on CPU.
#   HOST/PORT  bind address                   (default: 127.0.0.1 / 8080)
#
#   Sampling defaults (server-wide; a client request can still override per-call).
#   Defaults follow unsloth's Gemma 4 recommendation (https://unsloth.ai/docs/models/gemma-4/qat):
#   TEMP       temperature                    (default: 1.0)
#   TOP_P      top-p / nucleus                (default: 0.95)
#   TOP_K      top-k                          (default: 64)
#   EXTRA_ARGS any extra llama-server flags   (e.g. "--min-p 0.01 --repeat-penalty 1.1 --seed 42")
#
# Examples:
#   bash scripts/run-server.sh                 # Vulkan, 32K ctx, recommended sampling
#   TEMP=0.7 bash scripts/run-server.sh        # more deterministic
#   CTX=65536 NCMOE=27 BACKEND=cuda bash scripts/run-server.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="${ENV_NAME:-llamacpp}"
MODEL="${MODEL:-$REPO_ROOT/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
BACKEND="${BACKEND:-vulkan}"
CTX="${CTX:-32768}"
NCMOE="${NCMOE:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CUDA_ENV="${CUDA_ENV:-llamacpp-cuda}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server}"

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Run scripts/setup.sh first."; exit 1; }

# --- pick GPU backend / device / binary ------------------------------------
DEVICE_ARGS=()
if [ "$BACKEND" = "cuda" ]; then
  # Use the locally built CUDA binary (scripts/build-llama-cuda.sh) + its env.
  [ -x "$CUDA_BIN" ] || { echo "ERROR: CUDA build not found at $CUDA_BIN"; echo "Run scripts/build-llama-cuda.sh first (or set CUDA_BIN)."; exit 1; }
  RUN=(mamba run --no-capture-output -n "$CUDA_ENV")
  SERVER_BIN="$CUDA_BIN"
  # default CUDA device (CUDA0); -ngl 99 offloads to it
  echo ">> backend: CUDA  binary: $CUDA_BIN"
else
  RUN=(mamba run --no-capture-output -n "$ENV_NAME")
  SERVER_BIN="llama-server"
  DEV="$("${RUN[@]}" llama-server --list-devices 2>/dev/null | grep -iE 'Vulkan[0-9]+: NVIDIA' | head -1 | sed -E 's/^ *([A-Za-z0-9]+):.*/\1/')"
  DEV="${DEV:-Vulkan0}"
  DEVICE_ARGS=(--device "$DEV")
  echo ">> backend: Vulkan  device: $DEV"
fi

# --- MoE offload mode -------------------------------------------------------
if [ -n "$NCMOE" ]; then
  MOE_ARGS=(--n-cpu-moe "$NCMOE")
  echo ">> MoE: experts of first $NCMOE layers on CPU, the rest on GPU"
else
  MOE_ARGS=(--cpu-moe)
  echo ">> MoE: all experts on CPU (RAM)"
fi

# --- sampling defaults (server-wide; clients may override per request) ------
SAMPLER_ARGS=(--temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K")
# shellcheck disable=SC2206  # intentional word-splitting for passthrough flags
[ -n "$EXTRA_ARGS" ] && SAMPLER_ARGS+=($EXTRA_ARGS)

echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K ${EXTRA_ARGS:+(+ $EXTRA_ARGS)}"
echo ">> context: $CTX   listening: http://$HOST:$PORT/v1"
echo

exec "${RUN[@]}" "$SERVER_BIN" \
  -m "$MODEL" \
  --alias gemma-4-26b-a4b-qat \
  "${DEVICE_ARGS[@]}" \
  -ngl 99 \
  "${MOE_ARGS[@]}" \
  --no-mmap \
  -c "$CTX" \
  -fa auto \
  --jinja \
  "${SAMPLER_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT"
