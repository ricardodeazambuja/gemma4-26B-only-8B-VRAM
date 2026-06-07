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
#   CTX        context window                 (default: 16384)
#   NCMOE      if set to N, keep only the first N layers' experts on CPU and put
#              the rest on the GPU (faster, uses more VRAM). Empty = all on CPU.
#   HOST/PORT  bind address                   (default: 127.0.0.1 / 8080)
#
# Examples:
#   bash scripts/run-server.sh                 # all experts on CPU, Vulkan, 16k ctx
#   CTX=32768 bash scripts/run-server.sh       # bigger context
#   NCMOE=20 bash scripts/run-server.sh        # offload some experts to GPU for speed
#   BACKEND=cuda bash scripts/run-server.sh    # if your driver matches the CUDA build
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="${ENV_NAME:-llamacpp}"
MODEL="${MODEL:-$REPO_ROOT/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
BACKEND="${BACKEND:-vulkan}"
CTX="${CTX:-16384}"
NCMOE="${NCMOE:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Run scripts/setup.sh first."; exit 1; }

RUN=(mamba run --no-capture-output -n "$ENV_NAME")

# --- pick GPU backend / device ---------------------------------------------
DEVICE_ARGS=()
if [ "$BACKEND" = "vulkan" ]; then
  DEV="$("${RUN[@]}" llama-server --list-devices 2>/dev/null | grep -iE 'Vulkan[0-9]+: NVIDIA' | head -1 | sed -E 's/^ *([A-Za-z0-9]+):.*/\1/')"
  DEV="${DEV:-Vulkan0}"
  DEVICE_ARGS=(--device "$DEV")
  echo ">> backend: Vulkan  device: $DEV"
else
  echo ">> backend: CUDA (ensure your driver supports the build's CUDA version)"
fi

# --- MoE offload mode -------------------------------------------------------
if [ -n "$NCMOE" ]; then
  MOE_ARGS=(--n-cpu-moe "$NCMOE")
  echo ">> MoE: experts of first $NCMOE layers on CPU, the rest on GPU"
else
  MOE_ARGS=(--cpu-moe)
  echo ">> MoE: all experts on CPU (RAM)"
fi

echo ">> context: $CTX   listening: http://$HOST:$PORT/v1"
echo

exec "${RUN[@]}" llama-server \
  -m "$MODEL" \
  --alias gemma-4-26b-a4b-qat \
  "${DEVICE_ARGS[@]}" \
  -ngl 99 \
  "${MOE_ARGS[@]}" \
  --no-mmap \
  -c "$CTX" \
  -fa auto \
  --jinja \
  --host "$HOST" \
  --port "$PORT"
