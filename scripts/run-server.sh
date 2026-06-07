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
#   BACKEND    cuda | vulkan | cpu            (default: auto — cuda if built, else vulkan)
#                                             cpu = no GPU offload (slow; benchmark baseline)
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
# Auto-detect: use the locally-built CUDA binary if present, else fall back to
# Vulkan (zero-build path for non-NVIDIA / unbuilt machines). Override with BACKEND=.
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server}"
BACKEND="${BACKEND:-$( [ -x "$CUDA_BIN" ] && echo cuda || echo vulkan )}"
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

# --- splashy banner (color only when stdout is a real terminal) -------------
if [ -t 1 ]; then
  BOLD=$'\e[1m'; RST=$'\e[0m'
  FG_GREEN=$'\e[1;32m'; FG_YELLOW=$'\e[1;33m'; FG_RED=$'\e[1;31m'
else
  BOLD=; RST=; FG_GREEN=; FG_YELLOW=; FG_RED=
fi

# splash <color> <icon> <title> <subtitle>
# Full-width colored rules with no right border, so multi-byte emoji can't
# misalign the box (the classic ANSI box-drawing pitfall).
splash() {
  local col="$1" icon="$2" title="$3" sub="$4"
  local rule='════════════════════════════════════════════════════════════'
  printf '\n%s%s%s\n' "$col" "$rule" "$RST"
  printf '%s  %s  %s%s\n' "$col" "$icon" "$title" "$RST"
  [ -n "$sub" ] && printf '%s     %s%s\n' "$col" "$sub" "$RST"
  printf '%s%s%s\n\n' "$col" "$rule" "$RST"
}

# --- pick backend / device / binary ----------------------------------------
DEVICE_ARGS=()
NGL_ARGS=(-ngl 99)
case "$BACKEND" in
  cuda)
    # Locally built CUDA binary (scripts/build-llama-cuda.sh) + its env.
    [ -x "$CUDA_BIN" ] || { echo "ERROR: CUDA build not found at $CUDA_BIN"; echo "Run scripts/build-llama-cuda.sh first (or set CUDA_BIN)."; exit 1; }
    RUN=(mamba run --no-capture-output -n "$CUDA_ENV")
    SERVER_BIN="$CUDA_BIN"
    # default CUDA device (CUDA0); -ngl 99 offloads to it
    splash "$FG_GREEN" "🟢" "BACKEND: CUDA  —  GPU accelerated (fast path)" "binary: $CUDA_BIN"
    ;;
  vulkan)
    RUN=(mamba run --no-capture-output -n "$ENV_NAME")
    SERVER_BIN="llama-server"
    DEV="$("${RUN[@]}" llama-server --list-devices 2>/dev/null | grep -iE 'Vulkan[0-9]+: NVIDIA' | head -1 | sed -E 's/^ *([A-Za-z0-9]+):.*/\1/')"
    DEV="${DEV:-Vulkan0}"
    DEVICE_ARGS=(--device "$DEV")
    splash "$FG_YELLOW" "🟡" "BACKEND: VULKAN  —  GPU (slow MoE path, ~5x under CUDA)" "device: $DEV  ·  build the CUDA backend for full speed"
    ;;
  cpu)
    RUN=(mamba run --no-capture-output -n "$ENV_NAME")
    SERVER_BIN="llama-server"
    DEVICE_ARGS=(--device none)
    NGL_ARGS=(-ngl 0)
    splash "$FG_RED" "🔴" "BACKEND: CPU ONLY  —  no GPU offload (very slow)" "benchmark baseline / testing only — expect ~2 tok/s"
    ;;
  *)
    echo "ERROR: unknown BACKEND='$BACKEND' (use cuda | vulkan | cpu)"; exit 1
    ;;
esac

# --- MoE offload mode -------------------------------------------------------
if [ "$BACKEND" = "cpu" ]; then
  MOE_ARGS=()
  echo ">> MoE: n/a — CPU-only run, all weights on CPU"
elif [ -n "$NCMOE" ]; then
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
  "${NGL_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  --no-mmap \
  -c "$CTX" \
  -fa auto \
  --jinja \
  "${SAMPLER_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT"
