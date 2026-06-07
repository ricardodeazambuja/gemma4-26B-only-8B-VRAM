#!/usr/bin/env bash
#
# run-server.sh — start llama.cpp's OpenAI-compatible server for Gemma 4 26B-A4B QAT.
#
# Memory split (the whole point of --cpu-moe / -cmoe):
#   * MoE expert FFN weights  -> system RAM  (the bulk of the 26B params)
#   * Attention + embeddings  -> GPU VRAM
#   * KV cache                -> GPU VRAM    (grows with context size, -c)
#
# Backend is auto-detected: it uses the locally-built CUDA binary if present
# (scripts/build-llama-cuda.sh), else falls back to Vulkan. Override with BACKEND=.
# Why Vulkan is the fallback and not the prebuilt CUDA: the conda-forge llama.cpp
# is compiled for a recent CUDA (e.g. 12.9); if your NVIDIA driver only supports an
# older CUDA (e.g. 12.2 on driver 535) those CUDA kernels fail with "device kernel
# image is invalid". Vulkan uses the driver's own ICD and sidesteps that entirely.
# --cpu-moe is backend-agnostic, so the RAM/VRAM split is identical either way.
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
#   MMPROJ     path to the multimodal projector (default: models/.../mmproj-BF16.gguf)
#
# Flags (CLI args, not env vars):
#   --image    load the multimodal projector ($MMPROJ) so the server accepts
#              images. Text-only without it. (The default mmproj is vision-only;
#              see docs/TECHNICAL.md §14.) Other CLI args pass through to llama-server.
#
# Examples:
#   bash scripts/run-server.sh                 # auto backend, 32K ctx, text only
#   TEMP=0.7 bash scripts/run-server.sh        # more deterministic
#   CTX=65536 NCMOE=27 BACKEND=cuda bash scripts/run-server.sh
#   bash scripts/run-server.sh --image         # enable image input via the mmproj
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared backend banner + resolution (see scripts/_banner.sh).
source "$REPO_ROOT/scripts/_banner.sh"

ENV_NAME="${ENV_NAME:-llamacpp}"
MODEL="${MODEL:-$REPO_ROOT/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
# Auto-detect: use the locally-built CUDA binary if present, else fall back to
# Vulkan (zero-build path for non-NVIDIA / unbuilt machines). Override with BACKEND=.
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server}"
BACKEND="$(resolve_backend)"
CTX="${CTX:-32768}"
NCMOE="${NCMOE:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CUDA_ENV="${CUDA_ENV:-llamacpp-cuda}"   # CUDA_BIN already set above (line ~54)

# Multimodal projector (the vision tower). Enabled only with the --image flag.
MMPROJ="${MMPROJ:-$REPO_ROOT/models/gemma4-26b-a4b-qat/mmproj-BF16.gguf}"

# --- CLI args ---------------------------------------------------------------
#   --image   load the multimodal projector ($MMPROJ) so the server accepts
#             images. Without it the server is text-only. (The default mmproj is
#             vision-only.) Any other args pass through to llama-server.
USE_IMAGE=0
PASS_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) USE_IMAGE=1 ;;
    *) PASS_ARGS+=("$1") ;;
  esac
  shift
done

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Run scripts/setup.sh first."; exit 1; }

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
    backend_banner cuda "binary: $CUDA_BIN"
    ;;
  vulkan)
    RUN=(mamba run --no-capture-output -n "$ENV_NAME")
    SERVER_BIN="llama-server"
    # `|| true`: under `set -euo pipefail`, a no-match grep makes the whole
    # command-sub exit non-zero and would kill the script before the fallback.
    DEV="$("${RUN[@]}" llama-server --list-devices 2>/dev/null | grep -iE 'Vulkan[0-9]+: NVIDIA' | head -1 | sed -E 's/^ *([A-Za-z0-9]+):.*/\1/' || true)"
    DEV="${DEV:-Vulkan0}"
    DEVICE_ARGS=(--device "$DEV")
    backend_banner vulkan "device: $DEV  ·  build the CUDA backend for full speed"
    ;;
  cpu)
    RUN=(mamba run --no-capture-output -n "$ENV_NAME")
    SERVER_BIN="llama-server"
    DEVICE_ARGS=(--device none)
    NGL_ARGS=(-ngl 0)
    backend_banner cpu "benchmark baseline / testing only — expect ~2 tok/s"
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

# --- multimodal projector (only with --image) -------------------------------
MMPROJ_ARGS=()
if [ "$USE_IMAGE" = 1 ]; then
  if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: --image given but projector not found at:"
    echo "       $MMPROJ"
    echo "Download it (~1.2 GB, BF16):"
    echo "  curl -L -o \"$MMPROJ\" \\"
    echo "    https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-BF16.gguf"
    exit 1
  fi
  # Keep the projector on the CPU: on an 8 GB card there is no VRAM left for a
  # 1.2 GB BF16 tower next to NCMOE experts + KV. Drop --no-mmproj-offload (or
  # set it via PASS_ARGS) if you have spare VRAM and want faster image encoding.
  MMPROJ_ARGS=(--mmproj "$MMPROJ" --no-mmproj-offload)
  echo ">> multimodal: ENABLED — vision projector on CPU ($MMPROJ)"
else
  echo ">> multimodal: off (text only; pass --image to enable vision)"
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
  "${MMPROJ_ARGS[@]}" \
  --no-mmap \
  -c "$CTX" \
  -fa auto \
  --jinja \
  "${SAMPLER_ARGS[@]}" \
  "${PASS_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT"
