#!/usr/bin/env bash
#
# run-12b-coder.sh — serve the Gemma 4 12B Coder (fable5-composer2.5) GGUF.
#
# This model is DENSE (not MoE), so the repo's headline --cpu-moe / NCMOE trick
# does NOT apply. The VRAM lever for a dense model is -ngl (how many of the 48
# transformer layers live on the GPU; the rest run on the CPU in RAM). See
# docs/gemma4-12b-coder-eval.md for the full evaluation and per-quant fit table.
#
# Measured on this box (RTX 2070 Max-Q 8 GB, 31 GB RAM, CUDA backend), 32768 ctx,
# q8_0 KV cache:
#   Q4_K_M  -ngl 42  ->  ~17 tok/s   (RECOMMENDED — sweet spot; 44+ OOMs)
#   Q6_K    -ngl 32  ->  ~7  tok/s   (higher fidelity, much slower; 34+ OOMs)
#   Q8_0    -ngl 24  ->  ~4.5 tok/s  (near-lossless, slowest;     26+ OOMs)
#   Q2_K    -- DO NOT USE: degenerate/unusable output on this model --
#
# Config (override via env vars):
#   QUANT   Q4_K_M | Q6_K | Q8_0        (default: Q4_K_M)
#   NGL     transformer layers on GPU   (default: per-quant safe value below)
#   CTX     context window              (default: 32768)
#   KVQUANT KV cache type               (default: q8_0; use q4_0 for ~2x context)
#   PORT/HOST                           (default: 8080 / 127.0.0.1)
#   TEMP/TOP_P/TOP_K  sampling          (default: 1.0 / 0.95 / 64 — model card's rec;
#                                        for deterministic coding set TEMP=0)
#   MODEL_DIR   where the GGUFs live    (default: ./models/gemma4-12b-coder)
#   BACKEND     cuda | vulkan           (default: cuda if built, else vulkan)
#
# Examples:
#   bash scripts/run-12b-coder.sh                         # Q4_K_M, ngl 40, 32k ctx
#   NGL=42 bash scripts/run-12b-coder.sh                  # squeeze max layers (faster, tighter)
#   QUANT=Q6_K bash scripts/run-12b-coder.sh              # higher fidelity, slower
#   CTX=65536 KVQUANT=q4_0 NGL=38 bash scripts/run-12b-coder.sh   # longer context
#   TEMP=0 bash scripts/run-12b-coder.sh                  # greedy / deterministic coding
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUANT="${QUANT:-Q4_K_M}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/gemma4-12b-coder}"
MODEL="$MODEL_DIR/gemma4-coding-${QUANT}.gguf"
CTX="${CTX:-32768}"
KVQUANT="${KVQUANT:-q8_0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"

# Safe default -ngl per quant (leaves VRAM headroom on an 8 GB card next to the
# desktop). Bump NGL to the measured max (Q4_K_M 42 / Q6_K 32 / Q8_0 24) for the
# top speed quoted above, at the cost of headroom.
if [ -z "${NGL:-}" ]; then
  case "$QUANT" in
    Q4_K_M) NGL=40 ;;
    Q6_K)   NGL=30 ;;
    Q8_0)   NGL=22 ;;
    Q2_K)   echo "ERROR: Q2_K is degenerate on this model — pick Q4_K_M/Q6_K/Q8_0."; exit 1 ;;
    *)      NGL=40 ;;
  esac
fi

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Download it, e.g.:"; echo "  mamba run -n llamacpp hf download yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF gemma4-coding-${QUANT}.gguf --local-dir $MODEL_DIR"; exit 1; }
command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH."; exit 1; }

# Backend: prefer the locally-built CUDA binary, else conda Vulkan.
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server}"
BACKEND="${BACKEND:-}"
if [ -z "$BACKEND" ]; then [ -x "$CUDA_BIN" ] && BACKEND=cuda || BACKEND=vulkan; fi
case "$BACKEND" in
  cuda)
    [ -x "$CUDA_BIN" ] || { echo "ERROR: CUDA build not found at $CUDA_BIN (run scripts/build-llama-cuda.sh)"; exit 1; }
    RUN=(mamba run --no-capture-output -n "${CUDA_ENV:-llamacpp-cuda}")
    SERVER_BIN="$CUDA_BIN"; DEVICE_ARGS=(--device CUDA0)
    echo "🟢 backend: CUDA ($CUDA_BIN)" ;;
  vulkan)
    RUN=(mamba run --no-capture-output -n "${ENV_NAME:-llamacpp}")
    SERVER_BIN="llama-server"
    DEV="$("${RUN[@]}" llama-server --list-devices 2>/dev/null | grep -iE 'Vulkan[0-9]+: NVIDIA' | head -1 | sed -E 's/^ *([A-Za-z0-9]+):.*/\1/' || true)"
    DEVICE_ARGS=(--device "${DEV:-Vulkan0}")
    echo "🟡 backend: Vulkan (${DEV:-Vulkan0}) — build CUDA for ~5x speed" ;;
  *) echo "ERROR: unknown BACKEND='$BACKEND'"; exit 1 ;;
esac

echo ">> model:   $MODEL ($QUANT)"
echo ">> dense:   -ngl $NGL of 48 layers on GPU, rest on CPU/RAM (no --cpu-moe — this model is dense)"
echo ">> context: $CTX   KV cache: $KVQUANT (flash-attn on)"
echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K   listening: http://$HOST:$PORT/v1"
echo

exec "${RUN[@]}" "$SERVER_BIN" \
  -m "$MODEL" \
  --alias gemma-4-12b-coder \
  "${DEVICE_ARGS[@]}" \
  -ngl "$NGL" \
  --no-mmap \
  -c "$CTX" \
  -ctk "$KVQUANT" -ctv "$KVQUANT" -fa on \
  --jinja \
  --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
  --host "$HOST" --port "$PORT" \
  "$@"
