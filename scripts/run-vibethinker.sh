#!/usr/bin/env bash
#
# run-vibethinker.sh — serve WeiboAI's VibeThinker-1.5B (a Qwen2.5-Math-1.5B
# reasoning fine-tune) GGUF on llama.cpp's OpenAI-compatible server.
#
# This model is small and DENSE (28 transformer layers, 1.5B params), so NONE of
# the repo's Gemma tricks apply: no --cpu-moe / NCMOE (it isn't MoE), no mmproj
# (text-only), no MTP head. The whole model fits in 8 GB VRAM many times over, so
# the default is -ngl 99 (every layer on the GPU) and it runs FAST.
#
# What it is (verified from the GGUF metadata + model card, see
# docs/vibethinker-eval.md):
#   * arch qwen2, 28 layers, native context 131072 (rope freq_base 640000, no scaling)
#   * a MATH / COMPETITIVE-CODING reasoning model — NOT a general chatbot
#   * long chain-of-thought INLINE in the response (ChatML, no <think> tags, no
#     separate reasoning_content channel); the final answer comes in \boxed{}
#   * default system prompt (when you send none): "Please reason step by step,
#     and put your final answer within \boxed{}."
#
# Sampling follows the model card: temp 0.6 (or 1.0), top-p 0.95, top-k disabled.
# The devs warn AGAINST temp 0 — greedy collapses the path exploration the RL
# training relies on. Give it a generous token budget: it thinks a lot.
#
# Config (override via env vars):
#   QUANT    Q8_0 | Q6_K | Q5_K_M | Q4_K_M | f16   (default: Q8_0 — near-lossless, ~1.8 GB)
#   NGL      transformer layers on GPU             (default: 99 = all of them)
#   CTX      context window                        (default: 40960; native max 131072)
#   NPREDICT max tokens to generate (-n)           (default: 38912; long CoT needs room)
#   KVQUANT  KV cache type (f16 default)           (use q8_0 only for very long ctx)
#   TEMP/TOP_P/TOP_K  sampling                     (default: 0.6 / 0.95 / 0 = top-k off)
#   MODEL_DIR  where the GGUFs live                (default: ./models/vibethinker-1.5b)
#   HOST/PORT                                      (default: 127.0.0.1 / 8080)
#   BACKEND    cuda | vulkan                       (default: cuda if built, else vulkan)
#
# Examples:
#   bash scripts/run-vibethinker.sh                 # Q8_0, all on GPU, 40k ctx, temp 0.6
#   TEMP=1.0 bash scripts/run-vibethinker.sh        # the card's other recommended temp
#   QUANT=f16 bash scripts/run-vibethinker.sh       # truly lossless (~3 GB, still fits)
#   CTX=131072 KVQUANT=q8_0 bash scripts/run-vibethinker.sh   # full native context
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUANT="${QUANT:-Q8_0}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/vibethinker-1.5b}"
MODEL="${MODEL:-$MODEL_DIR/VibeThinker-1.5B.${QUANT}.gguf}"
CTX="${CTX:-40960}"
NPREDICT="${NPREDICT:-38912}"
NGL="${NGL:-99}"
KVQUANT="${KVQUANT:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"   # 0 = top-k disabled (the card's top_k=-1)

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Download it, e.g.:"; echo "  mamba run -n llamacpp hf download mradermacher/VibeThinker-1.5B-GGUF VibeThinker-1.5B.${QUANT}.gguf --local-dir $MODEL_DIR"; exit 1; }
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
    echo "🟡 backend: Vulkan (${DEV:-Vulkan0})" ;;
  *) echo "ERROR: unknown BACKEND='$BACKEND'"; exit 1 ;;
esac

# KV-cache quant is optional here (the model is tiny); only wire flags if asked.
KV_ARGS=()
if [ -n "$KVQUANT" ]; then KV_ARGS=(-ctk "$KVQUANT" -ctv "$KVQUANT" -fa on); else KV_ARGS=(-fa on); fi

echo ">> model:    $MODEL ($QUANT)"
echo ">> dense:    -ngl $NGL (all 28 layers on GPU — no --cpu-moe; this model is dense, not MoE)"
echo ">> context:  $CTX (native max 131072)   KV: ${KVQUANT:-f16} (flash-attn on)   max gen: $NPREDICT"
echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K (0=off)   listening: http://$HOST:$PORT/v1"
echo ">> NOTE: math/coding reasoner. CoT is inline in 'content'; final answer in \\boxed{}. Not a chatbot."
echo

exec "${RUN[@]}" "$SERVER_BIN" \
  -m "$MODEL" \
  --alias vibethinker-1.5b \
  "${DEVICE_ARGS[@]}" \
  -ngl "$NGL" \
  --no-mmap \
  -c "$CTX" \
  -n "$NPREDICT" \
  "${KV_ARGS[@]}" \
  --jinja \
  --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
  --host "$HOST" --port "$PORT" \
  "$@"
