#!/usr/bin/env bash
#
# run-vibethinker.sh — serve a WeiboAI VibeThinker GGUF (1.5B on Qwen2.5-Math, or
# 3B on Qwen2.5-Coder) on llama.cpp's OpenAI-compatible server. Pick with SIZE.
#
# These models are small and DENSE (1.5B = 28 layers, 3B = 36 layers), so NONE of
# the repo's Gemma tricks apply: no --cpu-moe / NCMOE (not MoE), no mmproj
# (text-only), no MTP head. Both fit in 8 GB VRAM with room, so the default is
# -ngl 99 (every layer on the GPU) and they run FAST (Q8_0: 1.5B ~100 t/s).
#
# What it is (verified from the GGUF metadata + model card, see
# docs/vibethinker-eval.md):
#   * arch qwen2, native context 131072 (no RoPE scaling) — competitive MATH/CODING
#     reasoning models, NOT general chatbots
#   * long chain-of-thought INLINE in the response (ChatML, no <think> tags, no
#     separate reasoning_content channel)
#   * 1.5B default system prompt nudges "reason step by step … \boxed{}"; the 3B's
#     default is the generic "You are a helpful assistant."
#
# Sampling follows the model card: temp 0.6 (or 1.0), top-p 0.95, top-k disabled.
# The devs warn AGAINST temp 0 — greedy collapses the path exploration the RL
# training relies on. Give it a generous token budget: it thinks a lot.
#
# Config (override via env vars):
#   SIZE     1.5B | 3B                             (default: 1.5B — picks dir/file/alias)
#   THINK    1 | 0                                  (default: 1 — patched template: CoT ->
#                                                    reasoning_content, clean answer in content)
#   REASONING_BUDGET  tokens of thinking before </think> is forced  (default: unlimited)
#   QUANT    Q8_0 | Q6_K | Q5_K_M | Q4_K_M | f16   (default: Q8_0 — near-lossless)
#   NGL      transformer layers on GPU             (default: 99 = all of them)
#   CTX      context window                        (default: 40960; native max 131072)
#   NPREDICT max tokens to generate (-n)           (default: 38912; long CoT needs room)
#   KVQUANT  KV cache type (f16 default)           (use q8_0 only for very long ctx)
#   TEMP/TOP_P/TOP_K  sampling                     (default: 0.6 / 0.95 / 0 = top-k off)
#   MODEL_DIR/MODEL/ALIAS  override paths + pi alias (default: derived from SIZE)
#   HOST/PORT                                      (default: 127.0.0.1 / 8080)
#   BACKEND    cuda | vulkan                       (default: cuda if built, else vulkan)
#
# Examples:
#   bash scripts/run-vibethinker.sh                 # 1.5B, Q8_0, all on GPU, 40k ctx
#   SIZE=3B bash scripts/run-vibethinker.sh         # the 3B coder model
#   SIZE=3B TEMP=1.0 bash scripts/run-vibethinker.sh
#   QUANT=f16 bash scripts/run-vibethinker.sh       # truly lossless
#   CTX=131072 KVQUANT=q8_0 bash scripts/run-vibethinker.sh   # full native context
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# SIZE selects the model (1.5B or 3B); it drives the dir, filename, and pi alias.
SIZE="${SIZE:-1.5B}"
SIZE_LC="$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')"
QUANT="${QUANT:-Q8_0}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/vibethinker-${SIZE_LC}}"
MODEL="${MODEL:-$MODEL_DIR/VibeThinker-${SIZE}.${QUANT}.gguf}"
ALIAS="${ALIAS:-vibethinker-${SIZE_LC}}"
CTX="${CTX:-40960}"
NPREDICT="${NPREDICT:-38912}"
NGL="${NGL:-99}"
KVQUANT="${KVQUANT:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-0}"   # 0 = top-k disabled (the card's top_k=-1)
# THINK=1 (default) serves the patched chat template that makes llama.cpp recognise
# this model's <think>…</think> as a reasoning channel: the CoT then lands in
# `reasoning_content` (pi folds it) and `content` is the clean answer, and the
# REASONING_BUDGET lever below works. THINK=0 uses the GGUF's stock template (CoT
# inline in content, budget ignored — the old behaviour). See docs/vibethinker-eval.md §9.
THINK="${THINK:-1}"
THINK_TEMPLATE="${THINK_TEMPLATE:-$REPO_ROOT/config/vibethinker-${SIZE_LC}-think.jinja}"
REASONING_BUDGET="${REASONING_BUDGET:-}"   # tokens of thinking before </think> is forced; empty = unlimited

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Download it, e.g.:"; echo "  mamba run -n llamacpp hf download mradermacher/VibeThinker-${SIZE}-GGUF VibeThinker-${SIZE}.${QUANT}.gguf --local-dir $MODEL_DIR"; exit 1; }
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

# Reasoning template + budget (the §9 fix). THINK=1 swaps in the patched template so
# llama.cpp extracts <think>…</think> into reasoning_content; REASONING_BUDGET caps it.
THINK_ARGS=()
case "$THINK" in
  1|y|Y|yes|on)
    if [ -f "$THINK_TEMPLATE" ]; then
      THINK_ARGS+=(--chat-template-file "$THINK_TEMPLATE")
      [ -n "$REASONING_BUDGET" ] && THINK_ARGS+=(--reasoning-budget "$REASONING_BUDGET")
      echo ">> reasoning: ON — $THINK_TEMPLATE${REASONING_BUDGET:+  (budget $REASONING_BUDGET tok)}"
    else
      echo ">> WARNING: THINK=1 but template missing ($THINK_TEMPLATE) — using the GGUF's stock template."
    fi ;;
  *) echo ">> reasoning: OFF (stock template; CoT inline in content)" ;;
esac

echo ">> model:    $MODEL  (VibeThinker-$SIZE $QUANT, alias '$ALIAS')"
echo ">> dense:    -ngl $NGL (all layers on GPU — no --cpu-moe; this model is dense, not MoE)"
echo ">> context:  $CTX (native max 131072)   KV: ${KVQUANT:-f16} (flash-attn on)   max gen: $NPREDICT"
echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K (0=off)   listening: http://$HOST:$PORT/v1"
echo ">> NOTE: math/coding reasoner (not a general chatbot). THINK=1: CoT -> reasoning_content, clean answer in content. THINK=0: CoT inline in content."
echo

exec "${RUN[@]}" "$SERVER_BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  "${DEVICE_ARGS[@]}" \
  -ngl "$NGL" \
  --no-mmap \
  -c "$CTX" \
  -n "$NPREDICT" \
  "${KV_ARGS[@]}" \
  --jinja \
  "${THINK_ARGS[@]}" \
  --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" \
  --host "$HOST" --port "$PORT" \
  "$@"
