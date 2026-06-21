#!/usr/bin/env bash
#
# run-12b-agentic.sh — serve the Gemma 4 12B v2 "Coding + Agentic" GGUF
# (yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF).
#
# This is the agentic successor to the v1 coder (see scripts/run-12b-coder.sh and
# docs/gemma4-12b-coder-eval.md). Same family: a DENSE 48-layer gemma4_unified
# model, so the repo's headline --cpu-moe / NCMOE trick does NOT apply — the VRAM
# lever for a dense model is -ngl (how many of the 48 transformer layers live on
# the GPU; the rest run on the CPU in RAM). Needs a RECENT llama.cpp (gemma4_unified
# arch) and --jinja for Gemma 4's native thinking + tool-call protocol.
#
# Quants in the repo (no Q2_K this release — it failed stress-testing upstream):
#   Q3_K_M  5.7 GB   smallest reliable; good for an 8 GB card
#   Q4_K_M  6.87 GB  RECOMMENDED sweet spot
#   Q6_K    9.11 GB  near-lossless
#   Q8_0    11.8 GB  basically full quality
#
# -ngl defaults below are INHERITED from the v1 coder's measured fits (same arch +
# param count + 32K/q8_0 KV): Q4_K_M ~42 max, Q6_K ~32, Q8_0 ~24. Q3_K_M is new and
# smaller than Q4_K_M, so it should fit MORE layers — the default is an estimate,
# confirm with nvidia-smi and bump NGL until just before OOM.
#
# Config (override via env vars):
#   QUANT   Q3_K_M | Q4_K_M | Q6_K | Q8_0   (default: Q4_K_M)
#   NGL     transformer layers on GPU       (default: per-quant safe value below)
#   CTX     context window                  (default: 32768)
#   KVQUANT KV cache type                   (default: q8_0; use q4_0 for ~2x context)
#   PORT/HOST                               (default: 8080 / 127.0.0.1)
#   TEMP/TOP_P/TOP_K  sampling   (default: 1.0 / 0.95 / 64 — model card's rec;
#                                 for deterministic coding set TEMP=0)
#   REP_PEN repetition penalty   (default: 1.1 — the card's fix for "0000…" garble;
#                                 set 1.0 to disable)
#   MODEL_DIR   where the GGUFs live        (default: ./models/gemma4-12b-agentic)
#   BACKEND     cuda | vulkan               (default: cuda if built, else vulkan)
#
# Examples:
#   bash scripts/run-12b-agentic.sh                       # Q4_K_M, ngl 40, 32k ctx
#   QUANT=Q3_K_M bash scripts/run-12b-agentic.sh          # smallest reliable quant
#   NGL=42 bash scripts/run-12b-agentic.sh                # squeeze max layers (faster, tighter)
#   CTX=65536 KVQUANT=q4_0 NGL=38 bash scripts/run-12b-agentic.sh   # longer context
#   TEMP=0 bash scripts/run-12b-agentic.sh                # greedy / deterministic coding
#
# TurboQuant V-cache (model author's own 8 GB recipe, from @analogalok on X):
#   llama-server -m gemma4-v2-Q4_K_M.gguf -ngl 99 -c 25000 \
#     --cache-type-k q8_0 --cache-type-v turbo3 --port 8080
#   The trick is `--cache-type-v turbo3` — V cache at ~3-bit (Walsh-Hadamard rotated
#   polar quant, Google KV-compression research). It shrinks the KV cache enough to
#   FULLY offload all 48 layers (-ngl 99) + 25K ctx on an 8 GB card at ~30 tok/s.
#   `turbo3` is NOT in mainline llama.cpp — needs the TheTom/llama-cpp-turboquant
#   fork, which is NOT what this repo builds. With our stock CUDA/Vulkan build, use
#   the partial-offload defaults below (KVQUANT=q8_0/q4_0); -ngl 99 will OOM on 8 GB.
#
# Speculative decoding (MTP draft): the repo ships MTP/ drafts. See the model card —
# only llama.cpp b9553 (commit 9e3b928fd) loads the gemma4-assistant draft cleanly;
# newer builds crash. Not wired into this script; run llama-server manually with
# --model-draft … --spec-type draft-mtp if you want to try it.
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUANT="${QUANT:-Q4_K_M}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/gemma4-12b-agentic}"
MODEL="$MODEL_DIR/gemma4-v2-${QUANT}.gguf"
CTX="${CTX:-32768}"
KVQUANT="${KVQUANT:-q8_0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-64}"
REP_PEN="${REP_PEN:-1.1}"

# Safe default -ngl per quant (leaves VRAM headroom on an 8 GB card next to the
# desktop). Bump NGL to the measured/estimated max for the top speed, at the cost
# of headroom. Q3_K_M is smaller than Q4_K_M -> likely fits more; confirm & bump.
if [ -z "${NGL:-}" ]; then
  case "$QUANT" in
    Q3_K_M) NGL=44 ;;
    Q4_K_M) NGL=40 ;;
    Q6_K)   NGL=30 ;;
    Q8_0)   NGL=22 ;;
    Q2_K)   echo "ERROR: no Q2_K in this release (failed stress-testing) — pick Q3_K_M/Q4_K_M/Q6_K/Q8_0."; exit 1 ;;
    *)      NGL=40 ;;
  esac
fi

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Download it, e.g.:"; echo "  mamba run -n llamacpp hf download yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF gemma4-v2-${QUANT}.gguf --local-dir $MODEL_DIR"; exit 1; }
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
echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K rep-pen=$REP_PEN   listening: http://$HOST:$PORT/v1"
echo

exec "${RUN[@]}" "$SERVER_BIN" \
  -m "$MODEL" \
  --alias gemma-4-12b-agentic \
  "${DEVICE_ARGS[@]}" \
  -ngl "$NGL" \
  --no-mmap \
  -c "$CTX" \
  -ctk "$KVQUANT" -ctv "$KVQUANT" -fa on \
  --jinja \
  --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --repeat-penalty "$REP_PEN" \
  --host "$HOST" --port "$PORT" \
  "$@"
