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
#   KVQUANT KV cache type (both K and V)    (default: q8_0; use q4_0 for ~2x context)
#   TURBO   1 = use the turboquant build + turbo3 V cache (see below)  (default: 0)
#   CTK/CTV per-side KV types (override KVQUANT; TURBO sets q8_0 / turbo3)
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
#   TURBO=1 bash scripts/run-12b-agentic.sh               # full offload via turbo3 V cache (8 GB)
#   TURBO=1 CTX=25000 bash scripts/run-12b-agentic.sh     # the author's exact recipe
#
# TurboQuant V-cache (TURBO=1) — model author's own 8 GB recipe (from @analogalok on X):
#   The win is `--cache-type-v turbo3`: V cache at ~3.5-bit (Walsh-Hadamard rotated
#   polar quant, Google KV-compression research). It shrinks the KV cache enough to
#   FULLY offload all layers (-ngl 99) + a big context on an 8 GB card (~30 tok/s on
#   the author's RTX 4060). K stays q8_0 (K is sensitive; V tolerates aggression).
#   `turbo3` is NOT in mainline llama.cpp — it needs the TheTom/llama-cpp-turboquant
#   fork. Build it once into vendor/llama-cpp-turboquant (see the ERROR hint TURBO=1
#   prints if it's missing); then TURBO=1 here uses it + sets K=q8_0 V=turbo3 + -ngl 99.
#   Without TURBO, the stock build's partial-offload defaults apply (-ngl 99 OOMs at 8 GB).
#   Verified on this RTX 2070 (sm_75): q8_0-K + turbo3-V decodes coherently for gemma4.
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

# TurboQuant V-cache mode (TURBO=1): use the TheTom/llama-cpp-turboquant build and
# its ~3.5-bit `turbo3` V cache. The codec is a runtime cache type (the GGUF is a
# plain quant — nothing special), and it shrinks KV enough to FULLY offload Q3_K_M/
# Q4_K_M (-ngl 99) on an 8 GB card. K stays at q8_0 (K is sensitive, V is not).
# Verified on this RTX 2070 (sm_75): q8_0-K + turbo3-V decodes coherently at gemma4's
# head_dim 256. turbo3 needs the turbo build + CUDA backend (not stock / Vulkan).
TURBO="${TURBO:-0}"
TURBO_BIN="$REPO_ROOT/vendor/llama-cpp-turboquant/build/bin/llama-server"
if [ "$TURBO" = 1 ]; then
  CTK="${CTK:-q8_0}"
  CTV="${CTV:-turbo3}"
else
  CTK="${CTK:-$KVQUANT}"
  CTV="${CTV:-$KVQUANT}"
fi

# Reject the dropped quant early (whatever the mode).
[ "$QUANT" = Q2_K ] && { echo "ERROR: no Q2_K in this release (failed stress-testing) — pick Q3_K_M/Q4_K_M/Q6_K/Q8_0."; exit 1; }

# Safe default -ngl per quant (leaves VRAM headroom on an 8 GB card next to the
# desktop). Bump NGL to the measured/estimated max for the top speed, at the cost
# of headroom. Q3_K_M is smaller than Q4_K_M -> likely fits more; confirm & bump.
if [ -z "${NGL:-}" ]; then
  if [ "$TURBO" = 1 ] && { [ "$QUANT" = Q3_K_M ] || [ "$QUANT" = Q4_K_M ]; }; then
    NGL=99   # turbo3 V cache frees enough VRAM to fully offload these on 8 GB
  else
    case "$QUANT" in
      Q3_K_M) NGL=44 ;;
      Q4_K_M) NGL=40 ;;
      Q6_K)   NGL=30 ;;   # weights (>9 GB) still exceed 8 GB -> partial even with turbo3
      Q8_0)   NGL=22 ;;
      *)      NGL=40 ;;
    esac
  fi
fi

[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Download it, e.g.:"; echo "  mamba run -n llamacpp hf download yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF gemma4-v2-${QUANT}.gguf --local-dir $MODEL_DIR"; exit 1; }
command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH."; exit 1; }

# Backend: prefer the locally-built CUDA binary, else conda Vulkan. In TURBO mode
# the turboquant build is required (turbo3 isn't in the stock build).
if [ "$TURBO" = 1 ]; then
  CUDA_BIN="${CUDA_BIN:-$TURBO_BIN}"
  [ -x "$CUDA_BIN" ] || { echo "ERROR: TURBO=1 needs the turboquant CUDA build at $CUDA_BIN"; echo "  build it: SRC_DIR=\$PWD/vendor/llama-cpp-turboquant LLAMA_REF=feature/turboquant-kv-cache ENV_NAME=llamacpp-cuda bash scripts/build-llama-cuda.sh"; exit 1; }
else
  CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server}"
fi
BACKEND="${BACKEND:-}"
if [ -z "$BACKEND" ]; then [ -x "$CUDA_BIN" ] && BACKEND=cuda || BACKEND=vulkan; fi
[ "$TURBO" = 1 ] && [ "$BACKEND" != cuda ] && { echo "ERROR: TURBO=1 (turbo3) requires the CUDA backend; got BACKEND=$BACKEND."; exit 1; }
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

# Memory-saving defaults for full-offload TURBO mode on an 8 GB card: the default
# n_parallel=4 + n_batch=2048 inflate the CUDA compute buffer by ~0.5 GB and OOM at
# -ngl 99. One slot + a 512 batch fit comfortably (verified: 7.76/8 GB, ~28 t/s).
# Override via NP / BATCH / UBATCH; harmless to set in non-turbo mode too.
EXTRA_ARGS=()
if [ "$TURBO" = 1 ]; then
  EXTRA_ARGS=(--parallel "${NP:-1}" -b "${BATCH:-512}" -ub "${UBATCH:-512}")
fi

echo ">> model:   $MODEL ($QUANT)"
echo ">> dense:   -ngl $NGL of 48 layers on GPU, rest on CPU/RAM (no --cpu-moe — this model is dense)"
echo ">> context: $CTX   KV cache: K=$CTK V=$CTV (flash-attn on)$([ "$TURBO" = 1 ] && echo '   [TurboQuant build]')"
echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K rep-pen=$REP_PEN   listening: http://$HOST:$PORT/v1"
echo

exec "${RUN[@]}" "$SERVER_BIN" \
  -m "$MODEL" \
  --alias gemma-4-12b-agentic \
  "${DEVICE_ARGS[@]}" \
  -ngl "$NGL" \
  --no-mmap \
  -c "$CTX" \
  -ctk "$CTK" -ctv "$CTV" -fa on \
  --jinja \
  --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --repeat-penalty "$REP_PEN" \
  "${EXTRA_ARGS[@]}" \
  --host "$HOST" --port "$PORT" \
  "$@"
