#!/usr/bin/env bash
#
# run-qwen.sh — serve one of the Qwen bake-off candidates (qwen_on_8gb_vram)
# with llama.cpp's OpenAI-compatible server, on the same port pi already uses.
#
# Pick the model with QMODEL (download first: scripts/download-qwen-bakeoff.sh):
#   qwen36   Qwen3.6-35B-A3B UD-IQ4_XS        MoE, 3B active -> experts in RAM (--cpu-moe)
#   qwen38   Qwen3.8-27B GSQ-RCO IQ3_XXS      DENSE          -> split by layers (-ngl)
#   distill  Qwen3.8-35B-A3B-Distill IQ4_XS   MoE, 3B active -> experts in RAM (--cpu-moe)
#   bonsai2  Ternary-Bonsai-2-27B PTQ1_0      DENSE, 5.95 GB -> all on GPU (-ngl 99), PrismML fork
#
# Both MoE candidates are Qwen3.6-class hybrids (30 Gated DeltaNet + 10 full-attention
# layers), so the KV cache is small: 128K ctx at q8_0 is ~1.4 GB. Qwen recommends at
# least 128K for thinking, so that is the MoE default. The dense 27B has 16
# full-attention layers with 2x the KV heads (~1.1 GB at 32K, q8_0) and has to share
# the 8 GB card with its weights, so it defaults to 32K.
#
# Config (override via env vars):
#   QMODEL     qwen36 | qwen38 | distill | bonsai2     (default: qwen36)
#   MODEL      path to the .gguf                       (default: per QMODEL)
#   MMPROJ     vision projector                        (default: per QMODEL; used with --image)
#   CTX        context window                          (default: 131072 MoE / 32768 qwen38 / 16384 bonsai2)
#   NCMOE      MoE only: keep only the first N layers' experts on CPU, rest on GPU
#              (faster, more VRAM). NCMOE= (empty) = all experts on CPU.
#              (default: measured fit at 128K — qwen36 32, distill 34; the distill's
#              experts are bigger and OOM at 32. See docs/qwen-bakeoff.md)
#   NGL        dense only: transformer layers on GPU   (default: qwen38 30 of 64 — 36 OOMs
#              on the first request at 32K ctx, see docs/qwen-bakeoff.md; bonsai2 99 = all)
#   KVQUANT    KV-cache type, both K and V             (default: q8_0; bonsai2 f16 — the fork's
#              q8_0 flash-attention kernel has no Turing code)
#   MTP        dense only: 1 = speculative decoding with the GGUF's embedded MTP head
#              (measured SLOWER here: 2.0 vs 2.5 tok/s)  (default: 0)
#   TEMP / TOP_P / TOP_K / MIN_P / PRESENCE_PENALTY  sampling  (default: per QMODEL, from the
#              model cards: qwen38 1.0/0.95/20/0, qwen36 + distill 0.6/0.95/20/0, the
#              "precise coding" thinking preset; bonsai2 1.0/0.95/20, min-p 0.05 per its card)
#   NP / BATCH / UBATCH  slots / batch sizes           (default: 1 / 512 / 512)
#   EXTRA_ARGS any extra llama-server flags
#   HOST/PORT  bind address                            (default: 127.0.0.1 / 8080)
#   CUDA_BIN / CUDA_ENV  CUDA build + its env          (default: vendor/llama.cpp — bonsai2:
#              vendor/prism-llama.cpp — and env llamacpp-cuda)
#
# Flags (CLI args, not env vars):
#   --image    load the vision projector ($MMPROJ) on the CPU. Other args pass through.
#
# Examples:
#   bash scripts/run-qwen.sh                           # Qwen3.6-35B-A3B, 128K ctx
#   QMODEL=distill bash scripts/run-qwen.sh            # the 3.8-distilled 35B-A3B
#   NCMOE= bash scripts/run-qwen.sh                    # all experts on CPU (least VRAM)
#   QMODEL=qwen38 NGL=40 CTX=16384 bash scripts/run-qwen.sh
#   QMODEL=qwen38 MTP=1 bash scripts/run-qwen.sh       # try the MTP head
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QMODEL="${QMODEL:-qwen36}"
case "$QMODEL" in
  qwen36)
    KIND=moe
    _dir="$REPO_ROOT/models/qwen36-35b-a3b"
    _model="$_dir/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
    _mmproj="$_dir/mmproj-F16.gguf"
    ALIAS=qwen3.6-35b-a3b
    _ngl=30; _ctx=; _minp=0; _bin=
    _ncmoe=32
    _temp=0.6 ;;
  qwen38)
    KIND=dense
    _dir="$REPO_ROOT/models/qwen38-27b-gsq"
    _model="$_dir/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
    _mmproj="$_dir/mmproj-Qwen3.8-27B-BF16.gguf"
    ALIAS=qwen3.8-27b
    _ngl=30; _ctx=; _minp=0; _bin=
    _ncmoe=
    _temp=1.0 ;;
  distill)
    KIND=moe
    _dir="$REPO_ROOT/models/qwen38-35b-a3b-distill"
    _model="$_dir/Qwen3.8-35B-A3B-IQ4_XS.gguf"
    _mmproj="$_dir/mmproj-Qwen3.8-35B-A3B-F16.gguf"
    ALIAS=qwen3.8-35b-a3b-distill
    _ngl=30; _ctx=; _minp=0; _bin=
    _ncmoe=34
    _temp=0.6 ;;
  bonsai2)
    # Qwen3.8-27B in ternary weights (5.95 GB) — fits fully on an 8 GB GPU. Needs the
    # PrismML llama.cpp fork (stock llama.cpp rejects PTQ1_0). See docs/bonsai2-on-8gb.md.
    KIND=dense
    _dir="$REPO_ROOT/models/bonsai2-27b"
    _model="$_dir/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    _mmproj="$_dir/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
    ALIAS=bonsai2-27b
    _ngl=99; _ctx=16384; _minp=0.05
    # f16 KV: the fork's flash-attention kernel for a q8_0 KV has no sm_75 (Turing) code and
    # aborts on the first request ("no device code compatible with CUDA arch 750").
    _kv=f16
    _bin="$REPO_ROOT/vendor/prism-llama.cpp/build/bin/llama-server"
    _ncmoe=
    _temp=1.0 ;;
  *)
    echo "ERROR: unknown QMODEL='$QMODEL' (use qwen36 | qwen38 | distill | bonsai2)"; exit 1 ;;
esac

MODEL="${MODEL:-$_model}"
MMPROJ="${MMPROJ:-$_mmproj}"
if [ "$KIND" = moe ]; then CTX="${CTX:-131072}"; else CTX="${CTX:-${_ctx:-32768}}"; fi
NCMOE="${NCMOE-$_ncmoe}"   # unset -> per-model default; set-but-empty -> all on CPU
NGL="${NGL:-$_ngl}"
KVQUANT="${KVQUANT:-${_kv:-q8_0}}"
MTP="${MTP:-0}"
TEMP="${TEMP:-$_temp}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-$_minp}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
NP="${NP:-1}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-512}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CUDA_BIN="${CUDA_BIN:-${_bin:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server}}"
CUDA_ENV="${CUDA_ENV:-llamacpp-cuda}"

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
[ -x "$CUDA_BIN" ] || { echo "ERROR: CUDA build not found at $CUDA_BIN"; echo "Run scripts/build-llama-cuda.sh first (or set CUDA_BIN)."; exit 1; }
[ -f "$MODEL" ] || { echo "ERROR: model not found: $MODEL"; echo "Run: bash scripts/download-qwen-bakeoff.sh $QMODEL"; exit 1; }
RUN=(mamba run --no-capture-output -n "$CUDA_ENV")

echo ">> model: $QMODEL ($KIND) — $MODEL"

# --- weight placement -------------------------------------------------------
if [ "$KIND" = moe ]; then
  OFFLOAD_ARGS=(-ngl 99)
  if [ -n "$NCMOE" ]; then
    OFFLOAD_ARGS+=(--n-cpu-moe "$NCMOE")
    echo ">> MoE: experts of first $NCMOE layers on CPU, the rest on GPU"
  else
    OFFLOAD_ARGS+=(--cpu-moe)
    echo ">> MoE: all experts on CPU (RAM)"
  fi
else
  OFFLOAD_ARGS=(-ngl "$NGL")
  echo ">> dense: $NGL layers on GPU, the rest on CPU"
fi

# --- speculative decoding (dense only) --------------------------------------
SPEC_ARGS=()
if [ "$MTP" = 1 ]; then
  if [ "$KIND" = dense ]; then
    SPEC_ARGS=(--spec-type draft-mtp --spec-draft-p-min 0)
    echo ">> MTP: speculative decoding with the embedded MTP head (p_min forced to 0)"
  else
    echo ">> MTP: ignored — only the qwen38 build ships an MTP head"
  fi
fi

# --- multimodal projector (only with --image) -------------------------------
MMPROJ_ARGS=()
if [ "$USE_IMAGE" = 1 ]; then
  [ -f "$MMPROJ" ] || { echo "ERROR: --image given but projector not found: $MMPROJ"; exit 1; }
  # CPU-side: no VRAM to spare for a ~0.9 GB tower on an 8 GB card.
  MMPROJ_ARGS=(--mmproj "$MMPROJ" --no-mmproj-offload)
  echo ">> multimodal: ENABLED — vision projector on CPU"
else
  echo ">> multimodal: off (text only; pass --image to enable vision)"
fi

# --- KV cache ---------------------------------------------------------------
KV_ARGS=()
KVQUANT="${KVQUANT,,}"
case "$KVQUANT" in
  ""|f16) : ;;
  f32|bf16|q8_0|q5_1|q5_0|q4_1|q4_0|iq4_nl)
    KV_ARGS=(-ctk "$KVQUANT" -ctv "$KVQUANT") ;;
  *)
    echo "ERROR: unknown KVQUANT='$KVQUANT' (use f16|f32|bf16|q8_0|q5_1|q5_0|q4_1|q4_0|iq4_nl)"; exit 1 ;;
esac
echo ">> KV-cache: ${KVQUANT:-f16}"

SAMPLER_ARGS=(--temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" --presence-penalty "$PRESENCE_PENALTY")
# shellcheck disable=SC2206  # intentional word-splitting for passthrough flags
[ -n "$EXTRA_ARGS" ] && SAMPLER_ARGS+=($EXTRA_ARGS)

echo ">> sampling: temp=$TEMP top-p=$TOP_P top-k=$TOP_K min-p=$MIN_P presence-penalty=$PRESENCE_PENALTY ${EXTRA_ARGS:+(+ $EXTRA_ARGS)}"
echo ">> context: $CTX   listening: http://$HOST:$PORT/v1"
echo

exec "${RUN[@]}" "$CUDA_BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  "${OFFLOAD_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  "${MMPROJ_ARGS[@]}" \
  -c "$CTX" \
  "${KV_ARGS[@]}" \
  -fa on \
  --parallel "$NP" -b "$BATCH" -ub "$UBATCH" \
  --jinja \
  "${SAMPLER_ARGS[@]}" \
  "${PASS_ARGS[@]}" \
  --host "$HOST" \
  --port "$PORT"
