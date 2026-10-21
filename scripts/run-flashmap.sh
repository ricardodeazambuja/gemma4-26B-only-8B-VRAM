#!/usr/bin/env bash
#
# run-flashmap.sh — serve Qwen3-8B (Q4_K_M) with the FlashMap flow-map drafter
# (j-shen/FlashMap-Qwen3-8B) for lossless speculative decoding.
#
# FlashMap is not in any llama.cpp; it runs on the buun-llama.cpp fork patched with
# patches/buun-llama.cpp-flashmap.patch (a variant of the fork's DFlash drafter, driven
# by its DSpark anchor-first protocol). See docs/flashmap.md.
#
# Setup:
#   bash scripts/download-flashmap.sh                     # target GGUF + drafter + embeddings
#   build buun-llama.cpp + patches/buun-llama.cpp-flashmap.patch  (exact commands: docs/flashmap.md)
#   mamba run -n local python scripts/convert-flashmap.py models/qwen3-8b/flashmap-hf \
#       models/qwen3-8b/hf/embed_tokens.safetensors models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf \
#       models/qwen3-8b/flashmap-qwen3-8b-f16.gguf
#
# Config (override via env vars):
#   DRAFT       1 = FlashMap drafter, 0 = target alone (baseline)   (default: 1)
#   DRAFT_MAX   draft tokens per step, <= 16 (the trained block)    (default: 16)
#   DRAFT_GGUF  drafter GGUF       (default: models/qwen3-8b/flashmap-qwen3-8b-f16.gguf)
#   CTX         context window                                       (default: 8192)
#   KVQUANT     target KV cache type                                 (default: q8_0)
#   NGL         target layers on GPU                                 (default: 99)
#   PORT/HOST                                                        (default: 8080 / 127.0.0.1)
#   TEMP        sampling temperature (0 = greedy)                    (default: 0)
#   CUDA_BIN    llama-server binary  (default: vendor/buun-llama.cpp/build/bin/llama-server)
#
# Examples:
#   bash scripts/run-flashmap.sh                 # FlashMap, 16-token drafts, greedy
#   DRAFT=0 bash scripts/run-flashmap.sh         # baseline: the target alone
#   DRAFT_MAX=8 bash scripts/run-flashmap.sh     # shallower drafts
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-$REPO_ROOT/models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf}"
DRAFT="${DRAFT:-1}"
DRAFT_MAX="${DRAFT_MAX:-16}"
DRAFT_GGUF="${DRAFT_GGUF:-$REPO_ROOT/models/qwen3-8b/flashmap-qwen3-8b-f16.gguf}"
CTX="${CTX:-8192}"
KVQUANT="${KVQUANT:-q8_0}"
NGL="${NGL:-99}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TEMP="${TEMP:-0}"
CUDA_BIN="${CUDA_BIN:-$REPO_ROOT/vendor/buun-llama.cpp/build/bin/llama-server}"

[ -f "$MODEL" ] || { echo "ERROR: target not found: $MODEL (run scripts/download-flashmap.sh)"; exit 1; }
[ -x "$CUDA_BIN" ] || { echo "ERROR: llama-server not found at $CUDA_BIN (build the patched buun fork)"; exit 1; }

SPEC_ARGS=()
if [ "$DRAFT" = 1 ]; then
  [ -f "$DRAFT_GGUF" ] || { echo "ERROR: drafter not found: $DRAFT_GGUF (run scripts/convert-flashmap.py)"; exit 1; }
  SPEC_ARGS=(-md "$DRAFT_GGUF" -ngld 99 -cd "$CTX" --spec-draft-n-max "$DRAFT_MAX")
fi

echo ">> target:  $MODEL (-ngl $NGL, ctx $CTX, KV $KVQUANT)"
if [ "$DRAFT" = 1 ]; then
  echo ">> drafter: $DRAFT_GGUF (FlashMap, $DRAFT_MAX-token drafts)"
else
  echo ">> drafter: none (baseline)"
fi
echo ">> listening: http://$HOST:$PORT/v1   temp=$TEMP"
echo

exec mamba run --no-capture-output -n "${CUDA_ENV:-llamacpp-cuda}" "$CUDA_BIN" \
  -m "$MODEL" \
  --alias qwen3-8b \
  --device CUDA0 \
  -ngl "$NGL" \
  -c "$CTX" \
  -ctk "$KVQUANT" -ctv "$KVQUANT" -fa on \
  --jinja \
  --temp "$TEMP" \
  -np 1 \
  "${SPEC_ARGS[@]}" \
  --host "$HOST" --port "$PORT" \
  "$@"
