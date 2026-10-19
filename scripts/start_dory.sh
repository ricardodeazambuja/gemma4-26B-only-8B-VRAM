#!/usr/bin/env bash
#
# start_dory.sh — NVIDIA Nemotron-Labs Dory1.0-4B + pi, no env vars needed.
#
#   bash scripts/start_dory.sh                                 # server (if needed) + pi
#   LOOPS=4 bash scripts/start_dory.sh                         # more loops: slower, more accurate
#   bash scripts/start_dory.sh -p "explain @README.md" --no-session   # one-shot
#
# A thin wrapper over scripts/start-qwen.sh with QMODEL=dory. The server settings come from
# run-qwen.sh's dory defaults: all layers on the GPU, 64K context, q8_0 KV, flash attention
# on, 2 loops with a per-loop KV cache (LOOPS=3/4 switch to one shared last-loop cache), the
# llama.cpp PR build in vendor/dory-llama.cpp.
# Details and measurements: docs/dory-on-8gb.md. Every start-qwen.sh / run-qwen.sh knob
# (LOOPS, LOOP_KV, CTX, PORT, ...) still works if you set it; you never have to.
#
# Hides shepherd-pi's tools (-xt 'shepherd_*') for the same reason as start_bonsai2.sh: ~2,000
# tokens before the first message. Passing your own -xt/--exclude-tools or -t/--tools
# replaces that default.
#
# Refuses to start if another model is already serving on the port: start-qwen.sh would
# reuse that server, and pi would be talking to the wrong model. Stop it first with
# scripts/stop-server.sh.
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

MODEL_FILE="$REPO_ROOT/models/dory1-4b/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: model not found: $MODEL_FILE"
  echo "       download it: bash scripts/download-qwen-bakeoff.sh dory"
  exit 1
fi
SERVER_BIN="$REPO_ROOT/vendor/dory-llama.cpp/build/bin/llama-server"
if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: Dory llama.cpp build not found: $SERVER_BIN"
  echo "       build it (see docs/dory-on-8gb.md):"
  echo "       CUDA_VER=12.2 LLAMA_REPO=https://github.com/brianchmielNV/llama.cpp \\"
  echo "         LLAMA_REF=43e865ec059a35d57f1167eaea76ca187077f063 SRC_DIR=\$PWD/vendor/dory-llama.cpp \\"
  echo "         bash scripts/build-llama-cuda.sh"
  exit 1
fi

# Something already serving? Fine if it is Dory, refuse otherwise.
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  SERVED_ID="$(curl -fsS "http://$HOST:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  if [ "$SERVED_ID" != "dory1-4b" ]; then
    echo "ERROR: port $PORT is already serving '${SERVED_ID:-unknown}', not Dory1.0-4B."
    echo "       stop it first: bash scripts/stop-server.sh"
    exit 1
  fi
  echo ">> Dory1.0-4B is already running on port $PORT; reusing it."
fi

# Default pi to no shepherd-pi tools, unless the caller chose.
EXCLUDE=(--exclude-tools 'shepherd_*')
for _arg in "$@"; do case "$_arg" in
  -xt|--exclude-tools|--exclude-tools=*|-t|--tools|--tools=*|-nt|--no-tools) EXCLUDE=() ;;
esac; done

export QMODEL=dory
exec bash "$REPO_ROOT/scripts/start-qwen.sh" "${EXCLUDE[@]}" "$@"
