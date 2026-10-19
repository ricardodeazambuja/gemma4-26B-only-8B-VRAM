#!/usr/bin/env bash
#
# start_qwen3vl.sh — Qwen3-VL-8B-Instruct (vision + text) + pi, no env vars needed.
#
#   bash scripts/start_qwen3vl.sh                                  # server (if needed) + pi
#   bash scripts/start_qwen3vl.sh -p "describe @photo.jpg" --no-session   # one-shot
#
# A thin wrapper over scripts/start-qwen.sh with QMODEL=qwen3vl and the vision projector
# always loaded (--image). The server settings come from run-qwen.sh's qwen3vl defaults:
# all layers and the projector on the GPU, 16K context, q8_0 KV, at most 1024 tokens per
# image, stock llama.cpp in vendor/llama.cpp. Details and measurements:
# docs/qwen3vl-on-8gb.md. Every start-qwen.sh / run-qwen.sh knob (CTX, IMAGE_MAX_TOKENS,
# PORT, ...) still works if you set it; you never have to.
#
# Hides shepherd-pi's tools (-xt 'shepherd_*') for the same reason as start_bonsai2.sh: ~2,000
# of the 16K tokens before the first message. Passing your own -xt/--exclude-tools or
# -t/--tools replaces that default.
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

MODEL_FILE="$REPO_ROOT/models/qwen3vl-8b/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: model not found: $MODEL_FILE"
  echo "       download it: bash scripts/download-qwen-bakeoff.sh qwen3vl"
  exit 1
fi

# Something already serving? Fine if it is Qwen3-VL, refuse otherwise.
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  SERVED_ID="$(curl -fsS "http://$HOST:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  if [ "$SERVED_ID" != "qwen3-vl-8b" ]; then
    echo "ERROR: port $PORT is already serving '${SERVED_ID:-unknown}', not Qwen3-VL-8B."
    echo "       stop it first: bash scripts/stop-server.sh"
    exit 1
  fi
  echo ">> Qwen3-VL-8B is already running on port $PORT; reusing it."
fi

# Default pi to no shepherd-pi tools, unless the caller chose.
EXCLUDE=(--exclude-tools 'shepherd_*')
for _arg in "$@"; do case "$_arg" in
  -xt|--exclude-tools|--exclude-tools=*|-t|--tools|--tools=*|-nt|--no-tools) EXCLUDE=() ;;
esac; done

export QMODEL=qwen3vl
exec bash "$REPO_ROOT/scripts/start-qwen.sh" --image "${EXCLUDE[@]}" "$@"
