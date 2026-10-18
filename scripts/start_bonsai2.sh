#!/usr/bin/env bash
#
# start_bonsai2.sh — Bonsai 2 27B (Qwen3.8-27B in ternary weights) + pi, no env vars needed.
#
#   bash scripts/start_bonsai2.sh                 # server (if needed) + pi, thinking "medium"
#   bash scripts/start_bonsai2.sh --thinking low  # any pi flag passes through
#   bash scripts/start_bonsai2.sh -p "explain @README.md" --no-session   # one-shot
#   bash scripts/start_bonsai2.sh -xt 'shepherd_*,goal_*'                # hide more tools
#
# A thin wrapper over scripts/start-qwen.sh with QMODEL=bonsai2 and pi's "medium" thinking
# level (at the template's default it thinks for minutes per turn at 27 tok/s). The server
# settings come from run-qwen.sh's bonsai2 defaults: all layers on the GPU, 16K context,
# f16 KV (q8_0 crashes on Turing), PrismML fork in vendor/prism-llama.cpp. Details and
# measurements: docs/bonsai2-on-8gb.md. Every start-qwen.sh knob (CTX, PORT, STOP_ON_EXIT,
# ...) still works if you set it; you never have to.
#
# Hides shepherd-pi's tools (-xt 'shepherd_*'): their 10 definitions cost ~2,000 of the 16K
# tokens before the first message (measured: ~5,400 with them, ~3,400 without), and
# delegating to tmux agents is not a job for this model. Passing your own -xt/--exclude-tools
# or -t/--tools replaces that default, so include shepherd_* in it if you still want them hidden.
#
# Refuses to start if another model is already serving on the port: start-qwen.sh would
# reuse that server, and pi would be talking to the wrong model. Stop it first with
# scripts/stop-server.sh.
#
# First time on a new machine: build the fork and download the model (docs/bonsai2-on-8gb.md,
# "Run it"); this script checks both and says what is missing.
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

MODEL_FILE="$REPO_ROOT/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
SERVER_BIN="$REPO_ROOT/vendor/prism-llama.cpp/build/bin/llama-server"
if [ ! -f "$MODEL_FILE" ]; then
  echo "ERROR: model not found: $MODEL_FILE"
  echo "       download it: bash scripts/download-qwen-bakeoff.sh bonsai2"
  exit 1
fi
if [ ! -x "$SERVER_BIN" ]; then
  echo "ERROR: the PrismML llama.cpp fork is not built: $SERVER_BIN"
  echo "       build it as in docs/bonsai2-on-8gb.md, \"Run it\""
  exit 1
fi

# Something already serving? Fine if it is Bonsai, refuse otherwise.
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  SERVED_ID="$(curl -fsS "http://$HOST:$PORT/v1/models" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  if [ "$SERVED_ID" != "bonsai2-27b" ]; then
    echo "ERROR: port $PORT is already serving '${SERVED_ID:-unknown}', not Bonsai 2."
    echo "       stop it first: bash scripts/stop-server.sh"
    exit 1
  fi
  echo ">> Bonsai 2 is already running on port $PORT; reusing it."
fi

# Default pi to "medium" thinking and no shepherd-pi tools, unless the caller chose.
THINKING=(--thinking medium)
EXCLUDE=(--exclude-tools 'shepherd_*')
for _arg in "$@"; do case "$_arg" in
  --thinking|--thinking=*) THINKING=() ;;
  -xt|--exclude-tools|--exclude-tools=*|-t|--tools|--tools=*|-nt|--no-tools) EXCLUDE=() ;;
esac; done

export QMODEL=bonsai2
exec bash "$REPO_ROOT/scripts/start-qwen.sh" "${THINKING[@]}" "${EXCLUDE[@]}" "$@"
