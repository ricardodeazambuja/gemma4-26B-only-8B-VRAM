#!/usr/bin/env bash
#
# run-pi.sh — launch the pi coding agent against the local Gemma 4 server.
#
# Any extra arguments are passed straight through to pi, e.g.:
#   bash scripts/run-pi.sh                       # interactive
#   bash scripts/run-pi.sh -p "explain this code @file.py"   # one-shot
#
# Config (env vars):
#   PROVIDER   pi provider name      (default: llamacpp)
#   MODEL_ID   pi model id           (default: gemma-4-26b-a4b-qat)
#   HOST/PORT  server to health-check (default: 127.0.0.1 / 8080)
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

PROVIDER="${PROVIDER:-llamacpp}"
MODEL_ID="${MODEL_ID:-gemma-4-26b-a4b-qat}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

command -v pi >/dev/null 2>&1 || {
  echo "ERROR: 'pi' not found. Install with: npm i -g @mariozechner/pi-coding-agent"
  exit 1
}

if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo "WARNING: no server responding at http://$HOST:$PORT"
  echo "         start one first:  bash scripts/run-server.sh   (or use scripts/start.sh)"
fi

exec pi --provider "$PROVIDER" --model "$MODEL_ID" "$@"
