#!/usr/bin/env bash
#
# stop-server.sh — stop the local llama.cpp server by the port it listens on.
#
#   PORT   port the server binds (default: 8080)
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

PORT="${PORT:-8080}"

# find the PID listening on $PORT (ss first, lsof fallback)
PIDS=""
if command -v ss >/dev/null 2>&1; then
  PIDS="$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"
fi
if [ -z "$PIDS" ] && command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sort -u || true)"
fi

if [ -z "$PIDS" ]; then
  echo ">> no server listening on port $PORT"
  exit 0
fi

echo ">> stopping server PID(s) on port $PORT: $PIDS"
for p in $PIDS; do kill "$p" 2>/dev/null || true; done
sleep 2
# escalate if still alive
for p in $PIDS; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true; done
echo ">> stopped."
