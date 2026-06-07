#!/usr/bin/env bash
#
# start.sh — one command to bring up the model server (if not already running)
# and drop you into pi.
#
# Starts scripts/run-server.sh in the background (if nothing is serving yet),
# waits for it to finish loading, then launches pi in the foreground. The server
# is left running after pi exits (stop it with scripts/stop-server.sh).
#
# All run-server.sh knobs pass through via env, e.g.:
#   BACKEND=cuda NCMOE=22 bash scripts/start.sh
#   CTX=32768 bash scripts/start.sh
#   bash scripts/start.sh -p "summarize @README.md"   # extra args go to pi
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_LOG="${SERVER_LOG:-/tmp/gemma4-server.log}"

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> server already running at http://$HOST:$PORT — reusing it"
else
  echo ">> starting server in background (logs: $SERVER_LOG) ..."
  nohup bash "$REPO_ROOT/scripts/run-server.sh" > "$SERVER_LOG" 2>&1 &
  SRV_PID=$!
  printf ">> waiting for the model to load"
  for _ in $(seq 1 150); do          # up to ~5 min
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then printf " ready.\n"; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      printf "\nERROR: server exited while starting. Last log lines:\n"
      tail -20 "$SERVER_LOG"; exit 1
    fi
    printf "."; sleep 2
  done
  if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo; echo "ERROR: server did not become healthy in time. See $SERVER_LOG"; exit 1
  fi
  echo ">> server up (PID $SRV_PID). Stop it later with: bash scripts/stop-server.sh"
fi

echo ">> launching pi ..."
exec bash "$REPO_ROOT/scripts/run-pi.sh" "$@"
