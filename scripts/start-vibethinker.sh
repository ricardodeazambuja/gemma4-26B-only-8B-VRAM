#!/usr/bin/env bash
#
# start-vibethinker.sh — all-in-one launcher for VibeThinker-1.5B: start the
# server, wait for it to load, launch pi against it, and on exit offer to shut
# the server down. A minimal sibling of start.sh (no menu / autotune / MTP — this
# model is a tiny dense reasoner that needs none of that; see run-vibethinker.sh).
#
# If a server is already serving on the port, it is REUSED (and left running on
# exit — it's yours, not ours). Otherwise we start one and clean it up if asked.
#
# Settings (env vars; run-vibethinker.sh knobs like QUANT/CTX/TEMP pass through):
#   HOST/PORT      server to use / start      (default: 127.0.0.1 / 8080)
#   MODEL_ID       pi model id                (default: vibethinker-1.5b)
#   STOP_ON_EXIT   1 = always stop on exit, 0 = never, unset = ask  (default: ask)
#
# Examples:
#   bash scripts/start-vibethinker.sh                       # server + interactive pi
#   bash scripts/start-vibethinker.sh -p "Write a function" # one-shot pi prompt
#   QUANT=f16 bash scripts/start-vibethinker.sh             # lossless quant
#   STOP_ON_EXIT=1 bash scripts/start-vibethinker.sh        # auto-stop server when pi exits
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
export MODEL_ID="${MODEL_ID:-vibethinker-1.5b}"   # run-pi.sh reads this
export HOST PORT

STARTED_SERVER=0

# --- 1. start the server (unless one is already up) -------------------------
if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> reusing the server already listening on http://$HOST:$PORT (left running on exit)."
else
  SERVER_LOG="$(mktemp -t vibethinker-server.XXXXXX.log)"
  echo ">> starting server in background (logs: $SERVER_LOG) ..."
  # Own process group (setsid) so an abort during load can kill the whole tree:
  # run-vibethinker.sh exec's `mamba run … llama-server`, so llama-server is a
  # grandchild — killing only $SRV_PID would orphan it.
  SRV_PGRP=0
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$REPO_ROOT/scripts/run-vibethinker.sh" > "$SERVER_LOG" 2>&1 &
    SRV_PGRP=1
  else
    nohup bash "$REPO_ROOT/scripts/run-vibethinker.sh" > "$SERVER_LOG" 2>&1 &
  fi
  SRV_PID=$!

  # Ctrl-C before pi comes up -> stop the half-loaded server we just started.
  _abort_during_load() {
    printf '\n>> aborted while loading — stopping the server we started.\n'
    if kill -0 "$SRV_PID" 2>/dev/null; then
      if [ "$SRV_PGRP" = 1 ]; then
        kill -TERM -- -"$SRV_PID" 2>/dev/null || true; sleep 1
        kill -KILL -- -"$SRV_PID" 2>/dev/null || true
      else
        pkill -TERM -P "$SRV_PID" 2>/dev/null || true
        kill -TERM "$SRV_PID" 2>/dev/null || true
        bash "$REPO_ROOT/scripts/stop-server.sh" >/dev/null 2>&1 || true
      fi
      echo "   server stopped."
    fi
    exit 130
  }
  trap _abort_during_load INT TERM

  echo ">> waiting for the model to load (Ctrl-C aborts) ..."
  _t0=$SECONDS
  for _ in $(seq 1 90); do          # up to ~3 min (this model loads in seconds)
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo ">> model ready in $((SECONDS - _t0))s."
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "ERROR: the server exited while starting. Last log lines:"; tail -20 "$SERVER_LOG"; exit 1
    fi
    printf '.'; sleep 2
  done
  trap - INT TERM
  if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo; echo "ERROR: the server did not become healthy. Last log lines:"; tail -20 "$SERVER_LOG"; exit 1
  fi
  echo ">> server up at http://$HOST:$PORT (PID $SRV_PID). Stop later: bash scripts/stop-server.sh"
  STARTED_SERVER=1
fi

# --- 2. launch pi (foreground; control returns here when it exits) ----------
echo ">> launching pi (--model $MODEL_ID) ..."
PI_RC=0
bash "$REPO_ROOT/scripts/run-pi.sh" "$@" || PI_RC=$?

# --- 3. offer to shut the server down ---------------------------------------
# Only when we started it this run and it's still up. A reused server is left alone.
if [ "$STARTED_SERVER" = 1 ] && curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  do_stop=""
  case "${STOP_ON_EXIT:-}" in
    1) do_stop=yes ;;
    0) do_stop=no ;;
    *)
      if [ -t 0 ]; then
        read -r -p $'\n>> pi exited. Stop the model server too? [y/N] ' ans || ans=""
        [[ "$ans" =~ ^[Yy]$ ]] && do_stop=yes || do_stop=no
      else
        do_stop=no   # non-interactive (e.g. -p one-shot): leave it up
      fi ;;
  esac
  if [ "$do_stop" = yes ]; then
    bash "$REPO_ROOT/scripts/stop-server.sh"
  else
    echo ">> leaving the server running. Stop it with: bash scripts/stop-server.sh"
  fi
fi

exit "$PI_RC"
