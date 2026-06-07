#!/usr/bin/env bash
#
# start.sh — one command to bring up the model server (if not already running)
# and drop you into pi.
#
# Starts scripts/run-server.sh in the background (if nothing is serving yet),
# waits for it to finish loading, then launches pi in the foreground. When pi
# exits, if THIS script started the server it offers to stop it (a server that
# was already running when you invoked start.sh is always left alone).
#
# All run-server.sh knobs pass through via env, e.g.:
#   BACKEND=cuda NCMOE=22 bash scripts/start.sh
#   CTX=32768 bash scripts/start.sh
#   bash scripts/start.sh --image                     # forwarded to the server
#   bash scripts/start.sh -p "summarize @README.md"   # other args go to pi
#
#   STOP_ON_EXIT  control the shutdown offer after pi exits:
#                 unset = ask (only when interactive); 1 = always stop; 0 = never
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared backend banner + resolution (same source as run-server.sh — no dupes).
source "$REPO_ROOT/scripts/_banner.sh"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_LOG="${SERVER_LOG:-/tmp/gemma4-server.log}"
STOP_ON_EXIT="${STOP_ON_EXIT:-}"   # unset = ask, 1 = always, 0 = never
STARTED_SERVER=0                   # set to 1 only if we launch it below

# Split args: server flags (currently just --image) go to run-server.sh, the
# rest go to pi. Without this, --image would be sent to pi and silently ignored.
SERVER_ARGS=()
PI_ARGS=()
for a in "$@"; do
  case "$a" in
    --image) SERVER_ARGS+=("$a") ;;
    *) PI_ARGS+=("$a") ;;
  esac
done

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> server already running at http://$HOST:$PORT — reusing it"
  # Surface the backend the *running* server reported, read from its log (its
  # banner went there). Skipped silently if the log is absent/stale.
  reused_be="$(sed -n 's/.*BACKEND: \([A-Za-z]*\).*/\1/p' "$SERVER_LOG" 2>/dev/null | tail -1 | tr '[:upper:]' '[:lower:]')"
  [ -n "$reused_be" ] && backend_banner "$reused_be"
  if [ ${#SERVER_ARGS[@]} -gt 0 ]; then
    echo ">> NOTE: ${SERVER_ARGS[*]} only applies to a fresh server; the running one is reused as-is."
    echo "         Restart it to apply: bash scripts/stop-server.sh && bash scripts/start.sh ${SERVER_ARGS[*]}"
  fi
else
  # Show which backend the server will come up on (same logic run-server.sh uses).
  backend_banner "$(resolve_backend)"
  echo ">> starting server in background (logs: $SERVER_LOG) ..."
  nohup bash "$REPO_ROOT/scripts/run-server.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
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
  STARTED_SERVER=1
fi

echo ">> launching pi ..."
# Run pi in the foreground (not exec) so control returns here when it exits.
# `|| PI_RC=$?` keeps `set -e` from aborting on a non-zero pi exit (e.g. Ctrl-C).
PI_RC=0
bash "$REPO_ROOT/scripts/run-pi.sh" "${PI_ARGS[@]}" || PI_RC=$?

# --- offer to shut the server down ------------------------------------------
# Only when we started it this run and it's still up. A reused server is the
# user's pre-existing process — leave it alone.
if [ "$STARTED_SERVER" = 1 ] && curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  do_stop=""
  case "$STOP_ON_EXIT" in
    1) do_stop=yes ;;
    0) do_stop=no ;;
    *)
      if [ -t 0 ]; then
        read -r -p $'\n>> pi exited. Stop the model server too? [y/N] ' ans || ans=""
        [[ "$ans" =~ ^[Yy]$ ]] && do_stop=yes || do_stop=no
      else
        do_stop=no   # non-interactive (e.g. -p one-shot): don't guess, leave it up
      fi
      ;;
  esac

  if [ "$do_stop" = yes ]; then
    bash "$REPO_ROOT/scripts/stop-server.sh"
  else
    echo ">> leaving the server running. Stop it with: bash scripts/stop-server.sh"
  fi
fi

exit "$PI_RC"
