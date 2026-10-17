#!/usr/bin/env bash
#
# start-qwen.sh — one command to bring up a Qwen bake-off candidate
# (scripts/run-qwen.sh) and drop you into pi, then offer to stop the server when
# pi exits. Branch qwen_on_8gb_vram; download the models first with
# scripts/download-qwen-bakeoff.sh.
#
# Starts the server in the background (if nothing is serving on the port yet), waits
# for it to load, then launches pi in the foreground. When pi exits, if THIS script
# started the server it offers to stop it (a server that was already running when you
# invoked this is always left alone).
#
# Server knobs (forwarded via env to run-qwen.sh — see its -h for all of them):
#   QMODEL   qwen36 | qwen38 | distill    (default qwen36)
#   CTX      context window, in tokens    (default: 131072 MoE / 32768 dense qwen38)
#   NCMOE    MoE: first N layers' experts on CPU, rest on GPU  (default: qwen36 32, distill 34)
#   NGL      dense qwen38: layers on GPU  (default 30 of 64; 36 OOMs at 32K)
#   KVQUANT  KV type both sides           (default q8_0)
#   MTP      qwen38 only: 1 = MTP speculative decoding (default 0)
#   TEMP / TOP_P / TOP_K / PRESENCE_PENALTY  sampling (defaults per model card)
#   NP / BATCH / UBATCH  slots / batch sizes  (defaults 1 / 512 / 512)
#   PORT / HOST                                (defaults 8080 / 127.0.0.1)
#   --image  forwarded to the server; all other extra args go to pi (e.g. -p "...").
#
# This script's own knobs:
#   STOP_ON_EXIT  shutdown offer after pi exits: unset = ask (interactive) · 1 = always · 0 = never
#   SYNC_PI       1 = sync pi's contextWindow to the server's actual n_ctx so pi can't
#                 overrun it (default 1, also for a reused server) · 0 = leave pi config alone
#   SERVER_LOG    server log path            (default /tmp/qwen-bakeoff-server.log)
#   PROVIDER / MODEL_ID  pi provider/model   (forwarded to run-pi.sh; MODEL_ID defaults to
#                 the id the server reports in /v1/models, so pi's label shows the model
#                 that is really loaded — the ids are registered in config/pi-provider.json)
#
# Examples:
#   bash scripts/start-qwen.sh                          # Qwen3.6-35B-A3B, 128K ctx
#   QMODEL=distill bash scripts/start-qwen.sh           # the Qwen3.8-distilled 35B-A3B
#   QMODEL=qwen38 NGL=40 bash scripts/start-qwen.sh     # dense 27B, more layers on GPU
#   QMODEL=qwen38 bash scripts/start-qwen.sh -p "explain @README.md"   # one-shot, leaves server up
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
QMODEL="${QMODEL:-qwen36}"
# Effective context, mirroring run-qwen.sh's defaults so pi gets synced to the same value.
if [ -z "${CTX:-}" ]; then [ "$QMODEL" = qwen38 ] && CTX=32768 || CTX=131072; fi
export CTX QMODEL
SERVER_LOG="${SERVER_LOG:-/tmp/qwen-bakeoff-server.log}"
STOP_ON_EXIT="${STOP_ON_EXIT:-}"
SYNC_PI="${SYNC_PI:-1}"
STARTED_SERVER=0

# Split args: --image goes to the server; everything else goes to pi.
SERVER_ARGS=()
PI_ARGS=()
for a in "$@"; do
  case "$a" in
    --image) SERVER_ARGS+=("$a") ;;
    *)       PI_ARGS+=("$a") ;;
  esac
done

server_start_hints() {
  echo ">> Common causes & fixes:"
  echo "   • out of GPU memory — lower CTX, lower NGL (qwen38), raise NCMOE (MoE),"
  echo "     or shrink the batch (BATCH=256 UBATCH=256)."
  echo "   • 'Killed' after a clean load = host RAM OOM killer (check journalctl -k), not VRAM."
  echo "   • model file missing — bash scripts/download-qwen-bakeoff.sh $QMODEL"
  echo "   • port $PORT already in use — stop a stale server:  bash scripts/stop-server.sh"
  echo "   • full log:  $SERVER_LOG"
}

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> server already running at http://$HOST:$PORT — reusing it (left running on exit)"
  if [ ${#SERVER_ARGS[@]} -gt 0 ]; then
    echo ">> NOTE: ${SERVER_ARGS[*]} only applies to a fresh server; the running one is reused as-is."
  fi
else
  echo ">> starting $QMODEL server in background (logs: $SERVER_LOG) ..."
  echo "   QMODEL=$QMODEL CTX=$CTX PORT=$PORT"
  # Own process group (setsid) so an abort can kill the whole tree: run-qwen.sh
  # exec's `mamba run … llama-server`, so llama-server is a grandchild.
  SRV_PGRP=0
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$REPO_ROOT/scripts/run-qwen.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    SRV_PGRP=1
  else
    nohup bash "$REPO_ROOT/scripts/run-qwen.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
  fi
  SRV_PID=$!

  _abort_during_load() {
    printf '\n>> aborted while the model was loading — stopping the server we started.\n'
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

  echo ">> waiting for the model to load — first load can take a few minutes (Ctrl-C aborts)"
  _t0=$SECONDS; _tty=0; [ -t 1 ] && _tty=1
  [ "$_tty" = 1 ] && echo ">> server log follows (indented); '>> model ready' prints when it is up:"
  _seen=0; _last_out=$SECONDS
  for _ in $(seq 1 300); do          # up to ~10 min (18-20 GB GGUFs)
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo ">> model ready in $((SECONDS - _t0))s."
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "ERROR: the server exited while starting. Last log lines:"
      tail -20 "$SERVER_LOG"; server_start_hints; exit 1
    fi
    if [ "$_tty" = 1 ]; then
      _total=$(wc -l < "$SERVER_LOG" 2>/dev/null || echo 0)
      if [ "$_total" -gt "$_seen" ]; then
        sed -n "$((_seen + 1)),${_total}p" "$SERVER_LOG" 2>/dev/null \
          | sed -E 's/\x1b\[[0-9;?=]*[A-Za-z]//g' | tr -d '\000-\010\013-\037\177' \
          | while IFS= read -r _line; do if [ -n "$_line" ]; then printf '     %s\n' "$_line"; fi; done
        _seen=$_total; _last_out=$SECONDS
      elif [ "$((SECONDS - _last_out))" -ge 15 ]; then
        printf '     … still loading (%ds elapsed)\n' "$((SECONDS - _t0))"; _last_out=$SECONDS
      fi
    else
      printf '.'
    fi
    sleep 2
  done
  trap - INT TERM
  if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    [ "$_tty" = 1 ] || echo
    echo "ERROR: the server did not become healthy within ~10 min. Last log lines:"
    tail -20 "$SERVER_LOG"; server_start_hints; exit 1
  fi
  echo ">> server up at http://$HOST:$PORT  (PID $SRV_PID · CTX=$CTX). Stop later: bash scripts/stop-server.sh"
  STARTED_SERVER=1
fi

# Ask the server what it is actually running, so pi's label and context match it —
# also when we reused a server started by someone else (possibly another QMODEL).
# llama.cpp ignores the model id pi requests, so without this pi would show whatever
# id it was launched with (the gemma default) no matter which model is loaded.
SERVED_ID="$(curl -fsS "http://$HOST:$PORT/v1/models" 2>/dev/null \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
SERVED_CTX="$(curl -fsS "http://$HOST:$PORT/props" 2>/dev/null \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["default_generation_settings"]["n_ctx"])' 2>/dev/null || true)"
[ -n "$SERVED_CTX" ] && CTX="$SERVED_CTX"
if [ -z "${MODEL_ID:-}" ] && [ -n "$SERVED_ID" ]; then
  export MODEL_ID="$SERVED_ID"
fi
echo ">> server is running: ${SERVED_ID:-unknown model} · n_ctx=${SERVED_CTX:-unknown}"

# Keep pi's client context window in lockstep with the server's -c. pi silently caps
# the context at its own configured contextWindow, so a 16K server + a 32K pi entry
# would overrun the server. This also (re)registers the Qwen model ids in pi.
if [ "$SYNC_PI" = 1 ]; then
  echo ">> syncing pi's context window to $CTX (edits ~/.pi/agent/models.json; SYNC_PI=0 to skip) ..."
  if CTX="$CTX" bash "$REPO_ROOT/scripts/configure-pi.sh" >/dev/null 2>&1; then
    echo ">> pi context window set to $CTX."
  else
    echo ">> WARNING: could not sync pi automatically. Run it yourself: CTX=$CTX bash scripts/configure-pi.sh"
  fi
fi

echo ">> launching pi (model ${MODEL_ID:-gemma-4-26b-a4b-qat}) ..."
PI_RC=0
bash "$REPO_ROOT/scripts/run-pi.sh" "${PI_ARGS[@]}" || PI_RC=$?

# --- offer to shut the server down (only if we started it this run) ----------
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
        do_stop=no   # non-interactive (e.g. -p one-shot): leave it up
      fi ;;
  esac
  if [ "$do_stop" = yes ]; then
    PORT="$PORT" bash "$REPO_ROOT/scripts/stop-server.sh"
  else
    echo ">> leaving the server running. Stop it with: bash scripts/stop-server.sh"
  fi
fi

exit "$PI_RC"
