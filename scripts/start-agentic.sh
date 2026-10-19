#!/usr/bin/env bash
#
# start-agentic.sh — one command to bring up the Gemma 4 12B v2 "agentic" server
# (scripts/run-12b-agentic.sh) and drop you into pi, then offer to stop the server
# when pi exits.
#
# This is the start.sh-style turnkey launcher for the DENSE v2 model + TurboQuant.
# Unlike start.sh (the 26B-A4B MoE, with NCMOE / auto-tune machinery), this model is
# dense, so its levers are different: TURBO (full offload via turbo3 V cache), QUANT,
# NGL, CTX, KV type. See scripts/run-12b-agentic.sh -h and docs/gemma4-12b-agentic-eval.md.
#
# Starts the server in the background (if nothing is serving on the port yet), waits
# for it to load, then launches pi in the foreground. When pi exits, if THIS script
# started the server it offers to stop it (a server that was already running when you
# invoked this is always left alone).
#
# Server knobs (forwarded via env to run-12b-agentic.sh — see its -h for all of them):
#   TURBO    1 = turboquant build + turbo3 V cache + full offload (-ngl 99)  (default 0)
#   QUANT    Q3_K_M | Q4_K_M | Q6_K | Q8_0                         (default Q4_K_M)
#   CTX      context window, in tokens   (default: 16384 when TURBO=1, else 32768)
#   NGL      transformer layers on GPU   (default: per-quant; 99 in TURBO mode)
#   KVQUANT  KV type both sides          (default q8_0; TURBO sets K=q8_0 V=turbo3)
#   CTK/CTV  per-side KV type override
#   TEMP / TOP_P / TOP_K / REP_PEN  sampling   (defaults 1.0 / 0.95 / 64 / 1.1)
#   NP / BATCH / UBATCH  TURBO memory knobs    (defaults 1 / 512 / 512)
#   PORT / HOST                                (defaults 8080 / 127.0.0.1)
#   --image  forwarded to the server; all other extra args go to pi (e.g. -p "...").
#
# This script's own knobs:
#   STOP_ON_EXIT  shutdown offer after pi exits: unset = ask (interactive) · 1 = always · 0 = never
#   SYNC_PI       1 = sync pi's contextWindow to CTX so pi can't overrun the server
#                 (default 1; only when we start the server) · 0 = leave pi config alone
#   SERVER_LOG    server log path            (default /tmp/gemma4-agentic-server.log)
#   PROVIDER / MODEL_ID  pi provider/model   (forwarded to run-pi.sh)
#
# Examples:
#   TURBO=1 bash scripts/start-agentic.sh                 # full offload, 16K ctx, ~28 t/s
#   TURBO=1 CTX=20000 bash scripts/start-agentic.sh       # push the context (tighter on 8 GB)
#   QUANT=Q3_K_M TURBO=1 bash scripts/start-agentic.sh    # smaller quant
#   bash scripts/start-agentic.sh                         # stock build, partial offload
#   TURBO=1 bash scripts/start-agentic.sh -p "explain @README.md"   # one-shot, leaves server up
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
TURBO="${TURBO:-0}"
# Effective context: in TURBO mode 16K is the validated safe default on an 8 GB card
# (32K + full offload is tight); otherwise inherit run-12b-agentic.sh's 32K default.
if [ -z "${CTX:-}" ]; then [ "$TURBO" = 1 ] && CTX=16384 || CTX=32768; fi
export CTX TURBO
SERVER_LOG="${SERVER_LOG:-/tmp/gemma4-agentic-server.log}"
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
  echo "   • out of GPU memory — lower CTX, drop a quant (QUANT=Q3_K_M), or in TURBO mode"
  echo "     shrink the batch (BATCH=256 UBATCH=256). Stock build OOMs at -ngl 99 — use TURBO=1."
  echo "   • TURBO=1 but the turboquant build is missing — build it (the run script prints how)."
  echo "   • port $PORT already in use — stop a stale server:  bash scripts/stop-server.sh"
  echo "   • full log:  $SERVER_LOG"
}

if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> server already running at http://$HOST:$PORT — reusing it (left running on exit)"
  if [ ${#SERVER_ARGS[@]} -gt 0 ]; then
    echo ">> NOTE: ${SERVER_ARGS[*]} only applies to a fresh server; the running one is reused as-is."
  fi
else
  echo ">> starting agentic server in background (logs: $SERVER_LOG) ..."
  echo "   TURBO=$TURBO QUANT=${QUANT:-Q4_K_M} CTX=$CTX PORT=$PORT"
  # Own process group (setsid) so an abort can kill the whole tree: run-12b-agentic.sh
  # exec's `mamba run … llama-server`, so llama-server is a grandchild.
  SRV_PGRP=0
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$REPO_ROOT/scripts/run-12b-agentic.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
    SRV_PGRP=1
  else
    nohup bash "$REPO_ROOT/scripts/run-12b-agentic.sh" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
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
  for _ in $(seq 1 150); do          # up to ~5 min
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
    echo "ERROR: the server did not become healthy within ~5 min. Last log lines:"
    tail -20 "$SERVER_LOG"; server_start_hints; exit 1
  fi
  echo ">> server up at http://$HOST:$PORT  (PID $SRV_PID · CTX=$CTX). Stop later: bash scripts/stop-server.sh"
  STARTED_SERVER=1

  # Keep pi's client context window in lockstep with the server's -c. pi silently caps
  # the context at its own configured contextWindow, so a 16K server + a 32K pi entry
  # would overrun the server. We only do this when WE started the server (we know CTX).
  if [ "$SYNC_PI" = 1 ]; then
    echo ">> syncing pi's context window to $CTX (edits ~/.pi/agent/models.json; SYNC_PI=0 to skip) ..."
    if CTX="$CTX" bash "$REPO_ROOT/scripts/configure-pi.sh" >/dev/null 2>&1; then
      echo ">> pi context window set to $CTX."
    else
      echo ">> WARNING: could not sync pi automatically. Run it yourself: CTX=$CTX bash scripts/configure-pi.sh"
    fi
  fi
fi

echo ">> launching pi ..."
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
