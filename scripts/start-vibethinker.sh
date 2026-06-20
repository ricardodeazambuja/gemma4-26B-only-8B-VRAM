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
#   SIZE           1.5B | 3B                  (default: 1.5B — sets server file + pi model id)
#   TOOLS          1 to ALSO run the tool-call recovery shim and route pi through it,
#                  so VibeThinker's tool calls execute (sets REASONING_BUDGET=600 +
#                  PROVIDER=vtshim, and runs pi LEAN / --no-extensions — the reminder
#                  extensions derail this 3B's result-reporting). §9.   (default: 0)
#   KEEP_EXTENSIONS 1 to keep pi's extensions under TOOLS=1 (default: 0 = lean).
#   HOST/PORT      server to use / start      (default: 127.0.0.1 / 8080)
#   SHIM_PORT      port for the recovery shim (default: 8088, only with TOOLS=1)
#   MODEL_ID       pi model id                (default: vibethinker-<size>)
#   STOP_ON_EXIT   1 = always stop on exit, 0 = never, unset = ask  (default: ask)
#
# Examples:
#   bash scripts/start-vibethinker.sh                       # 1.5B: server + interactive pi
#   SIZE=3B bash scripts/start-vibethinker.sh               # the 3B coder model
#   TOOLS=1 SIZE=3B bash scripts/start-vibethinker.sh       # + tool calls (server+shim+pi)
#   bash scripts/start-vibethinker.sh -p "Write a function" # one-shot pi prompt
#   STOP_ON_EXIT=1 SIZE=3B bash scripts/start-vibethinker.sh  # auto-stop server on exit
#
set -euo pipefail

for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
# SIZE picks the model (1.5B or 3B); it sets the server file/alias AND the pi model id.
export SIZE="${SIZE:-1.5B}"
SIZE_LC="$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')"
export MODEL_ID="${MODEL_ID:-vibethinker-${SIZE_LC}}"   # run-pi.sh reads this
export HOST PORT

# TOOLS=1: also run the tool-call recovery shim and route pi THROUGH it, so
# VibeThinker's tool calls actually execute (it emits correct intent in a wrapper
# llama.cpp won't parse; the shim recovers it). Needs a reasoning budget or the
# model ruminates and emits no call. See docs/vibethinker-eval.md §9.
TOOLS="${TOOLS:-0}"
SHIM_PORT="${SHIM_PORT:-8088}"
SHIM_PID=""
if [ "$TOOLS" = 1 ]; then
  export REASONING_BUDGET="${REASONING_BUDGET:-600}"   # required for the model to emit tool calls
  export PROVIDER="vtshim"                             # run-pi.sh routes pi at the shim
fi

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

# --- 1b. (TOOLS=1) register the shim provider in pi + start the recovery shim ---
if [ "$TOOLS" = 1 ]; then
  echo ">> TOOLS=1: registering pi provider 'vtshim' + starting the tool-call recovery shim ..."
  python3 - "$SHIM_PORT" "$MODEL_ID" <<'PY'
import json, os, sys
port, model_id = sys.argv[1], sys.argv[2]
path = os.path.expanduser("~/.pi/agent/models.json")
d = json.load(open(path)) if os.path.exists(path) else {"providers": {}}
d.setdefault("providers", {})["vtshim"] = {
    "api": "openai-completions", "apiKey": "dummy",
    "baseUrl": f"http://127.0.0.1:{port}/v1",
    "compat": {"supportsDeveloperRole": False},
    "models": [{"contextWindow": 40960, "id": model_id,
                "name": f"{model_id} (via tool-shim)", "input": ["text"], "reasoning": True}],
}
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(d, open(path, "w"), indent=2)
print(f">> pi provider 'vtshim' -> :{port} (model {model_id})")
PY
  if curl -fsS "http://127.0.0.1:$SHIM_PORT/health" >/dev/null 2>&1; then
    echo ">> a service is already on :$SHIM_PORT — reusing it as the shim."
  else
    SHIM_LOG="$(mktemp -t vibethinker-shim.XXXXXX.log)"
    echo ">> starting tool-shim on :$SHIM_PORT (upstream :$PORT, logs: $SHIM_LOG) ..."
    UPSTREAM="http://$HOST:$PORT" PORT="$SHIM_PORT" HOST="$HOST" \
      setsid python3 "$REPO_ROOT/scripts/vibethinker-toolshim.py" > "$SHIM_LOG" 2>&1 < /dev/null &
    SHIM_PID=$!
    # always stop the shim we started, on any exit
    trap '[ -n "${SHIM_PID:-}" ] && kill "$SHIM_PID" 2>/dev/null || true' EXIT
    for _ in $(seq 1 20); do
      if curl -fsS "http://127.0.0.1:$SHIM_PORT/health" >/dev/null 2>&1; then break; fi
      sleep 0.5
    done
    if ! curl -fsS "http://127.0.0.1:$SHIM_PORT/health" >/dev/null 2>&1; then
      echo "ERROR: the shim did not come up. Last log lines:"; tail -10 "$SHIM_LOG"; exit 1
    fi
    echo ">> shim up on :$SHIM_PORT (PID $SHIM_PID)."
  fi
fi

# --- 2. launch pi (foreground; control returns here when it exits) ----------
# TOOLS=1 runs LEAN by default (--no-extensions). The reminder-injecting extensions
# (grounding / operating-manual / semantic-memory) leak meta-instructions ("keep your
# response minimal", "verify this turn") that derail this small model into NOT
# reporting tool results — it thrashes (repeat calls) or deflects. Measured by
# scripts/test-vibethinker-agent.sh: with extensions 0/2, lean 2/2. Opt back in with
# KEEP_EXTENSIONS=1. (Core tools bash/read/write/edit/ls/find/grep are built-in and stay.)
PI_LEAN=()
[ "$TOOLS" = 1 ] && [ "${KEEP_EXTENSIONS:-0}" != 1 ] && PI_LEAN=(--no-extensions)
if [ "$TOOLS" = 1 ]; then
  echo ">> launching pi (--provider vtshim --model $MODEL_ID — tool calls via the shim${PI_LEAN:+, lean: no extensions}) ..."
else
  echo ">> launching pi (--model $MODEL_ID) ..."
fi
PI_RC=0
bash "$REPO_ROOT/scripts/run-pi.sh" ${PI_LEAN[@]+"${PI_LEAN[@]}"} "$@" || PI_RC=$?

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
