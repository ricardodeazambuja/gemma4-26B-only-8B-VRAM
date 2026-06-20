#!/usr/bin/env bash
#
# test-vibethinker-agent.sh — end-to-end check that VibeThinker-3B works as an
# AGENT in pi: it must CALL a tool and USE the real result, not hallucinate it.
#
# The trick: plant an UNGUESSABLE random token in a file and require the model to
# retrieve it. It cannot produce that token without actually reading the file, so
# the token appearing in the answer is proof the tool ran and its result was used.
# A second probe (`bash` echo of a fresh token) checks the shell path too.
#
# Brings up server (budget 600) + shim, runs the probes through pi via the shim,
# verifies, and cleans everything up. OOM-safe: one small server, stopped on exit.
#
#   bash scripts/test-vibethinker-agent.sh        # SIZE=3B by default
#
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST=127.0.0.1; PORT=8080; SHIM=8088
SIZE="${SIZE:-3B}"; SIZE_LC="$(echo "$SIZE" | tr '[:upper:]' '[:lower:]')"
MODEL_ID="vibethinker-${SIZE_LC}"
WS="$(mktemp -d -t vt-agent-test.XXXXXX)"
SHIMLOG="$(mktemp -t vt-agent-test-shim.XXXXXX.log)"
STARTED_SERVER=0; SHIM_PID=""

cleanup() {
  [ -n "$SHIM_PID" ] && kill "$SHIM_PID" 2>/dev/null
  lsof -ti tcp:$SHIM 2>/dev/null | xargs -r kill 2>/dev/null
  [ "$STARTED_SERVER" = 1 ] && bash "$REPO/scripts/stop-server.sh" >/dev/null 2>&1
  rm -rf "$WS"
}
trap cleanup EXIT

# unguessable tokens — the model cannot know these without using a tool
TOK_FILE="VT-$(head -c8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
TOK_BASH="VT-$(head -c8 /dev/urandom | od -An -tx1 | tr -d ' \n')"
# dot-free filename on purpose: the 3B tends to mangle "secret.txt" -> "secretxt"
printf '%s\n' "$TOK_FILE" > "$WS/token"

echo ">> workspace $WS"
echo ">> file-token=$TOK_FILE  bash-token=$TOK_BASH"

# --- server (reuse if already up) ---
if ! curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
  echo ">> starting server (SIZE=$SIZE, budget 600) ..."
  REASONING_BUDGET=600 SIZE="$SIZE" nohup setsid bash "$REPO/scripts/run-vibethinker.sh" >/tmp/vt-agent-test-server.log 2>&1 &
  STARTED_SERVER=1
  for _ in $(seq 1 90); do curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1 || { echo "FAIL: server did not come up"; tail -8 /tmp/vt-agent-test-server.log; exit 1; }
fi

# --- shim ---
UPSTREAM="http://$HOST:$PORT" PORT=$SHIM HOST=$HOST setsid python3 "$REPO/scripts/vibethinker-toolshim.py" >"$SHIMLOG" 2>&1 </dev/null &
SHIM_PID=$!
for _ in $(seq 1 20); do curl -fsS "http://$HOST:$SHIM/health" >/dev/null 2>&1 && break; sleep 0.5; done
curl -fsS "http://$HOST:$SHIM/health" >/dev/null 2>&1 || { echo "FAIL: shim did not come up"; tail "$SHIMLOG"; exit 1; }

# register the vtshim provider for pi (idempotent)
python3 - "$SHIM" "$MODEL_ID" <<'PY'
import json, os, sys
port, model_id = sys.argv[1], sys.argv[2]
p = os.path.expanduser("~/.pi/agent/models.json")
d = json.load(open(p)) if os.path.exists(p) else {"providers": {}}
d.setdefault("providers", {})["vtshim"] = {"api": "openai-completions", "apiKey": "dummy",
    "baseUrl": f"http://127.0.0.1:{port}/v1", "compat": {"supportsDeveloperRole": False},
    "models": [{"contextWindow": 40960, "id": model_id, "name": f"{model_id} (tool-shim)",
                "input": ["text"], "reasoning": True}]}
os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(d, open(p, "w"), indent=2)
PY

# run a one-shot pi task in the workspace; echo its final answer text.
# LEAN=1 (default) drops the reminder extensions (grounding/operating-manual),
# which otherwise leak meta-instructions ("keep your response minimal", "verify
# this turn") that derail this small model into NOT reporting a tool's result.
PI_EXTRA=(); [ "${LEAN:-1}" = 1 ] && PI_EXTRA=(--no-extensions)
probe() { (cd "$WS" && timeout 200 pi --provider vtshim --model "$MODEL_ID" "${PI_EXTRA[@]}" -p "$1" 2>/dev/null); }

pass=0; total=0
check() {  # $1=label $2=expected-token $3=output
  total=$((total+1))
  if printf '%s' "$3" | grep -qF "$2"; then
    echo "  PASS  $1 — real token present (tool result used, not hallucinated)"; pass=$((pass+1))
  else
    echo "  FAIL  $1 — token $2 NOT in answer (model likely hallucinated). Got: $(printf '%s' "$3" | tr '\n' ' ' | tail -c 160)"
  fi
}

echo ">> probe 1: read a planted token via a tool"
OUT1="$(probe "There is a file named token in the current directory. Use a tool to read it and tell me the exact value it contains.")"
check "file-read" "$TOK_FILE" "$OUT1"

echo ">> probe 2: shell command output via bash"
OUT2="$(probe "Use the bash tool to run exactly: echo $TOK_BASH   — then tell me what it printed.")"
check "bash-echo" "$TOK_BASH" "$OUT2"

echo ""
echo ">> recoveries seen by the shim:"; grep -oE "RECOVERED tool_call -> [a-z]+" "$SHIMLOG" | sort | uniq -c | sed 's/^/   /'
echo ">> RESULT: $pass/$total probes used real tool results"
[ "$pass" -ge 1 ] && exit 0 || exit 1
