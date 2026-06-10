#!/usr/bin/env bash
# gemma.sh — minimal client to the local llama-server (Gemma 4, OpenAI API).
#
# Usage:
#   gemma.sh [--system "<sys>"] [--max N] [--temp T] [--json] "<user prompt>"
#   echo "<user prompt>" | gemma.sh [flags]        # user prompt may come from stdin
#
# Exit codes:
#   0  success, completion printed to stdout
#   3  server unreachable (R4: caller should degrade silently to a normal turn)
#   4  request failed / empty completion
#
# Keep calls SHORT on the critical path (predict.sh): few tokens, low timeout.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

SYSTEM=""
MAX="${SPEC_MAX_TOKENS:-64}"
TEMP="${SPEC_TEMP:-0.2}"
RAW_JSON=0
USER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --system) SYSTEM="$2"; shift 2 ;;
    --max)    MAX="$2";    shift 2 ;;
    --temp)   TEMP="$2";   shift 2 ;;
    --json)   RAW_JSON=1;  shift ;;
    --)       shift; USER="$*"; break ;;
    *)        USER="$1";   shift ;;
  esac
done

# Allow the user prompt on stdin (when not a tty and none given as arg).
if [ -z "$USER" ] && [ ! -t 0 ]; then
  USER="$(cat)"
fi
[ -n "$USER" ] || { echo "gemma.sh: empty prompt" >&2; exit 4; }

# R4: bail quietly if the server is down so hooks never error.
spec_server_up || exit 3

# Build messages array (system optional).
messages="$(jq -cn --arg s "$SYSTEM" --arg u "$USER" \
  'if ($s|length)>0 then [{role:"system",content:$s},{role:"user",content:$u}]
   else [{role:"user",content:$u}] end')"

payload="$(jq -cn \
  --arg model "$SPEC_MODEL" \
  --argjson messages "$messages" \
  --argjson max "$MAX" \
  --argjson temp "$TEMP" \
  '{model:$model, messages:$messages, max_tokens:$max, temperature:$temp, stream:false}')"

resp="$(curl -fsS -m "$SPEC_TIMEOUT" \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "$SPEC_BASE/v1/chat/completions" 2>/dev/null)" || exit 4

[ -n "$resp" ] || exit 4

if [ "$RAW_JSON" -eq 1 ]; then
  printf '%s\n' "$resp"
  exit 0
fi

content="$(printf '%s' "$resp" | jq -r '.choices[0].message.content // empty' 2>/dev/null)"
[ -n "$content" ] || exit 4
printf '%s\n' "$content"
