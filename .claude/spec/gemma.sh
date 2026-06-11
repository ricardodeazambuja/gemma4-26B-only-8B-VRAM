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
# Default budgets are small; callers raise SPEC_MAX_TOKENS / SPEC_TIMEOUT as needed.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

SYSTEM=""
MAX="${SPEC_MAX_TOKENS:-64}"
TEMP="${SPEC_TEMP:-0.2}"
RAW_JSON=0
IMAGE=""
# Gemma 4 QAT ships with reasoning ON (--jinja, thinking=1). For the cheap draft tier
# that just burns tokens/latency and leaves message.content empty under small budgets,
# so we disable it by default via chat_template_kwargs.enable_thinking=false. Re-enable
# with --think or SPEC_THINK=1.
THINK="${SPEC_THINK:-0}"
USER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --system) SYSTEM="$2"; shift 2 ;;
    --max)    MAX="$2";    shift 2 ;;
    --temp)   TEMP="$2";   shift 2 ;;
    --image)  IMAGE="$2";  shift 2 ;;
    --think)  THINK=1;     shift ;;
    --json)   RAW_JSON=1;  shift ;;
    --)       shift; USER="$*"; break ;;
    *)        USER="$1";   shift ;;
  esac
done
[ "$THINK" = "1" ] && THINK_JSON=true || THINK_JSON=false

# Allow the user prompt on stdin (when not a tty and none given as arg).
if [ -z "$USER" ] && [ ! -t 0 ]; then
  USER="$(cat)"
fi
# With an image, text is optional — default to a describe/OCR instruction.
if [ -z "$USER" ] && [ -n "$IMAGE" ]; then
  USER="Describe this image in full and transcribe ALL visible text verbatim (OCR)."
fi
[ -n "$USER" ] || { echo "gemma.sh: empty prompt" >&2; exit 4; }

# R4: bail quietly if the server is down so hooks never error.
spec_server_up || exit 3

# Build the user message content. With --image, use OpenAI vision format so Gemma 4's
# multimodal input handles the picture locally (G6: keeps image tokens off Opus).
if [ -n "$IMAGE" ]; then
  [ -f "$IMAGE" ] || { echo "gemma.sh: image not found: $IMAGE" >&2; exit 4; }
  b64="$(base64 -w0 "$IMAGE" 2>/dev/null || base64 "$IMAGE" 2>/dev/null | tr -d '\n')"
  [ -n "$b64" ] || { echo "gemma.sh: failed to encode image" >&2; exit 4; }
  case "${IMAGE,,}" in
    *.png) mime=image/png ;; *.jpg|*.jpeg) mime=image/jpeg ;;
    *.gif) mime=image/gif ;; *.webp) mime=image/webp ;;
    *.bmp) mime=image/bmp ;; *) mime=image/png ;;
  esac
  user_content="$(jq -cn --arg t "$USER" --arg url "data:$mime;base64,$b64" \
    '[{type:"text",text:$t},{type:"image_url",image_url:{url:$url}}]')"
else
  user_content="$(jq -cn --arg u "$USER" '$u')"
fi

messages="$(jq -cn --arg s "$SYSTEM" --argjson uc "$user_content" \
  'if ($s|length)>0 then [{role:"system",content:$s},{role:"user",content:$uc}]
   else [{role:"user",content:$uc}] end')"

payload="$(jq -cn \
  --arg model "$SPEC_MODEL" \
  --argjson messages "$messages" \
  --argjson max "$MAX" \
  --argjson temp "$TEMP" \
  --argjson think "$THINK_JSON" \
  '{model:$model, messages:$messages, max_tokens:$max, temperature:$temp, stream:false}
   + (if $think then {} else {chat_template_kwargs:{enable_thinking:false}} end)')"

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
# Fallback: if content is empty (e.g. thinking was on and ate the budget), use the
# reasoning text rather than returning nothing.
[ -n "$content" ] || content="$(printf '%s' "$resp" | jq -r '.choices[0].message.reasoning_content // empty' 2>/dev/null)"
[ -n "$content" ] || exit 4
printf '%s\n' "$content"
