#!/usr/bin/env bash
# review.sh — PostToolUse(Write|Edit) hook. "Check the big model's output and help."
#
# After Opus writes/edits a file, the local Gemma tier skims it and feeds Opus a terse
# second opinion (likely bugs / omissions) as extra context. Advisory only — never blocks.
#
# OFF by default to avoid noise/latency on every edit. Enable with: export SPEC_REVIEW=1
#
# Contract (Claude Code PostToolUse hook): stdin JSON {tool_name, tool_input{file_path}, ...};
# JSON stdout with hookSpecificOutput.additionalContext is added to Opus's context; exit 0.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

quiet() { exit 0; }   # emit nothing -> no extra context

[ "${SPEC_REVIEW:-0}" = "1" ] || quiet

input="$(cat)"
tool="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)"
case "$tool" in Write|Edit|MultiEdit) : ;; *) quiet ;; esac

path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -n "$path" ] && [ -f "$path" ] || quiet

# Skip binaries and large files (keep the review cheap and the call short).
case "${path,,}" in *.png|*.jpg|*.jpeg|*.gif|*.webp|*.bmp|*.pdf|*.gguf|*.bin) quiet ;; esac
lines="$(wc -l < "$path" 2>/dev/null || echo 0)"
[ "${lines:-0}" -le "${SPEC_REVIEW_MAXLINES:-400}" ] || quiet

spec_server_up || quiet

SYS='You are a terse code reviewer. Given a file, list at most 3 concrete likely bugs or omissions
as short bullets. If it looks fine, reply exactly: LGTM. No praise, no restating the code.'
content="$(head -c 12000 "$path")"
review="$(SPEC_MAX_TOKENS="${SPEC_REVIEW_MAX:-160}" "$SPEC_DIR/gemma.sh" --system "$SYS" --temp 0.2 \
          "File: $path
\`\`\`
$content
\`\`\`" 2>/dev/null)"
[ -n "$review" ] || quiet

# Don't spend Opus context when Gemma is happy.
printf '%s' "$review" | grep -qi '^[[:space:]]*LGTM' && { spec_log "$(jq -cn --arg p "$path" '{event:"review",verdict:"lgtm",path:$p}')"; quiet; }

spec_log "$(jq -cn --arg p "$path" '{event:"review",verdict:"notes",path:$p}')"
ctx="[speculative-agent · GEMMA REVIEW of $path] Local tier flagged (verify before acting):
$review"
jq -cn --arg c "$ctx" \
  '{hookSpecificOutput:{hookEventName:"PostToolUse",additionalContext:$c}}'
exit 0
