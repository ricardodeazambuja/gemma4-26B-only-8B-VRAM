#!/usr/bin/env bash
# speculate.sh — Stop hook. The speculative-execution / branch-prediction engine.
#
# When Opus finishes a turn, predict the user's NEXT request and pre-compute a draft
# answer for it, into the cache. If the user's next prompt matches the prediction
# (predict.sh checks lexical similarity), that's a branch-prediction HIT — instant.
#
# Critical property: this must NOT block or delay the user. We detach a worker with
# setsid and return immediately (exit 0). The slow Gemma call happens off the turn.
#
# Contract (Claude Code Stop hook): stdin JSON {transcript_path, session_id, ...}; exit 0.
set -uo pipefail
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
source "$(dirname -- "$SELF")/lib.sh"

# ----------------------------------------------------------------------------
# Worker mode: the detached process that actually does the speculation.
# ----------------------------------------------------------------------------
if [ "${1:-}" = "--worker" ]; then
  in_file="$2"
  trap 'rm -f "$in_file"' EXIT
  spec_server_up || exit 0                      # nothing to speculate with

  # Run where the session runs (hooks pass cwd) — keeps find/pre-exec project-scoped.
  hook_cwd="$(jq -r '.cwd // empty' "$in_file" 2>/dev/null)"
  [ -n "$hook_cwd" ] && [ -d "$hook_cwd" ] && cd "$hook_cwd" 2>/dev/null

  transcript="$(jq -r '.transcript_path // empty' "$in_file" 2>/dev/null)"
  [ -n "$transcript" ] && [ -f "$transcript" ] || exit 0

  # Best-effort: pull the last REAL user turn and assistant turn as plain text.
  # In Claude Code transcripts tool results also arrive as type=="user" entries
  # (content blocks of type "tool_result"), so "last user entry" is usually NOT the
  # user's prompt — take the last entry that yields non-empty TEXT instead.
  last_user="$(jq -rs '
      [ .[] | select(.type=="user") | .message.content
        | if type=="array" then [ .[]? | select(.type=="text") | .text ] | join("\n")
          elif type=="string" then . else "" end
        | select(length > 0) ] | last // ""' "$transcript" 2>/dev/null)"
  last_asst="$(jq -rs '
      [ .[] | select(.type=="assistant") | .message.content
        | if type=="array" then [ .[]? | select(.type=="text") | .text ] | join("\n")
          elif type=="string" then . else "" end
        | select(length > 0) ] | last // ""' "$transcript" 2>/dev/null)"
  [ -n "$last_user$last_asst" ] || exit 0

  # 1) Predict the next user request (one short line).
  PSYS='Given the last exchange in a coding session, predict the user'\''s single most likely NEXT
request. Output ONLY that request as one short imperative line. No preamble.'
  ctx="LAST USER:
${last_user:0:1200}

LAST ASSISTANT:
${last_asst:0:1200}"
  predicted="$(SPEC_MAX_TOKENS=48 "$SPEC_DIR/gemma.sh" --system "$PSYS" --temp 0.3 "$ctx" 2>/dev/null \
               | head -n1 | sed 's/^[-*0-9. ]*//')"
  [ -n "$predicted" ] || exit 0

  # 2) Speculatively draft an answer to that predicted request (read-only; safe to discard).
  # The draft is consumed by a coding AGENT with full file/shell access — never plead lack
  # of access (that draft is pure token waste); give the concrete answer or exact steps.
  DSYS='You draft for a coding agent that CAN read files and run commands. For the request, give
either the direct answer or the exact commands/steps the agent should run. 2-4 lines, no preamble,
never say you lack access or ask for files.'
  draft="$(SPEC_MAX_TOKENS="${SPEC_SPEC_MAX:-256}" "$SPEC_DIR/gemma.sh" --system "$DSYS" --temp 0.2 \
           "Context of the session:
$ctx

Draft for this likely next request: $predicted" 2>/dev/null)"
  [ -n "$draft" ] || exit 0

  # 2a) SPECULATIVE PRE-EXECUTION: run a FIXED set of repo-status commands whose output
  # tends to help whatever comes next, and embed it in the cached draft. On a branch-
  # prediction hit Claude gets real repo state with zero tool round trips.
  #
  # SECURITY: the command set is HARD-CODED here — Gemma does NOT choose commands, and no
  # transcript-derived text reaches a shell. (An earlier version let the model propose
  # commands from an allowlist; that was a prompt-injection → file-disclosure vector,
  # e.g. `head /etc/passwd` passes a naive allowlist. Removed entirely; see PRD R10.)
  # These are repo-scoped git inspections, run by direct exec (no shell), 5s, capped.
  obs=""
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    while IFS= read -r -d '' c; do
      read -ra words <<< "$c"
      out_c="$(timeout 5 "${words[@]}" 2>&1 | head -c 1200)" || true
      [ -n "$out_c" ] && obs="${obs}\$ ${c}
${out_c}
"
    done < <(printf '%s\0' \
              "git status --short --branch" \
              "git diff --stat" \
              "git log --oneline -8")
  fi
  if [ -n "$obs" ]; then
    draft="${draft}

[repo state, pre-fetched while idle — re-run if staleness matters:]
${obs}"
  fi

  # 2b) Async multimodal: pre-OCR recently-touched images in the project while the
  # user is idle, so a future Read interception is instant (cache keyed path+mtime;
  # --prewarm exits early when already warm).
  find . -maxdepth 3 \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) \
       -mmin -30 -size -8M 2>/dev/null | head -2 | while IFS= read -r img; do
    bash "$SPEC_DIR/describe.sh" --prewarm "$img"
  done

  # 3) Store: a fuzzy-match record (last_prediction) + an exact-hash cache entry.
  key="$(spec_hash "$predicted")"
  jq -n --arg p "$predicted" --arg d "$draft" --arg k "$key" \
    '{predicted:$p, draft:$d, key:$k, kind:"next-turn draft"}' > "$LAST_PREDICTION" 2>/dev/null || true
  jq -n --arg d "$draft" '{draft:$d, kind:"next-turn draft"}' > "$(spec_cache_path "$key")" 2>/dev/null || true
  spec_log "$(jq -cn --arg p "$predicted" --arg k "$key" '{event:"speculate",predicted:$p,key:$k}')"
  exit 0
fi

# ----------------------------------------------------------------------------
# Hook mode: stash stdin, detach the worker, return instantly.
# ----------------------------------------------------------------------------
input="$(cat)"
# Warm the optimal server if it's down (non-blocking, single-flight; shared with predict.sh).
"$(dirname -- "$SELF")/ensure-server.sh" >/dev/null 2>&1 || true
tmp="$(mktemp 2>/dev/null)" || exit 0
printf '%s' "$input" > "$tmp"
setsid bash "$SELF" --worker "$tmp" >/dev/null 2>&1 < /dev/null &
exit 0
