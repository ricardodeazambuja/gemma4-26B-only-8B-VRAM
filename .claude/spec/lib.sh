#!/usr/bin/env bash
# lib.sh — shared helpers for the local Gemma draft tier.
# Sourced by gemma.sh / draft.sh / ensure-server.sh / stats.sh.
#
# Design notes (see docs/PRD-speculative-agent.md):
#   - Everything degrades gracefully when llama-server is down (R4): callers check
#     return codes and fall through to a normal turn, never erroring the command.
#   - Runtime state (lock, stats.jsonl) lives under .claude/spec/ and is gitignored.

# --- Resolve our own directory regardless of caller's cwd -------------------
SPEC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
export SPEC_DIR
CACHE_DIR="$SPEC_DIR/cache"
STATS_FILE="$SPEC_DIR/stats.jsonl"
mkdir -p "$CACHE_DIR" 2>/dev/null || true

# --- Server config (env-overridable, matches config/pi-provider.json) -------
SPEC_HOST="${SPEC_HOST:-${HOST:-127.0.0.1}}"
SPEC_PORT="${SPEC_PORT:-${PORT:-8080}}"
SPEC_MODEL="${SPEC_MODEL:-gemma-4-26b-a4b-qat}"
SPEC_BASE="http://${SPEC_HOST}:${SPEC_PORT}"

# Default generation budget. /gemma-draft raises SPEC_TIMEOUT itself — manual calls
# are off any critical path, so a real multi-second generation is fine.
SPEC_CONNECT_TIMEOUT="${SPEC_CONNECT_TIMEOUT:-1}"   # seconds to establish connection
SPEC_TIMEOUT="${SPEC_TIMEOUT:-8}"                   # seconds total for a generation

# --- Helpers ----------------------------------------------------------------

# spec_server_up -> 0 if /health is OK, non-zero otherwise (quiet).
spec_server_up() {
  curl -fsS -m "$SPEC_CONNECT_TIMEOUT" "$SPEC_BASE/health" >/dev/null 2>&1
}

# spec_session_rollover -> on the first /gemma-draft call of a Claude Code session,
# archive the previous sessions' stats and start fresh, so /spec-stats reports the
# current session only. Keyed on CLAUDE_CODE_SESSION_ID; outside Claude Code (var
# unset) stats just keep accumulating. Best-effort like spec_log: never errors.
spec_session_rollover() {
  local sid="${CLAUDE_CODE_SESSION_ID:-}" marker="$SPEC_DIR/last-session"
  [ -n "$sid" ] || return 0
  [ "$(cat "$marker" 2>/dev/null)" = "$sid" ] && return 0
  if [ -s "$STATS_FILE" ]; then
    cat "$STATS_FILE" >> "$SPEC_DIR/stats.archive.jsonl" 2>/dev/null && : > "$STATS_FILE"
  fi
  printf '%s\n' "$sid" > "$marker" 2>/dev/null || true
}

# spec_log <json-object-string> -> append one event to stats.jsonl (best-effort).
# Timestamp is added by jq from the shell's date (cheap, not in the model path).
spec_log() {
  local obj="$1" ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
  printf '%s\n' "$(jq -cn --arg ts "$ts" --argjson e "$obj" '$e + {ts:$ts}' 2>/dev/null)" \
    >> "$STATS_FILE" 2>/dev/null || true
}
