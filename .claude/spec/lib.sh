#!/usr/bin/env bash
# lib.sh — shared helpers for the speculative-agent hooks.
# Sourced by gemma.sh / predict.sh / speculate.sh / review.sh.
#
# Design notes (see docs/PRD-speculative-agent.md):
#   - Everything degrades gracefully when llama-server is down (R4): callers check
#     return codes and fall through to a normal Opus turn, never erroring the hook.
#   - Cache + stats live under .claude/spec/ and are gitignored.

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

# Hard ceiling so a synchronous (critical-path) call can never hang the prompt.
SPEC_CONNECT_TIMEOUT="${SPEC_CONNECT_TIMEOUT:-1}"   # seconds to establish connection
SPEC_TIMEOUT="${SPEC_TIMEOUT:-8}"                   # seconds total for a generation

# --- Helpers ----------------------------------------------------------------

# spec_hash <string...> -> stable short key for cache/stats.
spec_hash() {
  printf '%s' "$*" | sha1sum | cut -c1-16
}

# spec_server_up -> 0 if /health is OK, non-zero otherwise (quiet).
spec_server_up() {
  curl -fsS -m "$SPEC_CONNECT_TIMEOUT" "$SPEC_BASE/health" >/dev/null 2>&1
}

# spec_log <json-object-string> -> append one event to stats.jsonl (best-effort).
# Timestamp is added by jq from the shell's date (cheap, not in the model path).
spec_log() {
  local obj="$1" ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)"
  printf '%s\n' "$(jq -cn --arg ts "$ts" --argjson e "$obj" '$e + {ts:$ts}' 2>/dev/null)" \
    >> "$STATS_FILE" 2>/dev/null || true
}

# spec_cache_path <key> -> path to the cached speculative payload for a prompt key.
spec_cache_path() {
  printf '%s/%s.json' "$CACHE_DIR" "$1"
}
