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

# Where speculate.sh records its single most-recent next-turn prediction.
LAST_PREDICTION="$CACHE_DIR/last_prediction.json"

# Where predict.sh records the draft it injected this turn, so the Stop worker can
# judge (crudely) whether the big model accepted or superseded it (PRD Q2).
PENDING_OUTCOME="$CACHE_DIR/pending_outcome.json"

# Speculative results go stale fast (the repo moves on). Drop cache entries older
# than this many minutes (R3). Cheap: one find per prompt.
SPEC_CACHE_TTL_MIN="${SPEC_CACHE_TTL_MIN:-120}"
spec_expire_cache() {
  find "$CACHE_DIR" -maxdepth 1 -name '*.json' -mmin "+$SPEC_CACHE_TTL_MIN" -delete 2>/dev/null || true
}

# Image-OCR cache (async multimodal): keyed by path+mtime so an edited image
# invalidates naturally. Filled in the background by describe.sh --prewarm while
# the big model works; read instantly by the PreToolUse hook.
spec_image_key() {
  local p="$1" mt
  mt="$(stat -c %Y "$p" 2>/dev/null || echo 0)"
  spec_hash "img:$p:$mt"
}
spec_is_image() {
  case "${1,,}" in *.png|*.jpg|*.jpeg|*.gif|*.webp|*.bmp) return 0 ;; *) return 1 ;; esac
}

# Branch-prediction match threshold (% word overlap to count actual≈predicted as a HIT).
SPEC_MATCH_MIN="${SPEC_MATCH_MIN:-34}"

# spec_similarity <a> <b> -> integer 0..100, Jaccard overlap of lowercased word sets.
# Cheap, instant, dependency-free — the "was the branch predicted?" test.
spec_similarity() {
  awk -v a="$1" -v b="$2" 'BEGIN{
    n=split(tolower(a),A," "); m=split(tolower(b),B," ");
    for(i=1;i<=n;i++){gsub(/[^a-z0-9]/,"",A[i]); if(A[i]!="")sa[A[i]]=1}
    for(i=1;i<=m;i++){gsub(/[^a-z0-9]/,"",B[i]); if(B[i]!="")sb[B[i]]=1}
    for(k in sa){uni[k]=1; if(k in sb)inter++}
    for(k in sb)uni[k]=1
    u=0; for(k in uni)u++
    if(u==0){print 0; exit}
    printf "%d", (inter*100)/u
  }'
}

# spec_lead_word <text> -> first meaningful word, lowercased (the verb in an
# imperative request), skipping politeness/filler. Jaccard overlap is blind to
# WHICH word differs — "delete the tests" vs "show the tests" scores 66% — so
# predict.sh additionally requires the lead words to match (SPEC_MATCH_VERB=0
# disables the gate).
spec_lead_word() {
  awk -v s="$1" 'BEGIN{
    n=split(tolower(s), W, /[^a-z0-9]+/)
    for(i=1;i<=n;i++){
      w=W[i]
      if(w=="" || w=="please" || w=="now" || w=="ok" || w=="okay" || w=="just" ||
         w=="can" || w=="could" || w=="you" || w=="lets" || w=="let" || w=="us" ||
         w=="then" || w=="next" || w=="also" || w=="and" || w=="go" || w=="ahead")
        continue
      print w; exit
    }
  }'
}

# spec_containment <needle> <haystack> -> integer 0..100: % of needle's words that
# appear in haystack. Asymmetric on purpose — "did the long answer reuse the short
# draft's content?" is containment, not Jaccard (the answer's extra words would
# drown the union). Used by the outcome heuristic (accepted vs superseded).
spec_containment() {
  awk -v a="$1" -v b="$2" 'BEGIN{
    n=split(tolower(a),A," "); m=split(tolower(b),B," ");
    for(i=1;i<=n;i++){gsub(/[^a-z0-9]/,"",A[i]); if(A[i]!="")sa[A[i]]=1}
    for(i=1;i<=m;i++){gsub(/[^a-z0-9]/,"",B[i]); if(B[i]!="")sb[B[i]]=1}
    t=0; c=0; for(k in sa){t++; if(k in sb)c++}
    if(t==0){print 0; exit}
    printf "%d", (c*100)/t
  }'
}
