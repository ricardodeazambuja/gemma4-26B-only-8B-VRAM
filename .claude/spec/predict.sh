#!/usr/bin/env bash
# predict.sh — UserPromptSubmit hook. The "start with A" entry point.
#
# Contract (Claude Code UserPromptSubmit hook):
#   - stdin  : JSON with {prompt, cwd, session_id, ...}
#   - stdout : plain text is INJECTED into Opus's context (this is the draft channel)
#   - exit 0 : always. We never block the prompt and never error the turn (R2/R4).
#
# Behaviour:
#   1. Warm-cache HIT  -> inject the pre-computed speculative draft (from speculate.sh). Instant.
#   2. Cache MISS      -> if server up, a SHORT classify+hint call (kept off the critical-path
#                         budget by SPEC_PREDICT_MAX + bounded timeout). Inject the hint.
#   3. Server down     -> inject nothing; Opus runs a normal turn.
#
# All paths log one event to stats.jsonl so /spec-stats can compute predictor hit rate.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

# --- Read hook input -------------------------------------------------------
input="$(cat)"
prompt="$(printf '%s' "$input" | jq -r '.prompt // empty' 2>/dev/null)"
[ -n "$prompt" ] || exit 0

key="$(spec_hash "$prompt")"
cache_file="$(spec_cache_path "$key")"

now_ms() { date +%s%3N 2>/dev/null || echo 0; }

# --- 1. Warm-cache hit (branch prediction paid off) ------------------------
if [ -f "$cache_file" ]; then
  draft="$(jq -r '.draft // empty' "$cache_file" 2>/dev/null)"
  kind="$(jq -r '.kind  // "draft"' "$cache_file" 2>/dev/null)"
  if [ -n "$draft" ]; then
    spec_log "$(jq -cn --arg k "$key" --arg kind "$kind" \
      '{event:"predict",result:"hit",key:$k,kind:$kind}')"
    cat <<EOF
[speculative-agent · CACHE HIT] The local Gemma draft tier predicted this request last turn and
pre-computed a result ($kind). Treat it as a DRAFT to VERIFY — confirm if correct, supersede if not:
---
$draft
---
EOF
    exit 0
  fi
fi

# --- 1b. Fuzzy branch-prediction hit (actual prompt ≈ last predicted) -------
if [ -f "$LAST_PREDICTION" ]; then
  predicted="$(jq -r '.predicted // empty' "$LAST_PREDICTION" 2>/dev/null)"
  pdraft="$(jq -r '.draft // empty' "$LAST_PREDICTION" 2>/dev/null)"
  if [ -n "$predicted" ] && [ -n "$pdraft" ]; then
    score="$(spec_similarity "$prompt" "$predicted")"
    if [ "${score:-0}" -ge "$SPEC_MATCH_MIN" ]; then
      spec_log "$(jq -cn --arg k "$key" --argjson s "${score:-0}" --arg p "$predicted" \
        '{event:"predict",result:"hit_predicted",key:$k,score:$s,predicted:$p}')"
      cat <<EOF
[speculative-agent · BRANCH PREDICTED ${score}%] Last turn Gemma guessed your next request would be:
  "$predicted"
and pre-drafted this. Treat as a DRAFT to VERIFY — confirm if right, supersede if wrong:
---
$pdraft
---
EOF
      exit 0
    fi
  fi
fi

# --- 2/3. Cache miss: cheap inline draft if the server is up ----------------
if ! spec_server_up; then
  spec_log "$(jq -cn --arg k "$key" '{event:"predict",result:"miss_offline",key:$k}')"
  exit 0
fi

SYS='You are the fast DRAFT tier in front of a stronger model. Be terse. Respond EXACTLY as:
DIFFICULTY: easy|hard
DRAFT: <one to three sentences: a direct answer if easy, else the key approach/first step>'

t0="$(now_ms)"
out="$(SPEC_MAX_TOKENS="${SPEC_PREDICT_MAX:-96}" \
      "$SPEC_DIR/gemma.sh" --system "$SYS" --temp 0.2 "$prompt" 2>/dev/null)"
rc=$?
t1="$(now_ms)"
elapsed=$(( t1 - t0 ))

if [ $rc -ne 0 ] || [ -z "$out" ]; then
  spec_log "$(jq -cn --arg k "$key" --argjson ms "$elapsed" \
    '{event:"predict",result:"miss_nodraft",key:$k,ms:$ms}')"
  exit 0
fi

diff_line="$(printf '%s' "$out" | grep -i -m1 '^DIFFICULTY:' | sed 's/^[Dd][Ii][Ff][Ff][Ii][Cc][Uu][Ll][Tt][Yy]:[[:space:]]*//')"
spec_log "$(jq -cn --arg k "$key" --arg d "${diff_line:-unknown}" --argjson ms "$elapsed" \
  '{event:"predict",result:"miss_drafted",key:$k,difficulty:$d,ms:$ms}')"

cat <<EOF
[speculative-agent · DRAFT] The local Gemma draft tier took a fast first pass (${elapsed}ms).
Treat it as a DRAFT to VERIFY — confirm if correct, supersede/ignore if wrong:
---
$out
---
EOF
exit 0
