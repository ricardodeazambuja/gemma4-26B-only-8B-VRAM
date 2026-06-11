#!/usr/bin/env bash
# predict.sh — UserPromptSubmit hook. The "start with A" entry point.
#
# Contract (Claude Code UserPromptSubmit hook):
#   - stdin  : JSON with {prompt, cwd, session_id, ...}
#   - stdout : plain text is INJECTED into Opus's context (this is the draft channel)
#   - exit 0 : always. We never block the prompt and never error the turn (R2/R4).
#
# Behaviour (token-economics first: an injected draft COSTS the big model input tokens,
# so we only inject when the draft plausibly replaces work):
#   1. Warm-cache HIT  -> inject the pre-computed speculative draft (consume-once). Instant.
#   1b. Fuzzy hit      -> actual prompt ≈ last prediction: inject the pre-draft (consume-once).
#   2. Cache MISS      -> if server up AND the prompt is short enough to be draftable
#                         (≤ SPEC_INLINE_MAXCHARS), one SHORT classify+draft call. Inject the
#                         draft ONLY if Gemma classified it easy — a wrong "hard" draft is
#                         pure token cost (observed live, see PRD R8).
#   3. Server down / long prompt -> inject nothing; the big model runs a normal turn.
#
# All paths log one event to stats.jsonl so /spec-stats can compute predictor hit rate.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

# --- Read hook input -------------------------------------------------------
input="$(cat)"
prompt="$(printf '%s' "$input" | jq -r '.prompt // empty' 2>/dev/null)"
[ -n "$prompt" ] || exit 0

# Bring up the optimal server in the background if it's down (non-blocking, single-flight).
# This turn still degrades; subsequent prompts get drafts once it's warm.
"$SPEC_DIR/ensure-server.sh" >/dev/null 2>&1 || true

key="$(spec_hash "$prompt")"
cache_file="$(spec_cache_path "$key")"

now_ms() { date +%s%3N 2>/dev/null || echo 0; }

# Async multimodal (G6): if the prompt mentions image paths, have Gemma OCR them in
# the background NOW — by the time Claude Reads one, the interception is instant.
mentions_image=0
while IFS= read -r img; do
  img="${img/#\~/$HOME}"
  if [ -f "$img" ]; then
    mentions_image=1
    setsid "$SPEC_DIR/describe.sh" --prewarm "$img" >/dev/null 2>&1 < /dev/null &
  fi
done < <(printf '%s\n' "$prompt" | grep -oE '(~|\.{1,2})?/?[A-Za-z0-9._/-]+\.(png|jpe?g|gif|webp|bmp)' | head -3)

# Speculative results are one-shot: expire stale files so an old draft can't
# masquerade as fresh work (R3), and consume entries on use so a hit can't re-fire.
spec_expire_cache

# record_outcome <src> <draft> — stash what we injected so the Stop worker can
# judge accepted-vs-superseded against the assistant's actual answer (PRD Q2).
record_outcome() {
  jq -n --arg k "$key" --arg src "$1" --arg d "$2" \
    '{key:$k, src:$src, draft:$d}' > "$PENDING_OUTCOME" 2>/dev/null || true
}

# --- 1. Warm-cache hit (branch prediction paid off) ------------------------
if [ -f "$cache_file" ]; then
  draft="$(jq -r '.draft // empty' "$cache_file" 2>/dev/null)"
  obs="$(jq -r '.obs   // empty' "$cache_file" 2>/dev/null)"
  kind="$(jq -r '.kind  // "draft"' "$cache_file" 2>/dev/null)"
  rm -f "$cache_file"                          # consume-once
  # The fuzzy record holds the SAME draft when its key matches — clear it too, or a
  # later vaguely-similar prompt would re-inject (and re-count) this consumed hit.
  if [ -f "$LAST_PREDICTION" ] && \
     [ "$(jq -r '.key // empty' "$LAST_PREDICTION" 2>/dev/null)" = "$key" ]; then
    rm -f "$LAST_PREDICTION"
  fi
  if [ -n "$draft" ]; then
    spec_log "$(jq -cn --arg k "$key" --arg kind "$kind" \
      '{event:"predict",result:"hit",key:$k,kind:$kind}')"
    record_outcome "hit" "$draft"
    cat <<EOF
[Gemma pre-computed this ($kind) — verify, use what's right, supersede the rest:]
$draft${obs:+

$obs}
EOF
    exit 0
  fi
fi

# --- 1b. Fuzzy branch-prediction hit (actual prompt ≈ last predicted) -------
if [ -f "$LAST_PREDICTION" ]; then
  predicted="$(jq -r '.predicted // empty' "$LAST_PREDICTION" 2>/dev/null)"
  pdraft="$(jq -r '.draft // empty' "$LAST_PREDICTION" 2>/dev/null)"
  pobs="$(jq -r '.obs    // empty' "$LAST_PREDICTION" 2>/dev/null)"
  if [ -n "$predicted" ] && [ -n "$pdraft" ]; then
    score="$(spec_similarity "$prompt" "$predicted")"
    if [ "${score:-0}" -ge "$SPEC_MATCH_MIN" ]; then
      # Verb gate: word-set overlap can't tell "delete X" from "show X" (both ≥34%
      # on the same X). Require the lead word to match too, unless disabled.
      lw_a="$(spec_lead_word "$prompt")"; lw_b="$(spec_lead_word "$predicted")"
      if [ "${SPEC_MATCH_VERB:-1}" = "1" ] && [ -n "$lw_a" ] && [ -n "$lw_b" ] && [ "$lw_a" != "$lw_b" ]; then
        spec_log "$(jq -cn --arg k "$key" --argjson s "${score:-0}" --arg a "$lw_a" --arg b "$lw_b" \
          '{event:"predict",result:"miss_verb_gate",key:$k,score:$s,prompt_verb:$a,predicted_verb:$b}')"
      else
        rm -f "$LAST_PREDICTION" "$(spec_cache_path "$(spec_hash "$predicted")")"   # consume-once
        spec_log "$(jq -cn --arg k "$key" --argjson s "${score:-0}" --arg p "$predicted" \
          '{event:"predict",result:"hit_predicted",key:$k,score:$s,predicted:$p}')"
        record_outcome "hit_predicted" "$pdraft"
        cat <<EOF
[Gemma predicted this request (${score}% match) and pre-drafted — verify, supersede if wrong:]
$pdraft${pobs:+

$pobs}
EOF
        exit 0
      fi
    fi
  fi
fi

# --- 2/3. Cache miss: cheap inline draft, only when worth attempting --------
if ! spec_server_up; then
  spec_log "$(jq -cn --arg k "$key" '{event:"predict",result:"miss_offline",key:$k}')"
  exit 0
fi

# Long prompts are practically never "easy" one-liners: a draft would be wrong
# (pure token cost) and the call adds seconds to the critical path. Skip inline.
if [ "${#prompt}" -gt "${SPEC_INLINE_MAXCHARS:-240}" ]; then
  spec_log "$(jq -cn --arg k "$key" --argjson n "${#prompt}" \
    '{event:"predict",result:"miss_skipped_long",key:$k,chars:$n}')"
  exit 0
fi

# Image-mention prompts: the work IS looking at the image — a text draft can't help,
# and it would fight the prewarm OCR for the GPU. The prewarm is the speculation here.
if [ "$mentions_image" = 1 ]; then
  spec_log "$(jq -cn --arg k "$key" '{event:"predict",result:"miss_image_prewarm",key:$k}')"
  exit 0
fi

SYS='You are the fast DRAFT tier in front of a stronger model. Be terse. Respond EXACTLY as:
DIFFICULTY: <easy or hard — pick exactly one>
DRAFT: <if easy: the direct answer in 1-3 sentences. if hard: leave empty>'

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

diff_line="$(printf '%s' "$out" | grep -i -m1 '^DIFFICULTY:' | sed 's/^[Dd][Ii][Ff][Ff][Ii][Cc][Uu][Ll][Tt][Yy]:[[:space:]]*//' | tr '[:upper:]' '[:lower:]')"
draft_line="$(printf '%s' "$out" | sed -n 's/^DRAFT:[[:space:]]*//Ip' | head -c 600)"

# Inject ONLY easy drafts with content: a hard/empty draft is pure input-token cost
# for the big model (R8). Hard prompts are logged and left to the big model untouched.
if [ "${diff_line%%[ |]*}" = "easy" ] && [ -n "$draft_line" ]; then
  spec_log "$(jq -cn --arg k "$key" --argjson ms "$elapsed" \
    '{event:"predict",result:"miss_drafted",key:$k,difficulty:"easy",ms:$ms}')"
  record_outcome "miss_drafted" "$draft_line"
  cat <<EOF
[Gemma judged this easy and drafted (${elapsed}ms) — if correct, confirm it instead of redoing the work:]
$draft_line
EOF
else
  spec_log "$(jq -cn --arg k "$key" --arg d "${diff_line:-unknown}" --argjson ms "$elapsed" \
    '{event:"predict",result:"miss_hard",key:$k,difficulty:$d,ms:$ms}')"
fi
exit 0
