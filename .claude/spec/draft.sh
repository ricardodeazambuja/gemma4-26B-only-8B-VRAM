#!/usr/bin/env bash
# draft.sh — backend of the /gemma-draft slash command: the ONLY entry point to the
# local Gemma draft tier (the always-on hook pipeline was retired in favour of this
# explicit, user-invoked command — see docs/PRD-speculative-agent.md, R11).
#
# Usage: draft.sh "<task — may mention image paths to OCR/describe locally>"
#
# Behaviour:
#   * Server down  -> kick the auto-launcher (non-blocking) and say so; never errors.
#   * Image paths in the task -> gemma.sh --image per image (OCR/describe, 0 big-model
#     image tokens), up to 3 images.
#   * Otherwise    -> one text draft from the cheap tier.
# Always exits 0: the command's output is advisory context, never a failure.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

task="$*"
if [ -z "${task// /}" ]; then
  echo "usage: /gemma-draft <task>   (include image paths to OCR/describe them locally)"
  exit 0
fi

# First real call of this session: archive older stats so /spec-stats is per-session.
spec_session_rollover

# Manual calls are off the critical path — allow a real generation, not the old
# 8 s hook budget. ~23 tok/s means a 512-token draft can take ~30 s.
export SPEC_TIMEOUT="${SPEC_DRAFT_TIMEOUT:-120}"

now_ms() { date +%s%3N 2>/dev/null || echo 0; }

if ! spec_server_up; then
  "$SPEC_DIR/ensure-server.sh" >/dev/null 2>&1 || true
  spec_log '{"event":"draft","result":"offline"}'
  echo "Gemma server is down — auto-launch kicked off (healthy in ~40 s; log: /tmp/gemma4-server.log)."
  echo "Proceed with the task normally, or re-run /gemma-draft shortly."
  exit 0
fi

# Image paths mentioned in the task -> local OCR/describe (max 3, existing files only).
imgs=()
while IFS= read -r img; do
  img="${img/#\~/$HOME}"
  [ -f "$img" ] && imgs+=("$img")
done < <(printf '%s\n' "$task" | grep -oE '(~|\.{1,2})?/?[A-Za-z0-9._/-]+\.(png|jpe?g|gif|webp|bmp)' | head -3)

t0="$(now_ms)"
if [ "${#imgs[@]}" -gt 0 ]; then
  ok=0
  for img in "${imgs[@]}"; do
    echo "── Gemma read of $img ──"
    if SPEC_MAX_TOKENS="${SPEC_DRAFT_MAX:-768}" "$SPEC_DIR/gemma.sh" --image "$img" "$task"; then
      ok=$((ok + 1))
    else
      echo "(local OCR failed for $img — Read the image normally instead)"
    fi
  done
  spec_log "$(jq -cn --argjson n "${#imgs[@]}" --argjson ok "$ok" --argjson ms "$(( $(now_ms) - t0 ))" \
    '{event:"draft",mode:"image",images:$n,ok:$ok,ms:$ms}')"
  exit 0
fi

SYS='You are the fast local DRAFT tier in front of a stronger model with file and shell
access. Give your best direct attempt at the task — terse, concrete, no preamble, no
questions back. If the task needs repo files you cannot see, outline the exact steps
the stronger model should take instead.'

if out="$(SPEC_MAX_TOKENS="${SPEC_DRAFT_MAX:-512}" \
         "$SPEC_DIR/gemma.sh" --system "$SYS" --temp 0.2 "$task")" && [ -n "$out" ]; then
  spec_log "$(jq -cn --argjson ms "$(( $(now_ms) - t0 ))" '{event:"draft",mode:"text",result:"ok",ms:$ms}')"
  printf '%s\n' "$out"
else
  spec_log "$(jq -cn --argjson ms "$(( $(now_ms) - t0 ))" '{event:"draft",mode:"text",result:"fail",ms:$ms}')"
  echo "(Gemma returned no draft — proceed with the task normally)"
fi
exit 0
