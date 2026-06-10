#!/usr/bin/env bash
# describe.sh — image → text via the local multimodal tier (G6: zero image tokens on Claude).
#
# Two modes:
#   hook (default)      PreToolUse(Read). If the target is an image and text is available
#                       (cached or via a synchronous Gemma call), DENY the raw read and hand
#                       Claude the text instead. Safe degrade (R5): not an image / Gemma down /
#                       OCR fails -> emit nothing (the normal Read proceeds).
#   --prewarm <path>    Async multimodal: OCR the image into the cache NOW (called in the
#                       background by predict.sh/speculate.sh while Claude works), so a later
#                       Read interception is instant instead of a 10-60s synchronous OCR.
#
# OCR cache: cache/img_<hash(path+mtime)>.json — an edited image re-OCRs naturally.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

OCR_SYS='You are a vision pre-processor in front of a stronger model. Produce a faithful, complete
text rendering so the next model needs no pixels: (1) one-paragraph description, (2) verbatim OCR
of all visible text, (3) any tables/code/diagrams as text. Do not speculate beyond what is visible.'

# do_ocr <path> -> OCR text on stdout (non-empty), or rc!=0.
do_ocr() {
  SPEC_MAX_TOKENS="${SPEC_IMAGE_MAX:-1024}" SPEC_TIMEOUT="${SPEC_IMAGE_TIMEOUT:-60}" \
    "$SPEC_DIR/gemma.sh" --system "$OCR_SYS" --temp 0.1 --image "$1" \
    "Describe and OCR this image." 2>/dev/null
}

# ocr_cache_file <path> -> cache file path for this image at its current mtime.
ocr_cache_file() { printf '%s/img_%s.json' "$CACHE_DIR" "$(spec_image_key "$1")"; }

# --- prewarm mode: fill the OCR cache in the background ----------------------
if [ "${1:-}" = "--prewarm" ]; then
  path="${2:-}"
  [ -n "$path" ] && [ -f "$path" ] && spec_is_image "$path" || exit 0
  cfile="$(ocr_cache_file "$path")"
  [ -f "$cfile" ] && exit 0                  # already warm for this mtime
  spec_server_up || exit 0
  text="$(do_ocr "$path")" || exit 0
  [ -n "$text" ] || exit 0
  jq -n --arg p "$path" --arg t "$text" '{path:$p, text:$t}' > "$cfile" 2>/dev/null || true
  spec_log "$(jq -cn --arg p "$path" '{event:"image_prewarm",path:$p}')"
  exit 0
fi

# --- hook mode (PreToolUse Read) ---------------------------------------------
allow() { exit 0; }   # emit no decision -> Claude Code proceeds with the Read

input="$(cat)"
tool="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)"
[ "$tool" = "Read" ] || allow

path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -n "$path" ] && [ -f "$path" ] || allow
spec_is_image "$path" || allow               # PDFs/notebooks/text stay with Claude

# 1) Cache hit (prewarmed in the background) -> instant, no model call.
cfile="$(ocr_cache_file "$path")"
cached=false
if [ -f "$cfile" ]; then
  desc="$(jq -r '.text // empty' "$cfile" 2>/dev/null)"
  [ -n "$desc" ] && cached=true
fi

# 2) Cache miss -> synchronous OCR, only if Gemma can actually do it (R5).
if [ "$cached" = false ]; then
  spec_server_up || allow
  desc="$(do_ocr "$path")" || allow
  [ -n "$desc" ] || allow
  jq -n --arg p "$path" --arg t "$desc" '{path:$p, text:$t}' > "$cfile" 2>/dev/null || true
fi

bytes="$(wc -c < "$path" 2>/dev/null || echo 0)"
spec_log "$(jq -cn --arg p "$path" --argjson b "${bytes:-0}" --argjson c "$cached" \
  '{event:"image_offload",path:$p,bytes:$b,cached:$c}')"

reason="[Gemma read this image locally (cached: $cached) — use this text as its contents; no image tokens spent.]
File: $path
$desc"

jq -cn --arg r "$reason" \
  '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
exit 0
