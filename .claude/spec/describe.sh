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

# --- prewarm-log mode: compute the Gemma log summary in the background -------
if [ "${1:-}" = "--prewarm-log" ]; then
  path="${2:-}"
  [ -n "$path" ] && [ -f "$path" ] || exit 0
  cfile="$CACHE_DIR/log_$(spec_image_key "$path").json"   # same path+mtime keying
  [ -f "$cfile" ] && exit 0
  spec_server_up || exit 0
  sample="$(head -c 3000 "$path"; echo; echo '...[middle omitted]...'; tail -c 3000 "$path")"
  LSYS='You summarize logs for a debugging agent. From this sample, state in <=6 terse
bullets: what process/run this is, recurring patterns, and any anomalies. No restating raw lines.'
  summary="$(SPEC_MAX_TOKENS=140 SPEC_TIMEOUT=45 "$SPEC_DIR/gemma.sh" --system "$LSYS" --temp 0.1 "$sample" 2>/dev/null || true)"
  [ -n "$summary" ] || exit 0
  jq -n --arg p "$path" --arg t "$summary" '{path:$p, text:$t}' > "$cfile" 2>/dev/null || true
  spec_log "$(jq -cn --arg p "$path" '{event:"log_prewarm",path:$p}')"
  exit 0
fi

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

# --- Large-log offload: digest instead of raw dump --------------------------
# A 1 MB log is ~250K input tokens for the big model, mostly repetition. Serve a
# lossless-leaning digest instead: exact head/tail + deterministic error-grep
# (+ a Gemma pattern summary when the server is up). ESCAPE HATCH: a Read with
# offset/limit passes through untouched, so exact ranges are always reachable.
if [ "${SPEC_LOG_OFFLOAD:-1}" = "1" ]; then
  case "${path,,}" in
    *.log|*.out)
      # *.out also matches compiled binaries (a.out style) — digesting one would
      # serve garbage. NUL byte in the first 4 KB -> not a text log, pass through.
      nul="$(head -c 4096 "$path" 2>/dev/null | tr -cd '\0' | wc -c)"
      [ "${nul:-0}" -gt 0 ] && allow
      has_range="$(printf '%s' "$input" | jq -r '(.tool_input.offset // .tool_input.limit // empty)' 2>/dev/null)"
      sz="$(wc -c < "$path" 2>/dev/null || echo 0)"
      if [ -z "$has_range" ] && [ "${sz:-0}" -ge $(( ${SPEC_LOG_MINKB:-64} * 1024 )) ]; then
        nlines="$(wc -l < "$path" 2>/dev/null || echo '?')"
        head_v="$(head -30 "$path")"
        tail_v="$(tail -30 "$path")"
        errs="$(grep -inE 'error|fail|exception|fatal|panic|traceback|warn' "$path" 2>/dev/null | head -40)"
        # Async-first: use a cached Gemma summary if one exists; otherwise serve the
        # deterministic digest NOW and compute the summary in the background for the
        # next read (same pattern as image prewarm — never stall the Read on the GPU).
        summary=""
        lfile="$CACHE_DIR/log_$(spec_image_key "$path").json"
        if [ -f "$lfile" ]; then
          summary="$(jq -r '.text // empty' "$lfile" 2>/dev/null)"
        else
          setsid "$SPEC_DIR/describe.sh" --prewarm-log "$path" >/dev/null 2>&1 < /dev/null &
        fi
        spec_log "$(jq -cn --arg p "$path" --argjson b "$sz" '{event:"log_offload",path:$p,bytes:$b}')"
        reason="[Large log digested locally (${sz} bytes, ${nlines} lines) — raw dump would burn ~$(( sz / 4 / 1000 ))K tokens. For exact ranges, Read again with offset/limit (passes through raw).]
--- first 30 lines (verbatim) ---
$head_v
--- last 30 lines (verbatim) ---
$tail_v
--- error/warn lines (grep -in, first 40) ---
${errs:-(none matched)}
${summary:+--- Gemma pattern summary ---
$summary}"
        jq -cn --arg r "$reason" \
          '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
        exit 0
      fi
      ;;
  esac
fi

spec_is_image "$path" || allow               # PDFs/notebooks/text stay with Claude

# Kill switch + escape hatch (parity with the log offload, R5): offload stays the
# default (G6), but SPEC_IMAGE_OFFLOAD=0 disables it, and a Read with offset/limit
# passes through raw — the way to see the actual pixels when the OCR text isn't
# enough (e.g. a dense diagram). Without this, a poor OCR is cached by path+mtime
# and the image is unreachable.
[ "${SPEC_IMAGE_OFFLOAD:-1}" = "1" ] || allow
img_range="$(printf '%s' "$input" | jq -r '(.tool_input.offset // .tool_input.limit // empty)' 2>/dev/null)"
[ -n "$img_range" ] && allow

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

reason="[Gemma read this image locally (cached: $cached) — use this text as its contents; no image tokens spent. If the text is insufficient (dense diagram, layout matters), Read again with any offset/limit to get the actual image.]
File: $path
$desc"

jq -cn --arg r "$reason" \
  '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
exit 0
