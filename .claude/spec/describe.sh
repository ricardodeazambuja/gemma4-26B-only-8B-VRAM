#!/usr/bin/env bash
# describe.sh — PreToolUse(Read) hook. G6: MUST offload images to the local tier.
#
# When Opus is about to Read an image file, intercept it: Gemma 4 (multimodal, local)
# OCRs/describes the image and we hand Opus that TEXT instead, denying the raw image
# read so Opus spends zero image tokens.
#
# Contract (Claude Code PreToolUse hook):
#   - stdin : JSON {tool_name, tool_input:{file_path,...}, ...}
#   - stdout: JSON. To substitute, we DENY the read and put Gemma's text in the reason,
#             which is fed back to Opus as context.
#   - exit 0 always.
#
# Safe degrade (R5): not an image / Gemma down / encode fails -> emit nothing (allow the
# normal Read). Never strand an image.
set -uo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

allow() { exit 0; }   # emit no decision -> Claude Code proceeds with the Read

input="$(cat)"
tool="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)"
[ "$tool" = "Read" ] || allow

path="$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null)"
[ -n "$path" ] && [ -f "$path" ] || allow

# Only images. (Read also handles PDFs/notebooks — leave those to Opus.)
case "${path,,}" in
  *.png|*.jpg|*.jpeg|*.gif|*.webp|*.bmp) : ;;
  *) allow ;;
esac

# Only intercept if Gemma can actually do the work (R5).
spec_server_up || allow

SYS='You are a vision pre-processor in front of a stronger model. Produce a faithful, complete
text rendering so the next model needs no pixels: (1) one-paragraph description, (2) verbatim OCR
of all visible text, (3) any tables/code/diagrams as text. Do not speculate beyond what is visible.'

desc="$(SPEC_MAX_TOKENS="${SPEC_IMAGE_MAX:-1024}" SPEC_TIMEOUT="${SPEC_IMAGE_TIMEOUT:-60}" \
        "$SPEC_DIR/gemma.sh" --system "$SYS" --temp 0.1 --image "$path" \
        "Describe and OCR this image." 2>/dev/null)"
[ -n "$desc" ] || allow   # encode/inference failed -> let the normal Read happen

bytes="$(wc -c < "$path" 2>/dev/null || echo 0)"
spec_log "$(jq -cn --arg p "$path" --argjson b "${bytes:-0}" \
  '{event:"image_offload",path:$p,bytes:$b}')"

reason="[speculative-agent · IMAGE OFFLOAD] Gemma (local multimodal tier) read this image so you
spend no image tokens. Raw Read suppressed — use the text below as the image's contents.

File: $path
--- Gemma vision output ---
$desc
--- end ---"

jq -cn --arg r "$reason" \
  '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$r}}'
exit 0
