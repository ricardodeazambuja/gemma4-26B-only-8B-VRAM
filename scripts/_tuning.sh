#!/usr/bin/env bash
#
# _tuning.sh — shared store for the auto-tuned expert split, sourced (not
# executed) by benchmark-config.sh (the writer) and start.sh (the reader /
# prompt) so the "remember the user's choice" logic lives in one place.
#
#   source "$REPO_ROOT/scripts/_tuning.sh"
#   tune_set cuda 32768 22         # save: fastest fitting NCMOE for this key
#   tune_set cuda 262144 nofit     # save: nothing fit at this context
#   tune_get cuda 32768            # -> "22"   (echoes the stored value, or nothing)
#
# The cache is a tiny `backend:ctx=value` file. `value` is the chosen NCMOE
# (a number), or `nofit` (no split fit), or `declined` (the user said no — so
# start.sh stops asking). Keyed by backend+context because both move the
# fit/speed boundary; re-tune after changing the model or GPU.
#
# Location: repo-local and gitignored (it's machine-specific state). Override
# with TUNE_CACHE=/path. REPO_ROOT must be set by the sourcing script.

tune_cache_file() { printf '%s\n' "${TUNE_CACHE:-${REPO_ROOT:-.}/.gemma4-tuning}"; }

# tune_get <backend> <ctx>  — echo the stored value for the key (empty if none).
tune_get() {
  local f; f="$(tune_cache_file)"
  [ -f "$f" ] || return 0
  sed -n "s/^$1:$2=//p" "$f" | tail -1     # last write wins
}

# tune_set <backend> <ctx> <value>  — upsert key=value, preserving other keys.
tune_set() {
  local f key tmp; f="$(tune_cache_file)"; key="$1:$2"
  tmp="$(mktemp)"
  { [ -f "$f" ] && grep -v "^$key=" "$f" || true; } > "$tmp"
  printf '%s=%s\n' "$key" "$3" >> "$tmp"
  mv "$tmp" "$f"
}
