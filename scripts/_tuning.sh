#!/usr/bin/env bash
#
# _tuning.sh — shared store for the auto-tuned expert split, sourced (not
# executed) by benchmark-config.sh (the writer) and start.sh (the reader /
# prompt) so the "remember the user's choice" logic lives in one place.
#
#   source "$REPO_ROOT/scripts/_tuning.sh"
#   k="$(tune_key cuda 32768 q8_0)"   # build the canonical key: "cuda:32768:q8_0"
#   tune_set "$k" 22                  # save: fastest fitting NCMOE for this key
#   tune_set "$k" nofit               # save: nothing fit at this (ctx, kv) combo
#   tune_get "$k"                     # -> "22"  (echoes the stored value, or nothing)
#
# The cache is a tiny `key=value` file. `value` is the chosen NCMOE (a number),
# or `nofit` (no split fit), or `declined` (the user said no — so start.sh stops
# asking). The key is `backend:ctx:kvquant` (built by tune_key), because all
# three move the fit/speed boundary: the backend, the context size, and the
# KV-cache quant (which frees VRAM, so it changes the fitting expert split).
# `ctx` may be the literal `chosen`, used to remember a picked context.
# Re-tune after changing the model or GPU.
#
# The store is dimension-agnostic: tune_get/tune_set take one already-built key,
# so adding a tuning dimension is a tune_key change only.
#
# Location: repo-local and gitignored (it's machine-specific state). Override
# with TUNE_CACHE=/path. REPO_ROOT must be set by the sourcing script.

tune_cache_file() { printf '%s\n' "${TUNE_CACHE:-${REPO_ROOT:-.}/.gemma4-tuning}"; }

# tune_key <backend> <ctx|chosen> <kvquant>  — the canonical cache key.
# kvquant defaults to f16 (the unquantized KV cache) so callers can omit it.
tune_key() { printf '%s:%s:%s' "$1" "$2" "${3:-f16}"; }

# tune_get <key>  — echo the stored value for the key (empty if none).
tune_get() {
  local f; f="$(tune_cache_file)"
  [ -f "$f" ] || return 0
  sed -n "s/^$1=//p" "$f" | tail -1        # last write wins
}

# tune_set <key> <value>  — upsert key=value, preserving other keys.
tune_set() {
  local f tmp; f="$(tune_cache_file)"
  tmp="$(mktemp)"
  { [ -f "$f" ] && grep -v "^$1=" "$f" || true; } > "$tmp"
  printf '%s=%s\n' "$1" "$2" >> "$tmp"
  mv "$tmp" "$f"
}

# tune_migrate  — one-time upgrade of legacy two-part keys written before the
# KVQUANT dimension existed (`backend:ctx=value`) to the current three-part
# schema (`backend:ctx:f16=value`). Idempotent and cheap: a no-op once there are
# no legacy keys left. Preserves a user's existing tuned cache instead of
# silently invalidating it. Called at source time below.
tune_migrate() {
  local f tmp; f="$(tune_cache_file)"
  [ -f "$f" ] || return 0
  grep -qE '^[A-Za-z0-9]+:[A-Za-z0-9]+=' "$f" || return 0   # any legacy two-part key?
  tmp="$(mktemp)"
  sed -E 's/^([A-Za-z0-9]+:[A-Za-z0-9]+)=/\1:f16=/' "$f" > "$tmp" && mv "$tmp" "$f"
}

tune_migrate
