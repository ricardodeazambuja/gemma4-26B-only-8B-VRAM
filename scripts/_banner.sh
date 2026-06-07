#!/usr/bin/env bash
#
# _banner.sh — shared backend signal, sourced (not executed) by run-server.sh
# and start.sh so the CUDA / Vulkan / CPU banner and the backend-resolution
# logic live in exactly one place.
#
#   source "$REPO_ROOT/scripts/_banner.sh"
#   backend_banner "$(resolve_backend)"
#
# Colour is decided from the *sourcing* script's stdout, so run-server.sh
# backgrounded into a logfile stays plain while start.sh in a terminal is
# coloured — automatically, with no extra flags.

if [ -t 1 ]; then
  _RST=$'\e[0m'
  _FG_GREEN=$'\e[1;32m'; _FG_YELLOW=$'\e[1;33m'; _FG_RED=$'\e[1;31m'
else
  _RST=; _FG_GREEN=; _FG_YELLOW=; _FG_RED=
fi

# splash <color> <icon> <title> [subtitle]
# Full-width coloured rules with no right border, so multi-byte emoji can't
# misalign the box (the classic ANSI box-drawing pitfall).
splash() {
  local col="$1" icon="$2" title="$3" sub="${4:-}"
  local rule='════════════════════════════════════════════════════════════'
  printf '\n%s%s%s\n' "$col" "$rule" "$_RST"
  printf '%s  %s  %s%s\n' "$col" "$icon" "$title" "$_RST"
  [ -n "$sub" ] && printf '%s     %s%s\n' "$col" "$sub" "$_RST"
  printf '%s%s%s\n\n' "$col" "$rule" "$_RST"
}

# backend_banner <cuda|vulkan|cpu> [detail]
# The single source of truth for what each backend's banner looks like.
backend_banner() {
  case "$1" in
    cuda)   splash "$_FG_GREEN"  "🟢" "BACKEND: CUDA  —  GPU accelerated (fast path)"            "${2:-}" ;;
    vulkan) splash "$_FG_YELLOW" "🟡" "BACKEND: VULKAN  —  GPU (slow MoE path, ~5x under CUDA)"  "${2:-}" ;;
    cpu)    splash "$_FG_RED"    "🔴" "BACKEND: CPU ONLY  —  no GPU offload (very slow)"         "${2:-}" ;;
    *)      splash ""            "•"  "BACKEND: $1"                                              "${2:-}" ;;
  esac
}

# resolve_backend — echo the backend run-server.sh will use: $BACKEND if set,
# else cuda when the locally-built CUDA binary exists, else vulkan. Honours
# $CUDA_BIN / $REPO_ROOT if the caller set them (so start.sh and run-server.sh
# always agree).
resolve_backend() {
  if [ -n "${BACKEND:-}" ]; then
    echo "$BACKEND"; return
  fi
  local cuda_bin="${CUDA_BIN:-${REPO_ROOT:-.}/vendor/llama.cpp/build/bin/llama-server}"
  [ -x "$cuda_bin" ] && echo cuda || echo vulkan
}
