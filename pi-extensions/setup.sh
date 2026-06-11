#!/usr/bin/env bash
# One-shot installer for the whole pi-extensions set. Everything is installable
# from this repo: deps → symlink into pi → (optionally) the embedding backend.
#
# Usage:  ./setup.sh [--yes]
#   --yes   non-interactive: also pull + persist embeddings without prompting.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YES="no"; [ "${1:-}" = "--yes" ] && YES="yes"

echo "== pi-extensions setup =="

# 1. dependencies
if [ ! -d "$SRC/node_modules" ]; then
  echo "[1/3] installing npm deps…"
  ( cd "$SRC" && npm install )
else
  echo "[1/3] npm deps present."
fi

# 2. link extensions + shared node_modules into pi
echo "[2/3] linking extensions into ~/.pi/agent/extensions…"
"$SRC/install.sh"

# 3. embedding backend for semantic-memory (optional)
echo "[3/3] semantic-memory embedding backend"
do_embed="no"
if [ "$YES" = "yes" ]; then
  do_embed="yes"
elif [ -t 0 ]; then
  printf "  Pull EmbeddingGemma via Ollama and enable semantic recall now? [Y/n] "
  read -r reply || reply=""
  case "$reply" in [Nn]*) do_embed="no" ;; *) do_embed="yes" ;; esac
else
  echo "  (non-interactive without --yes: skipping; run semantic-memory/setup-embeddings.sh later)"
fi

if [ "$do_embed" = "yes" ]; then
  if [ "$YES" = "yes" ]; then
    "$SRC/semantic-memory/setup-embeddings.sh" --persist
  else
    "$SRC/semantic-memory/setup-embeddings.sh"   # will prompt about persisting
  fi
fi

cat <<EOF

== Done. Restart or /reload pi to pick up the extensions. ==
Tests:  ./run-tests.sh
EOF
