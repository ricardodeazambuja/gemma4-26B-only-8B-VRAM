#!/usr/bin/env bash
# Set up the embedding backend that semantic-memory needs, the easy way: pull
# EmbeddingGemma into the already-running Ollama and verify the extension's own
# embed() gets a real vector back. Idempotent — safe to re-run.
#
# Usage:  ./setup-embeddings.sh [--persist|--no-persist]
# Env:    OLLAMA_HOST (default http://127.0.0.1:11434)
#         EMBED_MODEL (default embeddinggemma)
#         PI_EMBED_CONFIG (default ~/.pi/agent/embed-config.json)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
EMBED_MODEL="${EMBED_MODEL:-embeddinggemma}"
CONFIG_PATH="${PI_EMBED_CONFIG:-$HOME/.pi/agent/embed-config.json}"
PERSIST="ask"
case "${1:-}" in
  --persist) PERSIST="yes" ;;
  --no-persist) PERSIST="no" ;;
esac

echo "== semantic-memory embedding setup =="

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: 'ollama' not found on PATH." >&2
  echo "Install Ollama (https://ollama.com), or use the llama.cpp path in README.md." >&2
  exit 1
fi

# Ensure the Ollama server is reachable (don't try to start it — that's the user's daemon).
if ! curl -sf --max-time 5 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Ollama server not reachable at $OLLAMA_HOST." >&2
  echo "Start it with 'ollama serve' (or your start.sh) and re-run." >&2
  exit 1
fi

# Pull the model (Ollama skips the download if it's already present).
if ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$EMBED_MODEL\(:latest\)\?"; then
  echo "Model '$EMBED_MODEL' already present — skipping pull."
else
  echo "Pulling '$EMBED_MODEL' (~620 MB)…"
  ollama pull "$EMBED_MODEL"
fi

# Verify through the extension's real embed() so a pass means it actually works.
echo "Verifying via semantic-memory/embed()…"
PI_EMBED_URL="$OLLAMA_HOST/v1/embeddings" PI_EMBED_MODEL="$EMBED_MODEL" \
  node --experimental-strip-types "$SRC/verify-embeddings.mjs"

# Persist so semantic-memory uses this backend on every pi launch — no env vars,
# no start.sh edits. The extension reads this file (env still overrides it).
EMBED_URL="$OLLAMA_HOST/v1/embeddings"
if [ "$PERSIST" = "ask" ]; then
  if [ -t 0 ]; then
    printf "\nPersist as semantic-memory's default backend?\n  writes %s\n  [Y/n] " "$CONFIG_PATH"
    read -r reply || reply=""
    case "$reply" in [Nn]*) PERSIST="no" ;; *) PERSIST="yes" ;; esac
  else
    PERSIST="no"  # non-interactive with no flag → don't touch config silently
    echo "(non-interactive: not persisting; pass --persist to write $CONFIG_PATH)"
  fi
fi

if [ "$PERSIST" = "yes" ]; then
  mkdir -p "$(dirname "$CONFIG_PATH")"
  printf '{\n  "url": "%s",\n  "model": "%s"\n}\n' "$EMBED_URL" "$EMBED_MODEL" > "$CONFIG_PATH"
  echo "Persisted → $CONFIG_PATH  (semantic-memory will use it automatically; /reload pi)"
else
  cat <<EOF
Not persisted. To use this backend, either re-run with --persist, or set in pi's env:
    export PI_EMBED_URL=$EMBED_URL
    export PI_EMBED_MODEL=$EMBED_MODEL
EOF
fi
