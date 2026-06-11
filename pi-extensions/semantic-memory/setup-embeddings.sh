#!/usr/bin/env bash
# Set up the embedding backend that semantic-memory needs, the easy way: pull
# EmbeddingGemma into the already-running Ollama and verify the extension's own
# embed() gets a real vector back. Idempotent — safe to re-run.
#
# Usage:  ./setup-embeddings.sh
# Env:    OLLAMA_HOST (default http://127.0.0.1:11434)
#         EMBED_MODEL (default embeddinggemma)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
EMBED_MODEL="${EMBED_MODEL:-embeddinggemma}"

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

cat <<EOF

== Done. To make pi use it, put these in pi's environment ==
    export PI_EMBED_URL=$OLLAMA_HOST/v1/embeddings
    export PI_EMBED_MODEL=$EMBED_MODEL
(e.g. add to your shell rc, or pi's settings env. Then /reload pi.)
EOF
