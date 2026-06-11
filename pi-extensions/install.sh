#!/usr/bin/env bash
# Symlink each pi extension in this repo into ~/.pi/agent/extensions/.
# pi resolves node_modules up the real directory tree, so the shared
# node_modules here serves every symlinked extension.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${PI_EXT_DIR:-$HOME/.pi/agent/extensions}"
mkdir -p "$DEST"

if [ ! -d "$SRC/node_modules" ]; then
  echo "node_modules missing — run 'npm install' in $SRC first." >&2
  exit 1
fi

for dir in "$SRC"/*/; do
  name="$(basename "$dir")"
  [ -f "$dir/index.ts" ] || continue          # only real extensions
  target="$DEST/$name"
  if [ -L "$target" ] || [ -e "$target" ]; then
    echo "skip $name (already exists at $target — remove it to relink)"
    continue
  fi
  ln -s "${dir%/}" "$target"
  echo "linked $name -> $target"
done
echo "Done. Restart or /reload pi to pick up changes."
