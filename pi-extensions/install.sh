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

# pi loads extensions with --preserve-symlinks, so module resolution starts from
# the symlink's location ($DEST/<name>), NOT the repo. Place a shared node_modules
# at $DEST so every symlinked extension resolves its deps (playwright, typebox,
# @earendil-works/*). One symlink back to the repo's install serves all of them.
if [ ! -e "$DEST/node_modules" ]; then
  ln -s "$SRC/node_modules" "$DEST/node_modules"
  echo "linked shared node_modules -> $DEST/node_modules"
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
