#!/usr/bin/env bash
# Symlink each pi extension in this repo into ~/.pi/agent/extensions/.
# pi resolves node_modules up the real directory tree, so the shared node_modules
# here serves every symlinked extension.
#
# Usage: ./install.sh [--force|--relink] [--prune]
#   --force, --relink   Replace existing links AND stale real-directory copies with
#                       fresh symlinks (sync to the repo) instead of skipping what
#                       exists. Fixes a stale copy or a moved repo in one command.
#   --prune             Remove OUR orphaned/broken symlinks in the dest — ones that
#                       point into this repo but no longer match a repo extension.
#                       Never touches extensions installed from elsewhere.
# Override the destination with PI_EXT_DIR=.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${PI_EXT_DIR:-$HOME/.pi/agent/extensions}"

FORCE=no
PRUNE=no
for a in "$@"; do
  case "$a" in
    --force|--relink) FORCE=yes ;;
    --prune)          PRUNE=yes ;;
    -h|--help)        sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "install.sh: unknown option '$a' (try --help)" >&2; exit 2 ;;
  esac
done

if [ ! -d "$SRC/node_modules" ]; then
  echo "node_modules missing — run 'npm install' in $SRC first." >&2
  exit 1
fi
mkdir -p "$DEST"

# True if $1 is a symlink whose (literal) target points into this repo ($SRC).
ours_link() {
  [ -L "$1" ] || return 1
  case "$(readlink "$1")" in "$SRC"/*) return 0 ;; *) return 1 ;; esac
}

# Shared node_modules symlink (pi loads with --preserve-symlinks, so every extension
# resolves its deps from here). See README for why this is required.
NM="$DEST/node_modules"
if [ -L "$NM" ] && { [ "$FORCE" = yes ] || [ ! -e "$NM" ]; }; then rm -f "$NM"; fi   # drop on --force or if broken
if [ ! -e "$NM" ] && [ ! -L "$NM" ]; then
  ln -s "$SRC/node_modules" "$NM"
  echo "linked shared node_modules -> $NM"
elif [ -d "$NM" ] && [ ! -L "$NM" ]; then
  echo "warning: $NM is a real directory, not a symlink — leaving it" >&2
fi

# Link each extension (any dir with an index.ts).
for dir in "$SRC"/*/; do
  name="$(basename "$dir")"
  [ -f "$dir/index.ts" ] || continue          # only real extensions
  target="$DEST/$name"
  if [ -L "$target" ] || [ -e "$target" ]; then
    if [ "$FORCE" = yes ]; then
      if [ -L "$target" ]; then rm -f "$target"
      else echo "replacing stale copy at $target"; rm -rf "$target"; fi
      ln -s "${dir%/}" "$target"
      echo "relinked $name -> $target"
    else
      echo "skip $name (already exists at $target — use --force to relink)"
    fi
    continue
  fi
  ln -s "${dir%/}" "$target"
  echo "linked $name -> $target"
done

# Optionally drop OUR symlinks whose repo extension is gone (orphans / broken links).
if [ "$PRUNE" = yes ]; then
  for entry in "$DEST"/*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue   # skip the literal glob when empty
    name="$(basename "$entry")"
    [ "$name" = node_modules ] && continue
    if ours_link "$entry" && [ ! -f "$SRC/$name/index.ts" ]; then
      rm -f "$entry"
      echo "pruned $name (no longer a repo extension)"
    fi
  done
fi

echo "Done. Restart or /reload pi to pick up changes."
