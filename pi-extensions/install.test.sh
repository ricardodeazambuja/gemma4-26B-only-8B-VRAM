#!/usr/bin/env bash
# Tests install.sh against a throwaway PI_EXT_DIR: add / skip / --force relink (incl. a
# stale real-directory copy, the web-search case) / --prune (ours only, not foreign links).
# Run: bash install.test.sh
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass + 1)); echo "  ✓ $1"; else fail=$((fail + 1)); echo "  ✗ $1"; fi; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export PI_EXT_DIR="$TMP/ext"

echo "baseline install:"
"$SRC/install.sh" >/dev/null 2>&1
ok "links an extension as a symlink"            '[ -L "$PI_EXT_DIR/grounding" ]'
ok "symlink resolves into the repo"             '[ "$(readlink "$PI_EXT_DIR/grounding")" = "$SRC/grounding" ]'
ok "links the shared node_modules"              '[ -L "$PI_EXT_DIR/node_modules" ]'

echo "idempotent skip (no flag):"
out="$("$SRC/install.sh" 2>&1)"
ok "a second run skips what already exists"      'echo "$out" | grep -q "skip grounding"'

echo "stale real-directory copy (the web-search case):"
rm "$PI_EXT_DIR/grounding"; mkdir -p "$PI_EXT_DIR/grounding"; : > "$PI_EXT_DIR/grounding/index.ts"
"$SRC/install.sh" >/dev/null 2>&1
ok "left alone without --force"                  '[ -d "$PI_EXT_DIR/grounding" ] && [ ! -L "$PI_EXT_DIR/grounding" ]'
"$SRC/install.sh" --force >/dev/null 2>&1
ok "--force replaces the copy with a symlink"    '[ -L "$PI_EXT_DIR/grounding" ]'
ok "--relink is an alias for --force"            'out="$("$SRC/install.sh" --relink 2>&1)"; echo "$out" | grep -q "Done"'

echo "--prune (ours only):"
ln -s "$SRC/nonexistent-ext" "$PI_EXT_DIR/ghost"   # ours: points into the repo, now stale
ln -s /tmp "$PI_EXT_DIR/foreign"                   # not ours: points elsewhere
"$SRC/install.sh" --prune >/dev/null 2>&1
ok "prunes our orphaned/broken symlink"          '[ ! -L "$PI_EXT_DIR/ghost" ] && [ ! -e "$PI_EXT_DIR/ghost" ]'
ok "leaves foreign symlinks untouched"           '[ -L "$PI_EXT_DIR/foreign" ]'
ok "keeps live extensions"                       '[ -L "$PI_EXT_DIR/grounding" ]'
ok "keeps the shared node_modules"               '[ -L "$PI_EXT_DIR/node_modules" ]'

echo "misc:"
ok "rejects an unknown option (exit 2)"          '"$SRC/install.sh" --nope >/dev/null 2>&1; [ "$?" -eq 2 ]'
ok "--help prints usage"                         '"$SRC/install.sh" --help 2>&1 | grep -q "Usage:"'

echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
