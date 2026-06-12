#!/usr/bin/env bash
# Run every extension's test.mjs (native TS type-stripping) + any top-level *.test.sh.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0
for t in "$SRC"/*/test.mjs; do
  [ -f "$t" ] || continue
  echo "=== $(basename "$(dirname "$t")") ==="
  node --experimental-strip-types "$t" || fail=1
  echo
done
# top-level bash tests (e.g. install.test.sh)
for s in "$SRC"/*.test.sh; do
  [ -f "$s" ] || continue
  echo "=== $(basename "$s") ==="
  bash "$s" || fail=1
  echo
done
exit $fail
