#!/usr/bin/env bash
# Run every extension's test.mjs with native TS type-stripping.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail=0
for t in "$SRC"/*/test.mjs; do
  [ -f "$t" ] || continue
  echo "=== $(basename "$(dirname "$t")") ==="
  node --experimental-strip-types "$t" || fail=1
  echo
done
exit $fail
