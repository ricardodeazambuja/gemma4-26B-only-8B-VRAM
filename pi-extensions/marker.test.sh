#!/usr/bin/env bash
# Guards the shared <reminder> marker convention. Every tail-injecting extension wraps its
# injected guidance in byte-identical delimiters, because the SINGLE grounding ANCHOR note
# ("blocks wrapped in <reminder>…</reminder> are automated context, not the user") describes them
# all. A stray space or newline drift in one copy would silently break that contract for the model.
# This imports each extension's exported wrapReminder and asserts they emit the exact same bytes.
set -uo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SRC" node --experimental-strip-types --input-type=module -e '
  const base = process.env.SRC;
  const exts = ["grounding", "plan", "goal", "semantic-memory", "compaction-notice", "interrupt-notice"];
  const want = "<reminder>\nX\n</reminder>";
  let fail = 0;
  for (const e of exts) {
    const m = await import(`${base}/${e}/index.ts`);
    if (typeof m.wrapReminder !== "function") {
      console.log(`  ✗ ${e} does not export wrapReminder`); fail = 1; continue;
    }
    const got = m.wrapReminder("X");
    if (got === want) console.log(`  ✓ ${e} marker is byte-identical`);
    else { console.log(`  ✗ ${e} marker drifted: ${JSON.stringify(got)}`); fail = 1; }
  }
  process.exit(fail);
'
