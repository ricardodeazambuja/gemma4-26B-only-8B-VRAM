import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// compaction-notice — when pi compacts the context (older turns replaced by a short summary to
// claw back the window), it injects nothing telling the model this happened. A small model then
// trusts its "memory" of detail that was actually summarized away — exact file contents, values,
// decisions, what was already tried — and acts on a stale recollection. This detects the
// compaction and injects one tail note on the next call, so the model knows the detail is gone
// and re-reads / re-derives instead of guessing.
//
// Detect on session_compact (fired AFTER compaction completes, so the summary is already in the
// message list); inject at the TAIL of the next context. Design-rule fit: R1 — compaction
// rewrites the prompt prefix by definition, so that turn already pays a full re-prefill; the tail
// note's marginal cost is ~zero. R5 — one terse line (with a concrete "~Nk summarized" anchor when
// available). R2 — it says what to do next, not just what happened. Self-limiting: fires once per
// compaction.

export function compactionNotice(tokensBefore?: number): string {
  const size =
    typeof tokensBefore === "number" && Number.isFinite(tokensBefore) && tokensBefore > 0
      ? ` (~${Math.round(tokensBefore / 1000)}k tokens of earlier turns were summarized)`
      : "";
  return (
    `[The conversation was just compacted${size}.] Earlier turns have been replaced by a short ` +
    `summary, so specific detail — exact file contents, values, decisions, what was already tried ` +
    `— may be lost or imprecise. Do not trust your memory of it: re-read the files, re-run the ` +
    `checks, or re-derive what you need before acting, and rely on the summary and the user's ` +
    `latest message for intent.`
  );
}

// Exported so the test can drive the same one-flag state machine the handlers use.
export function makeCompactionTracker() {
  let pending = false;
  let tokens: number | undefined;
  return {
    /** Latch a completed compaction. Returns true only if THIS call armed it (idempotent). */
    observe(tokensBefore?: number): boolean {
      if (pending) return false;
      pending = true;
      tokens = tokensBefore;
      return true;
    },
    /** {fired:true, tokensBefore} once after a compaction, then clears — so the note fires once. */
    consume(): { fired: boolean; tokensBefore?: number } {
      if (!pending) return { fired: false };
      pending = false;
      const t = tokens;
      tokens = undefined;
      return { fired: true, tokensBefore: t };
    },
  };
}

// Optional file-based debug. Console output can corrupt a TUI, so (like stats/advisor) we log to
// a file. PI_COMPACTION_DEBUG=1 -> default path; PI_COMPACTION_DEBUG=/some/path -> that file.
const _dbgEnv = process.env.PI_COMPACTION_DEBUG;
const DEBUG_LOG =
  _dbgEnv && _dbgEnv !== "0" && _dbgEnv !== "false"
    ? _dbgEnv.includes("/")
      ? _dbgEnv
      : join(tmpdir(), "compaction-notice-debug.log")
    : null;
function dbg(msg: string): void {
  if (!DEBUG_LOG) return;
  try {
    appendFileSync(DEBUG_LOG, `${new Date().toISOString()} compaction-notice: ${msg}\n`);
  } catch {
    /* never let debug logging break a turn */
  }
}

export default function (pi: ExtensionAPI) {
  const tracker = makeCompactionTracker();

  // Detect: compaction finished. compactionEntry.tokensBefore is the context size that was
  // summarized away, surfaced in the note as a concrete anchor.
  pi.on("session_compact", async (event) => {
    const tb = (event as { compactionEntry?: { tokensBefore?: number } }).compactionEntry
      ?.tokensBefore;
    if (tracker.observe(tb)) dbg(`armed via session_compact (tokensBefore=${tb ?? "?"})`);
  });

  // Inject on the next LLM call, at the TAIL so it sits after the compaction summary (R1).
  pi.on("context", async (event) => {
    const { fired, tokensBefore } = tracker.consume();
    if (!fired) return;
    dbg("injected compaction notice into next request");
    const note = {
      role: "user" as const,
      content: [{ type: "text" as const, text: compactionNotice(tokensBefore) }],
    };
    return { messages: [...event.messages, note] };
  });
}
