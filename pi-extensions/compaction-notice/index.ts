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

// ---- reminder marker (shared convention; see grounding's ANCHOR note) -----------------------
// Wrap the notice so the model can tell it from the user's own words, and fold it into the
// trailing user turn (or append a fresh user message when the tail is a toolResult/assistant) so
// it never reads as a new user instruction. Marker bytes MUST match every other extension's.
export const REMINDER_OPEN = "<reminder>\n";
export const REMINDER_CLOSE = "\n</reminder>";
export const wrapReminder = (text: string): string => `${REMINDER_OPEN}${text}${REMINDER_CLOSE}`;

type Msg = { role: string; content: unknown };
export function foldReminder(messages: Msg[], text: string): Msg[] {
  const block = { type: "text" as const, text: wrapReminder(text) };
  const out = messages.slice();
  const last = out[out.length - 1] as Msg | undefined;
  if (last && last.role === "user") {
    const prior = Array.isArray(last.content)
      ? (last.content as Array<{ type: string; text?: string }>)
      : [{ type: "text" as const, text: String(last.content ?? "") }];
    out[out.length - 1] = { ...last, content: [...prior, block] };
  } else {
    out.push({ role: "user", content: [block] } as Msg);
  }
  return out;
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

  // Inject on the next LLM call, folded into the trailing user turn as a wrapped <reminder> so it
  // sits after the compaction summary (R1) without impersonating the user.
  pi.on("context", async (event) => {
    const { fired, tokensBefore } = tracker.consume();
    if (!fired) return;
    dbg("injected compaction notice into next request");
    return { messages: foldReminder(event.messages as Msg[], compactionNotice(tokensBefore)) };
  });
}
