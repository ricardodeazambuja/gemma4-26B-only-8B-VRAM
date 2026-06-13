import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// interrupt-notice — when the user stops the agent mid-turn (Esc / abort), pi finalizes the
// half-written assistant message with stopReason "aborted" and goes idle, but injects nothing
// into the conversation. On the next turn the model can't tell its previous answer was cut off
// and may barrel on as if the work landed. This detects the aborted turn and injects one tail
// note so the model knows it was interrupted and waits for the user's steer.
//
// Detection rides two events for robustness: message_end (the aborted turn finalizes) and
// agent_end (the aborted run ends, carrying the conversation) — so it fires regardless of which
// a given pi build emits on abort. Design-rule fit: R1 (tail injection — the cached prompt prefix
// is untouched), R5 (one terse line), R2 (the note says what to do next). Self-limiting: fires
// once per interrupt.

export const NOTICE =
  "[Your previous response was interrupted by the user before it finished.] " +
  "Do not assume that work completed or that the task is done. Re-read the user's latest " +
  "message and act on it; if they didn't add anything new, ask what they want next.";

// Exported so the test can drive the same one-flag state machine the handlers use.
export function makeInterruptTracker() {
  let pending = false;
  return {
    /** Latch when an assistant turn was aborted. Returns true only if THIS call armed it. */
    observe(role: string | undefined, stopReason: string | undefined): boolean {
      if (role === "assistant" && stopReason === "aborted" && !pending) {
        pending = true;
        return true;
      }
      return false;
    },
    /** True once after an interrupt, then clears — so the note fires exactly once. */
    consume(): boolean {
      if (!pending) return false;
      pending = false;
      return true;
    },
  };
}

// Optional file-based debug. Console output can corrupt a TUI, so (like stats/advisor) we log to
// a file. PI_INTERRUPT_DEBUG=1 -> default path; PI_INTERRUPT_DEBUG=/some/path -> that file.
const _dbgEnv = process.env.PI_INTERRUPT_DEBUG;
const DEBUG_LOG =
  _dbgEnv && _dbgEnv !== "0" && _dbgEnv !== "false"
    ? _dbgEnv.includes("/")
      ? _dbgEnv
      : join(tmpdir(), "interrupt-notice-debug.log")
    : null;
function dbg(msg: string): void {
  if (!DEBUG_LOG) return;
  try {
    appendFileSync(DEBUG_LOG, `${new Date().toISOString()} interrupt-notice: ${msg}\n`);
  } catch {
    /* never let debug logging break a turn */
  }
}

// agent_end hands the whole conversation; the most recent assistant message reflects the turn
// that just ended (aborted or not).
function lastAssistant(messages: unknown): { role?: string; stopReason?: string } | undefined {
  const arr = Array.isArray(messages) ? messages : [];
  for (let i = arr.length - 1; i >= 0; i--) {
    const m = arr[i] as { role?: string; stopReason?: string };
    if (m?.role === "assistant") return m;
  }
  return undefined;
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
  const tracker = makeInterruptTracker();

  // Primary detect: the aborted assistant turn is finalized.
  pi.on("message_end", async (event) => {
    const m = event.message as { role?: string; stopReason?: string };
    if (tracker.observe(m?.role, m?.stopReason)) dbg("armed via message_end (stopReason=aborted)");
  });

  // Backstop detect: the aborted run ends. Catches the abort even if a pi build does not emit
  // message_end for the aborted partial.
  pi.on("agent_end", async (event) => {
    const m = lastAssistant((event as { messages?: unknown }).messages);
    if (m && tracker.observe(m.role, m.stopReason)) dbg("armed via agent_end (stopReason=aborted)");
  });

  // Inject on the next LLM call, folded into the trailing user turn as a wrapped <reminder> so the
  // KV-cache prefix is untouched (R1) and the note never reads as a fresh user instruction.
  pi.on("context", async (event) => {
    if (!tracker.consume()) return;
    dbg("injected interrupt notice into next request");
    return { messages: foldReminder(event.messages as Msg[], NOTICE) };
  });
}
