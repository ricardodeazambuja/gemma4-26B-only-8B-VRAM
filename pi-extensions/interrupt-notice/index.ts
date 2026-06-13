import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// interrupt-notice — when the user stops the agent mid-turn (Esc / abort), pi finalizes the
// half-written assistant message with stopReason "aborted" and goes idle, but injects nothing
// into the conversation. On the next turn the model can't tell its previous answer was cut off
// and may barrel on as if the work landed. This detects the aborted turn and injects one tail
// note so the model knows it was interrupted and waits for the user's steer.
//
// Design-rule fit: R1 (tail injection — the cached prompt prefix is untouched), R5 (one terse
// line), R2 (the note says what to do next). Self-limiting: fires once per interrupt.

export const NOTICE =
  "[Your previous response was interrupted by the user before it finished.] " +
  "Do not assume that work completed or that the task is done. Re-read the user's latest " +
  "message and act on it; if they didn't add anything new, ask what they want next.";

// Exported so the test can drive the same one-flag state machine the handlers use.
export function makeInterruptTracker() {
  let pending = false;
  return {
    /** Call on every finalized message. Latches when an assistant turn was aborted. */
    observe(role: string | undefined, stopReason: string | undefined): void {
      if (role === "assistant" && stopReason === "aborted") pending = true;
    },
    /** True once after an interrupt, then clears — so the note fires exactly once. */
    consume(): boolean {
      if (!pending) return false;
      pending = false;
      return true;
    },
  };
}

export default function (pi: ExtensionAPI) {
  const tracker = makeInterruptTracker();

  // Detect: a finalized assistant turn whose stopReason is "aborted" == the user hit stop.
  // (StopReason in @earendil-works/pi-ai: stop | length | toolUse | error | aborted.)
  pi.on("message_end", async (event) => {
    const m = event.message as { role?: string; stopReason?: string };
    tracker.observe(m?.role, m?.stopReason);
  });

  // Inject on the next LLM call, at the TAIL so the KV-cache prefix is untouched (R1).
  pi.on("context", async (event) => {
    if (!tracker.consume()) return;
    const note = { role: "user" as const, content: [{ type: "text" as const, text: NOTICE }] };
    return { messages: [...event.messages, note] };
  });
}
