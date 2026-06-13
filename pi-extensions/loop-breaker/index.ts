import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createHash } from "node:crypto";

// loop-breaker — small models repeat a failing action verbatim until the context
// fills with identical errors. After N identical *failing* calls, append one line
// telling Gemma to change approach. Deterministic, ~no cost, rescues the sessions
// that would otherwise burn an hour of laptop GPU going in circles.

const TRIGGER_AT = 3; // nudge on the 3rd identical failing call

function keyOf(toolName: string, input: unknown): string {
  const h = createHash("sha1").update(JSON.stringify(input ?? null)).digest("hex").slice(0, 12);
  return `${toolName}:${h}`;
}

// Exported so the test can drive the same state machine the handler uses.
export function makeTracker() {
  let lastKey: string | null = null;
  let count = 0;
  return {
    /** Returns the repeat count for this call (1 = first time), 0 if it's a success/reset. */
    record(toolName: string, input: unknown, isError: boolean): number {
      if (!isError) { lastKey = null; count = 0; return 0; }
      const key = keyOf(toolName, input);
      if (key === lastKey) { count++; } else { lastKey = key; count = 1; }
      return count;
    },
  };
}

export function nudgeText(toolName: string, count: number): string {
  return (
    `\n\n⟳ This exact \`${toolName}\` call has now failed ${count} times in a row. ` +
    `Repeating it will not help — change your approach: fix the root cause, try a ` +
    `different command or arguments, read the relevant file/outline first, or ask the user.`
  );
}

export default function (pi: ExtensionAPI) {
  const tracker = makeTracker();

  pi.on("tool_result", async (event) => {
    const count = tracker.record(event.toolName, event.input, event.isError);
    if (count < TRIGGER_AT) return;
    const note = { type: "text" as const, text: nudgeText(event.toolName, count) };
    return { content: [...event.content, note] };
  });
}
