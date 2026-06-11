import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// operating-manual — help Gemma use the empowering tools and avoid its own
// caveats. Two layers: (1) a terse if-then manual in the byte-stable system prefix
// (costs prefill once per session, then cached), and (2) just-in-time one-line
// nudges appended at the tail exactly when a tool result shows a known foot-gun.
// Rules are triggers→actions, never "you are weak at X" — small models execute
// if-then far better than they introspect. PLAN.md item 6.

// ≤600 bytes. Each line is a trigger and an action.
export const MANUAL = [
  "## Operating rules",
  "- Before reading a whole code file, call get_symbols first; read specific ranges only.",
  "- To locate a function/class, use find_symbol instead of grepping broadly.",
  "- Starting a multi-step task: call plan_set first, then plan_check as you finish steps.",
  "- Learned a durable project fact? Save it with remember.",
  "- Never do arithmetic, date, or unit math yourself — run a quick script via bash.",
  "- If the same command fails twice the same way, stop and change approach.",
  "- After editing a file, trust the auto-check result; fix reported errors before moving on.",
].join("\n");

export function buildManual(): string {
  return MANUAL;
}

// --- JIT nudges: keyed by tool, fired only when the result hits a foot-gun. ---
const GREP_MATCH_LIMIT = 80;

export function nudgeForResult(toolName: string, text: string): string | null {
  const lineCount = text ? text.split("\n").length : 0;
  if ((toolName === "grep") && lineCount > GREP_MATCH_LIMIT) {
    return "↳ That search returned a lot of matches. Narrow the pattern or add a path/glob filter so the result fits and stays useful.";
  }
  if ((toolName === "find" || toolName === "ls") && lineCount > GREP_MATCH_LIMIT) {
    return "↳ That listing is large. Scope it to a subdirectory or a name pattern.";
  }
  return null;
}

function resultText(content: { type: string; text?: string }[]): string {
  return content.filter((c) => c.type === "text").map((c) => c.text || "").join("\n");
}

export default function (pi: ExtensionAPI) {
  // Layer 1: the manual in the stable system prefix. Chains with other
  // before_agent_start handlers (e.g. semantic-memory).
  pi.on("before_agent_start", async (event) => {
    return { systemPrompt: `${event.systemPrompt}\n\n${MANUAL}` };
  });

  // Layer 2: JIT nudge at the tail when a result shows a known foot-gun.
  pi.on("tool_result", async (event) => {
    if (event.isError) return;
    const nudge = nudgeForResult(event.toolName, resultText(event.content as any));
    if (!nudge) return;
    return { content: [...event.content, { type: "text" as const, text: `\n${nudge}` }] };
  });
}
