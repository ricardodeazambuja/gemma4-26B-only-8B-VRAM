import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// thinking-router — spend decode tokens (the expensive part on a laptop) in
// proportion to how hard the turn is. Easy turns (short, no code, simple ask) get
// a low thinking budget; anything that looks like real work gets the full budget.
// One resident model, so this routes the *thinking level*, not the model. This is
// the pi-code half of the project's "engine-level" energy levers.

export type Level = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

const CODE_HINT = /```|\bfunction\b|\bclass\b|\bdef\b|\bimport\b|=>|;\s*$|\{[\s\S]*\}|\b(fix|debug|refactor|implement|build|optimi[sz]e)\b/i;
const HARD_HINT = /\b(why|how|explain|design|architect|plan|analy[sz]e|compare|trade-?off|review|prove|derive)\b/i;
const TRIVIAL_HINT = /^\s*(hi|hey|hello|thanks|thank you|ok|okay|yes|no|yep|nope|cool|nice|got it|continue|go|next|stop)\b/i;

/** Decide a thinking level from the user's input text. Pure + exported for tests. */
export function routeLevel(text: string): Level {
  const t = (text || "").trim();
  if (!t) return "low";
  const words = t.split(/\s+/).length;
  if (TRIVIAL_HINT.test(t) && words <= 6) return "off";
  if (CODE_HINT.test(t) || HARD_HINT.test(t)) return "medium";
  if (words <= 12) return "low";       // short factual ask
  return "medium";                      // longer prose → probably non-trivial
}

export default function (pi: ExtensionAPI) {
  // Only auto-route when the user hasn't pinned a level via /thinking. We detect a
  // manual pin by watching thinking_level_select with source "set".
  let userPinned = false;

  pi.on("thinking_level_select", async (event: any) => {
    if (event?.source === "set") userPinned = true;
  });

  pi.on("input", async (event) => {
    if (userPinned) return; // respect an explicit /thinking choice
    if (event.source && event.source !== "user") return; // only route human input
    try {
      const level = routeLevel(event.text);
      // getThinkingLevel/setThinkingLevel live on the ExtensionAPI, not ctx.
      if (pi.getThinkingLevel?.() !== level) pi.setThinkingLevel?.(level);
    } catch {}
    // return nothing → input proceeds unchanged
  });
}
