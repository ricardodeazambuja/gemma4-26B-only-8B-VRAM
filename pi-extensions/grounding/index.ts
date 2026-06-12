import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// grounding — stop hand-waving at THINK time, not after. Each turn we append a terse
// reasoning protocol at the TAIL of the context — the last thing Gemma reads before it
// thinks — so the "did this come from a tool or from memory?" check happens INSIDE the
// chain-of-thought, and the only answer ever decoded is already grounded. There is no
// API to seed the reasoning stream directly, so tail injection is the highest-salience
// way to reach it. No generate-then-review-then-regenerate: zero wasted answer/review
// tokens (the repo's energy thesis applied to itself). PLAN.md item 10 (think-time lever).
//
// Deliberately prevention-only: this is high-salience guidance the reasoning follows, not
// a hard gate. A guarantee would need to detect a bad answer and regenerate — the exact
// tokens this is built to save — so by design there is no backstop here.

export const PROTOCOL = [
  "## Grounding (apply this while reasoning, before you answer)",
  "Before asserting any fact — about this codebase, a file's contents, an API, a command's",
  "result, or the outside world — check in your reasoning: did this come from a tool, or from",
  "memory? If it is from memory, verify it now (read / grep / get_symbols / find_symbol /",
  "web_search / fetch_page / bash) before stating it. If you cannot verify it, say \"I haven't",
  "verified this\" rather than presenting a guess as fact. Ground the answer in tool output, not recall.",
].join("\n");

// Thinking levels at which essentially no reasoning happens (trivial turns: greetings,
// "thanks", "continue"). Nothing to steer, and no hand-wave risk → skip the prefill tax.
const SKIP_LEVELS = new Set(["off", "minimal"]);

export default function (pi: ExtensionAPI) {
  // Tail injection (rule R1): the protocol is appended AFTER the whole conversation, so
  // the byte-stable prefix / KV cache is untouched and only ~one short block re-prefills.
  pi.on("context", async (event) => {
    try {
      const level = pi.getThinkingLevel?.();
      if (level && SKIP_LEVELS.has(level)) return; // trivial turn → no reasoning to steer
    } catch {}
    const reminder = { role: "user" as const, content: [{ type: "text" as const, text: PROTOCOL }] };
    return { messages: [...event.messages, reminder] };
  });
}
