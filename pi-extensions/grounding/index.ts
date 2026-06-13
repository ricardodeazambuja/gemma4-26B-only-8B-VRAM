import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// grounding — make Gemma reason like an engineer, not from recollection, at THINK time.
// It brackets every reasoning pass with two DIFFERENT injections so the model can't derail
// from start to finish:
//   • beginning — a byte-stable "engineering mindset" framing in the system prefix (rule R1):
//     a remembered thing is a hypothesis, not a fact; establish each claim by deriving it,
//     simulating it, or reading a trusted reference.
//   • end — a sharper "prove it" check appended at the TAIL, the last thing read before the
//     reasoning starts, turning the principle into an act-now checklist for THIS answer.
// The point is the scientific method, not just looking things up: tools are only HOW you
// simulate (run it) or reference (read it). There is no API to seed the reasoning stream
// directly, so prefix + tail injection is the highest-salience way to reach it. No
// generate-then-review-then-regenerate: zero wasted answer/review tokens.
//
// Deliberately prevention-only: high-salience guidance the reasoning follows, not a hard
// gate. A guarantee would need detect-and-regenerate — the exact tokens this saves — so by
// design there is no backstop.

// Beginning: the standing principle, byte-stable in the system prefix (paid once, cached).
export const MINDSET = [
  "## Engineering mindset",
  "Reason like an engineer, not from recollection. A thing you \"remember\" is a hypothesis, not",
  "a fact — establish it before you rely on it, by one of three means:",
  "- derive it: work it out step by step — a mental experiment you could defend;",
  "- simulate it: run it and read the real result (a script, a test, a calculation via bash);",
  "- reference it: read the actual source — the file (read / grep / get_symbols), docs (web_search) —",
  "  not your memory of it.",
  "A claim resting only on memory is unproven: establish it, or say so. Never present recollection as fact.",
  "",
  "## Work economically",
  "Spend tokens only where they buy correctness — above all in your reasoning: think in the densest",
  "form an LLM can use (notes, not prose). Keep it simple; be creative, not over-engineered.",
].join("\n");

// End: the act-now check, re-injected at the tail each turn (different from the prefix).
export const CHECK = [
  "## Before you answer — prove it",
  "For each claim you are about to make: have you derived it, simulated it, or read it from a",
  "trusted source THIS turn? If it rests on memory, do that now — run the check, read the file,",
  "work it through — or label it \"unverified\". Recollection is not evidence.",
].join("\n");

// Thinking levels at which essentially no reasoning happens (trivial turns: greetings,
// "thanks", "continue"). Nothing to steer there → skip the tail check's prefill tax. The
// prefix stays unconditional so it remains byte-stable for the KV cache.
const SKIP_LEVELS = new Set(["off", "minimal"]);

export default function (pi: ExtensionAPI) {
  // Beginning (rule R1): append the mindset to the byte-stable system prefix. Always on,
  // so it stays cache-stable; chains with operating-manual / semantic-memory prefixes.
  pi.on("before_agent_start", async (event) => {
    return { systemPrompt: `${event.systemPrompt}\n\n${MINDSET}` };
  });

  // End (rule R1): append the prove-it check AFTER the whole conversation, so the prefix /
  // KV cache is untouched and only one short block re-prefills. Skip on trivial turns.
  pi.on("context", async (event) => {
    try {
      const level = pi.getThinkingLevel?.();
      if (level && SKIP_LEVELS.has(level)) return; // trivial turn → no reasoning to steer
    } catch {}
    const reminder = { role: "user" as const, content: [{ type: "text" as const, text: CHECK }] };
    return { messages: [...event.messages, reminder] };
  });
}
