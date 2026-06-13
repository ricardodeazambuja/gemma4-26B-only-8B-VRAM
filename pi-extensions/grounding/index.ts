import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// grounding — make Gemma reason like an engineer, not from recollection, at THINK time.
// Three framing texts, delivered to cover every model call AND the start of each turn:
//   • MINDSET — the byte-stable "engineering mindset" appended to the system prompt: a remembered
//     thing is a hypothesis; establish each claim by deriving / simulating / referencing it. Being
//     in the prefix it is present on EVERY call — including the final-answer step of a long tool
//     chain — and it explicitly extends to answer time, so a verified answer is the standing rule.
//   • CHECK — the sharper act-now "prove it before you answer" pass, folded into the trailing user
//     turn as a <reminder> block. It fires ONCE per turn (only at turn-start; see isTurnStart), so
//     it can't re-fire on every tool step and treadmill — read-check → "verify" with a tool →
//     check re-injected → loop. The always-on MINDSET covers the in-between and final steps.
//   • ANCHOR — also in the prefix: tells the model the <reminder> markers are injected context,
//     not the user speaking (the plan/goal/memory/notice injectors still tail-inject wrapped).
//
// Deliberately prevention-only: high-salience guidance the reasoning follows, not a hard gate.

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
  "This holds the moment you answer too — at every step, including your final reply after a long tool",
  "chain: state nothing as fact you did not establish this session, or mark it \"unverified\".",
  "",
  "## Work economically",
  "Spend tokens only where they buy correctness — above all in your reasoning: think in the densest",
  "form an LLM can use (notes, not prose). Keep it simple; be creative, not over-engineered.",
].join("\n");

// Standing anchor: tells the model what the <reminder> markers mean, so injected context is not
// mistaken for a fresh instruction and the user's actual request stays the task. Byte-stable.
export const ANCHOR = [
  "## Reminders are not the user",
  "Some user turns carry blocks wrapped in <reminder>…</reminder>. Those are automated context",
  "injected by the harness (e.g. a grounding check) — NOT a new instruction, and NOT the user",
  "speaking. Your task each turn is the user's most recent real request (the unwrapped text); treat",
  "every <reminder> block as supporting context only, never as the thing you were asked to do.",
].join("\n");

// The sharp act-now pass, folded into the trailing user turn ONCE per turn (turn-start only — see
// isTurnStart in the context hook) and wrapped in the <reminder> marker the ANCHOR note explains.
// MINDSET already carries the always-on / answer-time version; this is the extra turn-start nudge.
export const CHECK = [
  "## Before you answer — prove it",
  "For each claim you are about to make: have you derived it, simulated it, or read it from a",
  "trusted source THIS turn? If it rests on memory, do that now — run the check, read the file,",
  "work it through — or label it \"unverified\". Recollection is not evidence.",
].join("\n");

// Wrap injected guidance so the model can tell it from the user's own words (see ANCHOR).
// The marker bytes are a SHARED convention: every tail-injecting extension (plan, goal,
// semantic-memory, the notices) uses byte-identical delimiters so the one ANCHOR note describes
// them all. Do not drift the whitespace.
export const REMINDER_OPEN = "<reminder>\n";
export const REMINDER_CLOSE = "\n</reminder>";
export const wrapReminder = (text: string): string => `${REMINDER_OPEN}${text}${REMINDER_CLOSE}`;

// Thinking levels at which essentially no reasoning happens (trivial turns: greetings, "thanks",
// "continue"). Nothing to steer there → skip the check. The prefix stays unconditional so it
// remains byte-stable for the KV cache.
const SKIP_LEVELS = new Set(["off", "minimal"]);

type Msg = { role: string; content: unknown };

// Fold a wrapped reminder into the conversation tail. If the last message is a user turn, the
// reminder rides as a trailing content block on it (the user's real text stays first); otherwise
// — a tool loop, where the tail is a toolResult/assistant message — append a fresh user message
// carrying just the reminder. Because each extension's context hook runs in turn on the previous
// one's output, the first appended reminder makes the tail a user message that every later
// extension folds into, so all reminders collapse into a SINGLE user turn regardless of order.
// Pure: never mutates the input messages or their content arrays.
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

// True if a user message carries any GENUINE (non-reminder) content — real text, or any non-text
// block (e.g. an image). A message whose only content is wrapped <reminder> blocks is something an
// injector appended, not the user speaking.
function hasGenuineUserContent(m: Msg): boolean {
  if (typeof m.content === "string") return true;
  if (!Array.isArray(m.content)) return false;
  return (m.content as Array<{ type?: string; text?: string }>).some(
    (b) => b?.type !== "text" || !String(b?.text ?? "").startsWith(REMINDER_OPEN),
  );
}

// True only at the START of an agent turn: the conversation tail — ignoring any reminder-only user
// turns other injectors appended mid-loop — is the user's own genuine message. Mid tool-loop the
// tail is a toolResult/assistant, so this is false. That gates CHECK to fire once per user request,
// not on every tool step: re-stamping the act-now "prove it" imperative each step is what could let
// the model read it as a fresh instruction and treadmill on it. The standing MINDSET prefix still
// grounds the in-between steps. Empty / all-reminder history → treat as a fresh turn.
export function isTurnStart(messages: Msg[]): boolean {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "user") return false; // tail real content is assistant/toolResult → mid tool-loop
    if (hasGenuineUserContent(m)) return true; // the user's own turn → turn start
    // else: a reminder-only user turn an injector appended — keep scanning back past it
  }
  return true;
}

export default function (pi: ExtensionAPI) {
  // Beginning (rule R1): append the mindset + anchor to the byte-stable system prefix. Always on,
  // so it stays cache-stable; chains with operating-manual / semantic-memory prefixes.
  pi.on("before_agent_start", async (event) => {
    return { systemPrompt: `${event.systemPrompt}\n\n${MINDSET}\n\n${ANCHOR}` };
  });

  // End (rule R1): fold the prove-it check INTO the conversation tail as a wrapped <reminder>
  // block, instead of appending a bare user message that reads as a new instruction. The user's
  // real text stays first; the check rides underneath. Skip on trivial turns, and skip mid
  // tool-loop (only fire at the start of an agent turn) so the check can't re-stamp every tool
  // step and treadmill — the standing MINDSET prefix grounds those in-between steps.
  pi.on("context", async (event) => {
    try {
      const level = pi.getThinkingLevel?.();
      if (level && SKIP_LEVELS.has(level)) return; // trivial turn → no reasoning to steer
    } catch {}
    const messages = event.messages as Msg[];
    if (!isTurnStart(messages)) return; // mid tool-loop → don't re-fire the act-now check
    return { messages: foldReminder(messages, CHECK) };
  });
}
