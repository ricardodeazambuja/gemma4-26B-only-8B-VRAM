import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// pipe — chain slash-commands with nested-command expansion. pi has no real command
// piping (handlers return void, there's no executeCommand, and the input event is
// read-only), so /pipe parses a nested expression like
//   /pipe /goal implement the results from /plan a python script that says hello world
// into ordered steps (innermost runs first), expands each into a tool-using directive,
// and hands the whole thing to the agent in one go via sendUserMessage. The model then
// calls the underlying tools (plan_set, goal_set, …) in sequence.

export interface Stage { cmd: string; text: string; }

// Known commands → the tool the agent should call. Unknown commands get a generic
// "run /cmd" line so the pipe still composes (the model figures out the intent).
const ACTIONS: Record<string, (text: string) => string> = {
  plan: (t) => (t ? `create a plan (call plan_set) for: ${t}` : `create a plan (call plan_set)`),
  goal: (t) => (t ? `set the goal (call goal_set): ${t}` : `set the goal (call goal_set)`),
};

// Commands /pipe knows how to chain. Only these split the expression into stages, so a
// stray `/tmp/foo` or `and/or` in an argument is never mistaken for a command. Extend
// ACTIONS to teach /pipe a new command.
const KNOWN = new Set(Object.keys(ACTIONS));

export function describeStage(cmd: string, text: string): string {
  const fn = ACTIONS[cmd.toLowerCase()];
  if (fn) return fn(text);
  return text ? `run /${cmd}: ${text}` : `run /${cmd}`;
}

// Parse the expression into stages in TEXTUAL order (outer … innermost). A stage boundary
// is a `/<known-command>` token at the start or after whitespace; unknown `/tokens`
// (paths, "and/or") stay part of the surrounding argument. Each stage's text runs up to
// the next known command.
export function parsePipe(expr: string): Stage[] | null {
  const s = (expr ?? "").trim();
  if (!s) return null;
  const marks: { cmd: string; start: number; argStart: number }[] = [];
  for (const m of s.matchAll(/(?:^|\s)\/([a-zA-Z][a-zA-Z0-9_-]*)/g)) {
    if (!KNOWN.has(m[1].toLowerCase())) continue;   // ignore /paths and unknown commands
    marks.push({ cmd: m[1], start: m.index ?? 0, argStart: (m.index ?? 0) + m[0].length });
  }
  if (!marks.length) return null;
  return marks.map((mk, i) => ({
    cmd: mk.cmd,
    text: s.slice(mk.argStart, i + 1 < marks.length ? marks[i + 1].start : s.length).trim(),
  }));
}

// Build the ordered directive. Execution order is innermost-first (the outer command
// references the inner one's result), so we reverse the textual order and number from 1.
export function buildDirective(stages: Stage[]): string {
  const steps = [...stages].reverse();
  const lines = steps.map((st, i) => {
    const ref = i > 0 ? ` — use the result of step ${i}` : "";
    return `${i + 1}. ${describeStage(st.cmd, st.text)}${ref}`;
  });
  return [
    "Pipeline — complete these steps in order; each step builds on the previous one:",
    ...lines,
    "Then carry out the work so the final goal is achieved.",
  ].join("\n");
}

const USAGE = `/pipe chains slash-commands, e.g.\n  /pipe /goal implement the results from /plan write a hello-world python script\nThe expression must contain at least one /command.`;

export default function (pi: ExtensionAPI) {
  pi.registerCommand("pipe", {
    description: "Chain slash-commands: /pipe /goal implement /plan write a hello-world script — expands nested /cmd into an ordered directive the agent runs.",
    handler: async (args, ctx) => {
      const notify = (msg: string, kind: "info" | "error" = "info") => { try { (ctx as any)?.ui?.notify?.(msg, kind); } catch {} };
      const stages = parsePipe(args);
      if (!stages) { notify(USAGE, "error"); return; }
      const directive = buildDirective(stages);
      notify(`pipe → ${stages.map((s) => "/" + s.cmd).join(" ▸ ")}`);
      // The only lever: drive the agent with the expanded directive. sendUserMessage
      // always triggers a turn, so the model executes the steps in order.
      pi.sendUserMessage(directive, { deliverAs: "followUp" });
    },
  });
}
