import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { isEditToolResult, isWriteToolResult } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";

// plan — an external task checklist so Gemma stops re-deciding what it's doing
// every turn. State is re-injected at the TAIL of the context each turn (cache-safe)
// and snapshotted before compaction so the thread of work survives.

const MAX_STEPS = 10;
const MAX_STEP_LEN = 80;

export interface Step { text: string; done: boolean; }
interface PlanState { steps: Step[]; touched: string[]; }

// Stable per-project memory dir (shared convention with semantic-memory/stats).
export function memoryDir(cwd: string): string {
  const slug = cwd.replace(/^\/+/, "").replace(/[^a-zA-Z0-9]+/g, "-").replace(/-+$/g, "") || "root";
  return join(homedir(), ".pi", "memory", slug);
}

export function renderChecklist(steps: Step[]): string {
  if (!steps.length) return "";
  const lines = steps.map((s, i) => `  ${s.done ? "[x]" : "[ ]"} ${i + 1}. ${s.text}`);
  const next = steps.find((s) => !s.done);
  const tail = next ? `\nCurrent step: ${next.text}` : `\nAll steps complete — verify your work; if a goal is set, call goal_done to finish.`;
  return `## Active plan\n${lines.join("\n")}${tail}`;
}

export function buildSnapshot(steps: Step[], touched: string[]): string {
  const done = steps.filter((s) => s.done).map((s) => s.text);
  const next = steps.filter((s) => !s.done).map((s) => s.text);
  const task = steps[0]?.text ?? "(no plan set)";
  return [
    `# Session snapshot`,
    ``,
    `Task: ${task}`,
    `Done: ${done.length ? done.join("; ") : "—"}`,
    `Next: ${next.length ? next.join("; ") : "—"}`,
    `Files touched: ${touched.length ? touched.join(", ") : "—"}`,
  ].join("\n");
}

// ---- reminder marker (shared convention; see grounding's ANCHOR note) -----------------------
// Injected guidance is wrapped so the model can tell it from the user's own words, and folded
// into the trailing user turn (or appended as a fresh user message when the tail is a
// toolResult/assistant) so it never surfaces as a new user instruction. The marker bytes MUST be
// byte-identical to every other extension's — the one ANCHOR note describes them all.
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
  const state: PlanState = { steps: [], touched: [] };
  let planFile: string | null = null;
  let memDir: string | null = null;
  let sessionId = "session";

  const persist = () => {
    if (!planFile) return;
    try { writeFileSync(planFile, JSON.stringify(state)); } catch {}
  };

  pi.on("session_start", async (_event, ctx) => {
    try {
      const dir = ctx.sessionManager.getSessionDir();
      sessionId = ctx.sessionManager.getSessionId() || "session";
      planFile = join(dir, `plan-${sessionId}.json`);
      memDir = memoryDir(ctx.cwd);
      if (existsSync(planFile)) {
        const loaded = JSON.parse(readFileSync(planFile, "utf8")) as PlanState;
        if (Array.isArray(loaded.steps)) { state.steps = loaded.steps; state.touched = loaded.touched || []; }
      }
    } catch {}
  });

  pi.registerTool({
    name: "plan_set",
    label: "Set Plan",
    description: "Set the plan: an ordered list of short steps. Call once when starting multi-step work.",
    parameters: Type.Object({
      steps: Type.Array(Type.String(), { description: "Ordered steps, each a short phrase" }),
    }),
    async execute(_id, params) {
      const steps = params.steps ?? [];
      if (!steps.length) return errResult(`plan_set needs at least one step, e.g. plan_set(steps=["read config", "fix bug", "run tests"]).`);
      if (steps.length > MAX_STEPS) return errResult(`Too many steps (${steps.length}). Keep it to ${MAX_STEPS} or fewer — merge fine-grained steps.`);
      const tooLong = steps.find((s) => s.length > MAX_STEP_LEN);
      if (tooLong) return errResult(`Step too long (${tooLong.length} chars): "${tooLong.slice(0, 30)}…". Keep each step under ${MAX_STEP_LEN} chars.`);
      state.steps = steps.map((text) => ({ text, done: false }));
      persist();
      return { content: [{ type: "text", text: `Plan set (${steps.length} steps):\n${renderChecklist(state.steps)}` }] };
    },
  });

  pi.registerTool({
    name: "plan_check",
    label: "Check Plan Step",
    description: "Mark a plan step done by its number (1-based).",
    parameters: Type.Object({ step: Type.Number({ description: "1-based step number" }) }),
    async execute(_id, params) {
      const n = params.step;
      if (!state.steps.length) return errResult(`No plan set yet. Call plan_set first.`);
      if (!Number.isInteger(n) || n < 1 || n > state.steps.length) {
        return errResult(`Step ${n} is out of range. The plan has ${state.steps.length} steps (1–${state.steps.length}).`);
      }
      state.steps[n - 1].done = true;
      persist();
      return { content: [{ type: "text", text: `Step ${n} done.\n${renderChecklist(state.steps)}` }] };
    },
  });

  pi.registerTool({
    name: "plan_show",
    label: "Show Plan",
    description: "Show the plan and remaining steps.",
    parameters: Type.Object({}),
    async execute() {
      if (!state.steps.length) return { content: [{ type: "text", text: "No plan set. Use plan_set to start one." }] };
      return { content: [{ type: "text", text: renderChecklist(state.steps) }] };
    },
  });

  // Track files touched (for the snapshot's "Files touched" line).
  pi.on("tool_result", async (event) => {
    if (event.isError) return;
    if (!isEditToolResult(event) && !isWriteToolResult(event)) return;
    const p = (event.input as { path?: string }).path;
    if (p && !state.touched.includes(p)) { state.touched.push(p); if (state.touched.length > 50) state.touched.shift(); persist(); }
  });

  // Re-inject the checklist each turn (cache-safe — it rides the tail, never the prefix). Folded
  // into the trailing user turn as a wrapped <reminder> so it doesn't read as a new instruction
  // from the user. Only when a plan exists and has unfinished work worth reminding about.
  pi.on("context", async (event) => {
    if (!state.steps.length) return;
    return { messages: foldReminder(event.messages as Msg[], renderChecklist(state.steps)) };
  });

  // Persist a human-readable snapshot before compaction destroys the detail, so
  // semantic-memory can ingest "what this session was doing".
  pi.on("session_before_compact", async () => {
    if (!state.steps.length || !memDir) return;
    try {
      const snapDir = join(memDir, "snapshots");
      mkdirSync(snapDir, { recursive: true });
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      writeFileSync(join(snapDir, `${sessionId}_${stamp}.md`), buildSnapshot(state.steps, state.touched));
    } catch {}
    // returning nothing → compaction proceeds normally
  });
}

function errResult(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}
