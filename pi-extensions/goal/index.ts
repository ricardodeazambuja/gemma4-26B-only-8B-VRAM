import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";

// goal — the macro-loop manager. `plan` tracks the steps INSIDE one cycle; `goal`
// holds a durable, machine-checkable north-star and decides when the whole job is
// finished. In autonomous mode (a `done_when` command is set) it drives the loop:
// after each agent run it runs done_when, and while that command still fails it
// re-engages Gemma for another cycle — until done_when passes (STOP, done) or a
// cycle budget is exhausted (STOP, blocked). The verified stop is the energy lever:
// the loop dies the moment the objective is provably met. PLAN.md item 9.
//
// Enforcement = pull + bounded push (R4):
//   pull  — goal_done() runs done_when + checks ticked criteria; teaching error if unmet.
//   push  — agent_end auto-continues ONLY when done_when is set (a trustworthy machine
//           signal); criteria-only goals are pull-only, because auto-continuing on the
//           model's self-reported ticks is the unreliable signal this repo distrusts.

const MAX_OBJECTIVE_LEN = 200;
const MAX_CRITERIA = 8;
const MAX_CRITERION_LEN = 80;
const DEFAULT_MAX_CYCLES = 20;
const MAX_MAX_CYCLES = 500;
const DONE_WHEN_TIMEOUT_MS = Number(process.env.PI_GOAL_TIMEOUT_MS) || 120000;
const CLIP_LINES = 50;
const CLIP_CHARS = 2000;

export interface Criterion { text: string; done: boolean; }
export interface GoalState {
  objective: string;
  criteria: Criterion[];
  doneWhen: string | null;
  maxCycles: number;
  cycle: number;
  status: "active" | "done" | "blocked";
  blockedReason?: string;
}
export interface LastCheck { cmd: string; code: number; output: string; }

export function freshGoal(): GoalState {
  return { objective: "", criteria: [], doneWhen: null, maxCycles: DEFAULT_MAX_CYCLES, cycle: 0, status: "active" };
}

// True when the goal carries a machine-checkable stop condition → push is licensed.
export function isAutonomous(s: GoalState): boolean {
  return !!s.doneWhen;
}

// Stable per-project memory dir (shared convention with plan/semantic-memory/stats).
export function memoryDir(cwd: string): string {
  const slug = cwd.replace(/^\/+/, "").replace(/[^a-zA-Z0-9]+/g, "-").replace(/-+$/g, "") || "root";
  return join(homedir(), ".pi", "memory", slug);
}

// R3 output cap. Keeps the TAIL of the text (test failures / latest output live at
// the end) and flags truncation.
export function clip(text: string, maxLines = CLIP_LINES, maxChars = CLIP_CHARS): string {
  if (!text) return "";
  let truncated = false;
  let lines = text.split("\n");
  if (lines.length > maxLines) { lines = lines.slice(-maxLines); truncated = true; }
  let out = lines.join("\n");
  if (out.length > maxChars) { out = out.slice(-maxChars); truncated = true; }
  return truncated ? `…(clipped)\n${out}` : out;
}

export function unmetCriteria(s: GoalState): string[] {
  return s.criteria.filter((c) => !c.done).map((c) => c.text);
}

function criteriaBlock(s: GoalState): string {
  if (!s.criteria.length) return "";
  return "\n" + s.criteria.map((c, i) => `  ${c.done ? "[x]" : "[ ]"} ${i + 1}. ${c.text}`).join("\n");
}

// Tail status block (R1: dynamic, so it lives at the tail, never the prefix).
export function renderGoal(s: GoalState, last?: LastCheck | null): string {
  const lines = [`## Goal status`, `Objective: ${s.objective}`];
  if (s.criteria.length) lines.push(`Criteria:${criteriaBlock(s)}`);
  if (isAutonomous(s)) {
    lines.push(`Cycle: ${s.cycle}/${s.maxCycles}`);
    if (last) lines.push(`Last check: \`${last.cmd}\` exited ${last.code}` + (last.code === 0 ? "" : `:\n${clip(last.output)}`));
  }
  if (s.status === "active") {
    const unmet = unmetCriteria(s);
    lines.push(unmet.length ? `Remaining: ${unmet.join("; ")}` : `Next: call goal_done to verify and finish.`);
  } else {
    lines.push(`Status: ${s.status}${s.blockedReason ? ` — ${s.blockedReason}` : ""}`);
  }
  return lines.join("\n");
}

// Push message: the north-star restated at the tail for drift correction (R6 shape).
export function buildContinue(s: GoalState, last?: LastCheck | null): string {
  const parts = [`Goal not yet met (cycle ${s.cycle}/${s.maxCycles}).`, `Objective: ${s.objective}`];
  if (last && last.code !== 0) parts.push(`Last check: \`${last.cmd}\` exited ${last.code}:\n${clip(last.output)}`);
  const unmet = unmetCriteria(s);
  if (unmet.length) parts.push(`Unchecked criteria: ${unmet.join("; ")}`);
  parts.push(`Continue working toward the goal. When you believe it is achieved, call goal_done — it re-verifies before accepting.`);
  return parts.join("\n");
}

// Durable terminal/cycle snapshot (R6 template). One current-state file per project,
// mirroring the STATE.md autonomous-loop convention.
export function buildSnapshot(s: GoalState, last?: LastCheck | null): string {
  const unmet = unmetCriteria(s);
  const checkLine = last ? `\`${last.cmd}\` exited ${last.code}` : "—";
  return [
    `# Goal status`,
    ``,
    `Objective: ${s.objective}`,
    `Status: ${s.status}${s.blockedReason ? ` — ${s.blockedReason}` : ""}`,
    `Cycles: ${s.cycle}/${s.maxCycles}`,
    `Done-when: ${s.doneWhen ?? "—"}`,
    `Last check: ${checkLine}`,
    `Unmet criteria: ${unmet.length ? unmet.join("; ") : "—"}`,
    `Criteria:${s.criteria.length ? criteriaBlock(s) : " —"}`,
  ].join("\n");
}

function errResult(text: string) {
  return { content: [{ type: "text" as const, text }], isError: true };
}
function okResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

export default function (pi: ExtensionAPI) {
  let state = freshGoal();
  let goalFile: string | null = null;
  let memDir: string | null = null;
  let cwd = process.cwd();
  let lastCheck: LastCheck | null = null;
  let inAgentEnd = false; // re-entrancy guard: process each agent_end at most once

  const persist = () => {
    if (goalFile) { try { writeFileSync(goalFile, JSON.stringify(state)); } catch {} }
  };

  // Write the durable current-state file (+ full last-check log) into the project memory dir.
  const snapshot = () => {
    if (!memDir) return;
    try {
      mkdirSync(memDir, { recursive: true });
      writeFileSync(join(memDir, "goal-status.md"), buildSnapshot(state, lastCheck));
      if (lastCheck) writeFileSync(join(memDir, "goal-last-check.log"), lastCheck.output ?? "");
    } catch {}
  };

  // Run done_when through a shell; returns the exit code, or null if it cannot run.
  const runDoneWhen = async (): Promise<number | null> => {
    if (!state.doneWhen) return null;
    if (typeof pi.exec !== "function") return null;
    try {
      const res = await pi.exec("bash", ["-c", state.doneWhen], { cwd, timeout: DONE_WHEN_TIMEOUT_MS });
      const output = [res.stdout, res.stderr].filter(Boolean).join("\n");
      lastCheck = { cmd: state.doneWhen, code: res.code, output };
      return res.code;
    } catch (e) {
      lastCheck = { cmd: state.doneWhen, code: -1, output: `done_when could not run: ${(e as Error)?.message ?? e}` };
      return -1;
    }
  };

  pi.on("session_start", async (_event, ctx) => {
    try {
      cwd = ctx.cwd;
      const dir = ctx.sessionManager.getSessionDir();
      const sessionId = ctx.sessionManager.getSessionId() || "session";
      goalFile = join(dir, `goal-${sessionId}.json`);
      memDir = memoryDir(ctx.cwd);
      if (existsSync(goalFile)) {
        const loaded = JSON.parse(readFileSync(goalFile, "utf8")) as GoalState;
        if (loaded && typeof loaded.objective === "string") {
          state = { ...freshGoal(), ...loaded, criteria: Array.isArray(loaded.criteria) ? loaded.criteria : [] };
        }
      }
    } catch {}
  });

  pi.registerTool({
    name: "goal_set",
    label: "Set Goal",
    description: "Set the durable objective for this work. Add done_when (a shell command, exit 0 = done) to run unattended until it passes.",
    parameters: Type.Object({
      objective: Type.String({ description: "One-line north-star, e.g. 'all unit tests pass'" }),
      criteria: Type.Optional(Type.Array(Type.String(), { description: "Optional acceptance criteria, short phrases" })),
      done_when: Type.Optional(Type.String({ description: "Shell command that exits 0 when the objective is met" })),
      max_cycles: Type.Optional(Type.Number({ description: "Max auto-continue cycles (default 20)" })),
    }),
    async execute(_id, params) {
      const objective = (params.objective ?? "").trim();
      if (!objective) return errResult(`goal_set needs an objective, e.g. goal_set(objective="all tests green", done_when="pytest -q").`);
      if (objective.length > MAX_OBJECTIVE_LEN) return errResult(`Objective too long (${objective.length} chars). Keep it under ${MAX_OBJECTIVE_LEN} — one line; put detail in criteria.`);
      const criteria = params.criteria ?? [];
      if (criteria.length > MAX_CRITERIA) return errResult(`Too many criteria (${criteria.length}). Keep it to ${MAX_CRITERIA} or fewer.`);
      const tooLong = criteria.find((c) => c.length > MAX_CRITERION_LEN);
      if (tooLong) return errResult(`Criterion too long (${tooLong.length} chars): "${tooLong.slice(0, 30)}…". Keep each under ${MAX_CRITERION_LEN} chars.`);
      const doneWhen = (params.done_when ?? "").trim() || null;
      let maxCycles = Number.isFinite(params.max_cycles) ? Math.floor(params.max_cycles as number) : DEFAULT_MAX_CYCLES;
      maxCycles = Math.max(1, Math.min(MAX_MAX_CYCLES, maxCycles));
      state = { objective, criteria: criteria.map((text) => ({ text, done: false })), doneWhen, maxCycles, cycle: 0, status: "active" };
      lastCheck = null;
      persist();
      snapshot();
      const mode = isAutonomous(state)
        ? `Autonomous: after each turn I run \`${doneWhen}\`; while it fails I continue (up to ${maxCycles} cycles), and stop when it passes.`
        : `Advisory: no done_when, so I will not auto-continue. Call goal_done when finished — it checks the criteria.`;
      return okResult(`Goal set.\n${renderGoal(state)}\n\n${mode}`);
    },
  });

  pi.registerTool({
    name: "goal_check",
    label: "Check Goal Criterion",
    description: "Mark an acceptance criterion done by its number (1-based).",
    parameters: Type.Object({ n: Type.Number({ description: "1-based criterion number" }) }),
    async execute(_id, params) {
      if (!state.objective) return errResult(`No goal set. Call goal_set first.`);
      if (!state.criteria.length) return errResult(`This goal has no criteria to check. It completes when done_when passes (call goal_done).`);
      const n = params.n;
      if (!Number.isInteger(n) || n < 1 || n > state.criteria.length) {
        return errResult(`Criterion ${n} is out of range. This goal has ${state.criteria.length} criteria (1–${state.criteria.length}).`);
      }
      state.criteria[n - 1].done = true;
      persist();
      snapshot();
      return okResult(`Criterion ${n} done.\n${renderGoal(state, lastCheck)}`);
    },
  });

  pi.registerTool({
    name: "goal_status",
    label: "Show Goal",
    description: "Show the current goal, its progress, and the cycle budget.",
    parameters: Type.Object({}),
    async execute() {
      if (!state.objective) return okResult(`No goal set. Use goal_set to start one.`);
      return okResult(renderGoal(state, lastCheck));
    },
  });

  pi.registerTool({
    name: "goal_done",
    label: "Finish Goal",
    description: "Claim the goal is met. Verifies done_when and all criteria before accepting.",
    parameters: Type.Object({}),
    async execute() {
      if (!state.objective) return errResult(`No goal set — nothing to finish.`);
      if (state.status === "done") return okResult(`Goal already marked done.`);
      const unmet: string[] = [];
      if (state.doneWhen) {
        const code = await runDoneWhen();
        if (code !== 0) {
          const detail = code === null ? `(could not run \`${state.doneWhen}\`)` : `\`${state.doneWhen}\` exited ${code}:\n${clip(lastCheck?.output ?? "")}`;
          unmet.push(`done_when not satisfied: ${detail}`);
        }
      }
      const unticked = unmetCriteria(state);
      if (unticked.length) unmet.push(`unchecked criteria: ${unticked.join("; ")} (use goal_check)`);
      if (unmet.length) {
        return errResult(`Not done yet — fix these, then call goal_done again:\n- ${unmet.join("\n- ")}`);
      }
      state.status = "done";
      persist();
      snapshot();
      return okResult(`Goal complete: ${state.objective}`);
    },
  });

  // Human entry point: `/goal` (show), `/goal clear`, `/goal <objective>` (advisory set).
  pi.registerCommand("goal", {
    description: "Show, set (objective only), or clear the goal. done_when is set via goal_set.",
    handler: async (args, ctx) => {
      const notify = (msg: string) => { try { (ctx as any)?.ui?.notify?.(msg); } catch {} };
      const a = (args ?? "").trim();
      if (a === "clear") {
        state = freshGoal(); lastCheck = null; persist(); snapshot();
        notify("Goal cleared.");
        return;
      }
      if (!a) { notify(state.objective ? renderGoal(state, lastCheck) : "No goal set."); return; }
      state = { ...freshGoal(), objective: a.slice(0, MAX_OBJECTIVE_LEN) };
      persist(); snapshot();
      notify(`Goal set (advisory): ${state.objective}`);
    },
  });

  // R1: the immutable objective goes into the byte-stable system prefix (one line +
  // a fixed instruction), so it is the always-present anchor and is paid once.
  pi.on("before_agent_start", async (event) => {
    if (!state.objective || state.status !== "active") return;
    const block = `## Goal (north-star)\n${state.objective}\nWhen you believe this is achieved, call goal_done — it verifies before accepting.`;
    return { systemPrompt: `${event.systemPrompt}\n\n${block}` };
  });

  // R1: dynamic progress (ticks, cycle, last check) is injected at the TAIL each turn.
  pi.on("context", async (event) => {
    if (!state.objective || state.status !== "active") return;
    const reminder = { role: "user" as const, content: [{ type: "text" as const, text: renderGoal(state, lastCheck) }] };
    return { messages: [...event.messages, reminder] };
  });

  // The loop driver (push). Only autonomous goals (done_when set) auto-continue.
  pi.on("agent_end", async () => {
    if (state.status !== "active" || !isAutonomous(state) || inAgentEnd) return;
    inAgentEnd = true;
    try {
      const code = await runDoneWhen();
      if (code === 0) {
        // Machine-checkable termination: done even if Gemma forgot to call goal_done.
        state.status = "done"; persist(); snapshot();
        return;
      }
      if (state.cycle < state.maxCycles) {
        state.cycle++; persist(); snapshot();
        pi.sendUserMessage(buildContinue(state, lastCheck), { deliverAs: "followUp" });
      } else {
        // BLOCKED is durable, not silent — stop cleanly, never a silent runaway.
        state.status = "blocked";
        state.blockedReason = `cycle budget exhausted (${state.maxCycles}) without done_when passing`;
        persist(); snapshot();
      }
    } finally {
      inAgentEnd = false;
    }
  });
}
