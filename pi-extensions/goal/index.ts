import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";

// goal — the macro-loop manager. `plan` owns the steps (the HOW); `goal` holds the durable
// objective + a machine-checkable done condition (the WHAT/DONE) and decides when the whole
// job is finished. No checklist of its own — that would duplicate `plan`; instead goal_done
// reads plan's state and verifies the steps are complete. In autonomous mode (a `done_when`
// command is set) it drives the loop: after each agent run it runs done_when, and while that
// command still fails it re-engages Gemma for another cycle — until done_when passes (STOP,
// done) or a cycle budget is exhausted (STOP, blocked). The verified stop is the energy
// lever: the loop dies the moment the objective is provably met. PLAN.md item 9.
//
// Enforcement = pull + bounded push (R4):
//   pull  — goal_done() runs done_when AND checks plan's steps are complete; teaching error
//           if either is unmet.
//   push  — agent_end auto-continues ONLY when done_when is set (a trustworthy machine
//           signal). Without done_when a goal is advisory (pull-only): plan tracks the steps,
//           goal verifies them at finish.

const MAX_OBJECTIVE_LEN = 200;
const DEFAULT_MAX_CYCLES = 20;
const MAX_MAX_CYCLES = 500;
const DONE_WHEN_TIMEOUT_MS = Number(process.env.PI_GOAL_TIMEOUT_MS) || 120000;
const CLIP_LINES = 50;
const CLIP_CHARS = 2000;

export interface GoalState {
  objective: string;
  doneWhen: string | null;
  maxCycles: number;
  cycle: number;
  status: "active" | "done" | "blocked";
  blockedReason?: string;
}
export interface LastCheck { cmd: string; code: number; output: string; }

export function freshGoal(): GoalState {
  return { objective: "", doneWhen: null, maxCycles: DEFAULT_MAX_CYCLES, cycle: 0, status: "active" };
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

// Read plan's persisted state (it writes plan-<id>.json into the same session dir) and return
// the texts of any UNFINISHED steps. Empty = no plan, or every step done. This is the clean
// seam between the two extensions: goal verifies plan's checklist without owning one.
export function readPlanRemaining(sessionDir: string | null, sessionId: string): string[] {
  if (!sessionDir) return [];
  try {
    const f = join(sessionDir, `plan-${sessionId}.json`);
    if (!existsSync(f)) return [];
    const st = JSON.parse(readFileSync(f, "utf8"));
    const steps = Array.isArray(st?.steps) ? st.steps : [];
    return steps.filter((s: any) => s && !s.done).map((s: any) => String(s.text));
  } catch { return []; }
}

// R3 output cap. Keeps the TAIL of the text (test failures / latest output live at the end).
export function clip(text: string, maxLines = CLIP_LINES, maxChars = CLIP_CHARS): string {
  if (!text) return "";
  let truncated = false;
  let lines = text.split("\n");
  if (lines.length > maxLines) { lines = lines.slice(-maxLines); truncated = true; }
  let out = lines.join("\n");
  if (out.length > maxChars) { out = out.slice(-maxChars); truncated = true; }
  return truncated ? `…(clipped)\n${out}` : out;
}

// Tail status block (R1: dynamic, so it lives at the tail, never the prefix). It deliberately
// does NOT render a checklist — plan injects the steps; goal shows the objective and loop state.
export function renderGoal(s: GoalState, last?: LastCheck | null): string {
  const lines = [`## Goal status`, `Objective: ${s.objective}`];
  if (isAutonomous(s)) {
    lines.push(`Cycle: ${s.cycle}/${s.maxCycles}`);
    if (last) lines.push(`Last check: \`${last.cmd}\` exited ${last.code}` + (last.code === 0 ? "" : `:\n${clip(last.output)}`));
  }
  if (s.status === "active") {
    lines.push(`Next: when achieved, call goal_done — it verifies done_when and that your plan steps are complete.`);
  } else {
    lines.push(`Status: ${s.status}${s.blockedReason ? ` — ${s.blockedReason}` : ""}`);
  }
  return lines.join("\n");
}

// Push message: the north-star restated at the tail for drift correction (R6 shape). plan
// re-injects its own checklist each turn, so this stays lean.
export function buildContinue(s: GoalState, last?: LastCheck | null): string {
  const parts = [`Goal not yet met (cycle ${s.cycle}/${s.maxCycles}).`, `Objective: ${s.objective}`];
  if (last && last.code !== 0) parts.push(`Last check: \`${last.cmd}\` exited ${last.code}:\n${clip(last.output)}`);
  parts.push(`Continue working toward the goal. When you believe it is achieved, call goal_done — it re-verifies before accepting.`);
  return parts.join("\n");
}

// Durable terminal/cycle snapshot (R6 template). One current-state file per project,
// mirroring the STATE.md autonomous-loop convention.
export function buildSnapshot(s: GoalState, last?: LastCheck | null): string {
  const checkLine = last ? `\`${last.cmd}\` exited ${last.code}` : "—";
  return [
    `# Goal status`,
    ``,
    `Objective: ${s.objective}`,
    `Status: ${s.status}${s.blockedReason ? ` — ${s.blockedReason}` : ""}`,
    `Cycles: ${s.cycle}/${s.maxCycles}`,
    `Done-when: ${s.doneWhen ?? "—"}`,
    `Last check: ${checkLine}`,
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
  let sessionDir: string | null = null;
  let sessionId = "session";
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
      sessionDir = ctx.sessionManager.getSessionDir();
      sessionId = ctx.sessionManager.getSessionId() || "session";
      goalFile = join(sessionDir, `goal-${sessionId}.json`);
      memDir = memoryDir(ctx.cwd);
      if (existsSync(goalFile)) {
        const loaded: any = JSON.parse(readFileSync(goalFile, "utf8"));
        if (loaded && typeof loaded.objective === "string") {
          state = {
            objective: loaded.objective,
            doneWhen: typeof loaded.doneWhen === "string" ? loaded.doneWhen : null,
            maxCycles: Number.isFinite(loaded.maxCycles) ? loaded.maxCycles : DEFAULT_MAX_CYCLES,
            cycle: Number.isFinite(loaded.cycle) ? loaded.cycle : 0,
            status: loaded.status === "done" || loaded.status === "blocked" ? loaded.status : "active",
            blockedReason: loaded.blockedReason,
          };
        }
      }
    } catch {}
  });

  pi.registerTool({
    name: "goal_set",
    label: "Set Goal",
    description: "Set the objective. done_when (shell cmd, exit 0=done) runs it unattended until it passes; steps go in plan_set.",
    parameters: Type.Object({
      objective: Type.String({ description: "One-line objective, e.g. 'all unit tests pass'" }),
      done_when: Type.Optional(Type.String({ description: "Shell command; exit 0 = met" })),
      max_cycles: Type.Optional(Type.Number({ description: "Max auto-continue cycles (default 20)" })),
    }),
    async execute(_id, params) {
      const objective = (params.objective ?? "").trim();
      if (!objective) return errResult(`goal_set needs an objective, e.g. goal_set(objective="all tests green", done_when="pytest -q").`);
      if (objective.length > MAX_OBJECTIVE_LEN) return errResult(`Objective too long (${objective.length} chars). Keep it under ${MAX_OBJECTIVE_LEN} — one line; break the work into steps with plan_set.`);
      const doneWhen = (params.done_when ?? "").trim() || null;
      let maxCycles = Number.isFinite(params.max_cycles) ? Math.floor(params.max_cycles as number) : DEFAULT_MAX_CYCLES;
      maxCycles = Math.max(1, Math.min(MAX_MAX_CYCLES, maxCycles));
      state = { objective, doneWhen, maxCycles, cycle: 0, status: "active" };
      lastCheck = null;
      persist();
      snapshot();
      const mode = isAutonomous(state)
        ? `Autonomous: after each turn I run \`${doneWhen}\`; while it fails I continue (up to ${maxCycles} cycles), and stop when it passes.`
        : `Advisory: no done_when, so I will not auto-continue. Break the work into steps with plan_set; call goal_done when finished (it checks your plan is complete).`;
      return okResult(`Goal set.\n${renderGoal(state)}\n\n${mode}`);
    },
  });

  pi.registerTool({
    name: "goal_status",
    label: "Show Goal",
    description: "Show the goal, its done condition, and cycle budget.",
    parameters: Type.Object({}),
    async execute() {
      if (!state.objective) return okResult(`No goal set. Use goal_set to start one.`);
      return okResult(renderGoal(state, lastCheck));
    },
  });

  pi.registerTool({
    name: "goal_done",
    label: "Finish Goal",
    description: "Claim the goal is met; verifies done_when and that plan steps are complete first.",
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
      const remaining = readPlanRemaining(sessionDir, sessionId);
      if (remaining.length) unmet.push(`plan steps not done: ${remaining.join("; ")} (use plan_check, or revise the plan)`);
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
    description: "Show, set (objective only), or clear the goal. done_when is set via goal_set; steps via plan_set.",
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

  // R1: dynamic progress (cycle, last check) is injected at the TAIL each turn.
  pi.on("context", async (event) => {
    if (!state.objective || state.status !== "active") return;
    const reminder = { role: "user" as const, content: [{ type: "text" as const, text: renderGoal(state, lastCheck) }] };
    return { messages: [...event.messages, reminder] };
  });

  // The loop driver (push). Only autonomous goals (done_when set) auto-continue. done_when is
  // authoritative for the auto-stop — it's the machine signal, more trustworthy than ticks.
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
