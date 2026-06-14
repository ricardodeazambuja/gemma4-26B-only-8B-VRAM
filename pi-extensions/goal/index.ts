import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import {
  type Band, type AnnealConfig, bandFor, temperature, isColdBand,
  annealConfigFromEnv, annealEnabled, tempAnnealEnabled,
} from "./anneal.ts";

// goal — the macro-loop manager. `plan` owns the steps (the HOW); `goal` holds the durable
// objective + a machine-checkable done condition (the WHAT/DONE) and decides when the whole
// job is finished. No checklist of its own — that would duplicate `plan`; instead goal_done
// reads plan's state and verifies the steps are complete. In autonomous mode (a `done_when`
// command is set) it drives the loop: after each agent run it runs done_when, and while that
// command still fails it re-engages Gemma for another cycle — until done_when passes (STOP,
// done) or a cycle budget is exhausted (STOP, blocked). The verified stop is the energy
// lever: the loop dies the moment the objective is provably met.
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
  check: string | null;   // self-judged completion criteria Gemma applies (null → DEFAULT_CHECK)
  maxCycles: number;
  cycle: number;
  status: "active" | "done" | "blocked" | "concluded";
  blockedReason?: string;
  // The model's own stop decision via goal_conclude (PRD FR6). status="concluded" means the model
  // deliberately landed the work with a stated outcome — distinct from "blocked" (the loop ran out
  // of road) and "done" (the verified goal_done path).
  outcome?: "partial" | "abandoned";
  summary?: string;
}
export interface LastCheck { cmd: string; code: number; output: string; }

export function freshGoal(): GoalState {
  return { objective: "", doneWhen: null, check: null, maxCycles: DEFAULT_MAX_CYCLES, cycle: 0, status: "active" };
}

// The annealing config (env-overridable; pure defaults otherwise) and whether the cooling schedule
// is on. Read once per process — the schedule itself is pure (see anneal.ts).
const ANNEAL_CFG: AnnealConfig = annealConfigFromEnv();
const ANNEAL_ON = annealEnabled();
// Channel B (sampling-temperature annealing) is OFF by default — it rides an untyped provider seam
// (PRD §6.4) and needs a live-model spike to confirm end-to-end. Opt in with PI_GOAL_TEMP_ANNEAL=1.
const TEMP_ANNEAL_ON = tempAnnealEnabled();

// The current teacher band for an active goal (decide for a non-active/empty goal is harmless — it
// is only read while active). When annealing is off, the whole loop falls back to flat behavior.
export function currentBand(s: GoalState, cfg: AnnealConfig = ANNEAL_CFG): Band {
  return bandFor(Math.max(1, s.cycle), s.maxCycles, cfg);
}

// Channel B (PRD §6.4): cool the provider request's sampling temperature as the loop progresses.
// PURE + FAIL-OPEN by construction so the goal test can assert the guard logic without a live model:
//   • returns the payload UNTOUCHED unless both channels are on AND a goal is actively looping;
//   • shape-guards the payload (only a chat-completions body with a messages[] array is touched);
//   • returns a shallow copy so a shared/cached request object is never mutated in place.
// FR9 (no silent clobber): the hot end IS the request's OWN temperature — we only cool it DOWN toward
// min(tempLo, base), never raise it. cycle 1 leaves it essentially untouched; it descends as T cools.
// (When the request sets no temperature, tempHi is the assumed base.) The mutated field reaches the
// wire: in pi's openai-completions provider onPayload runs AFTER buildParams sets temperature and its
// return is sent as-is (openai-completions.ts:146-157) — only llama.cpp honoring the field is unspiked.
export function applyTempAnneal(
  payload: unknown,
  s: GoalState,
  opts: { annealOn?: boolean; tempOn?: boolean; cfg?: AnnealConfig } = {},
): unknown {
  const annealOn = opts.annealOn ?? ANNEAL_ON;
  const tempOn = opts.tempOn ?? TEMP_ANNEAL_ON;
  const cfg = opts.cfg ?? ANNEAL_CFG;
  if (!annealOn || !tempOn) return payload; // channel off
  if (!s || !s.objective || s.status !== "active") return payload; // no active goal → leave normal turns alone
  if (!payload || typeof payload !== "object") return payload; // shape guard → fail open
  const body = payload as Record<string, unknown>;
  if (!Array.isArray(body.messages)) return payload; // not the body we expect → fail open
  const base = typeof body.temperature === "number" && Number.isFinite(body.temperature) ? body.temperature : cfg.tempHi;
  const floor = Math.min(cfg.tempLo, base); // never cool ABOVE the base, even if tempLo > base
  const t = temperature(Math.max(1, s.cycle), s.maxCycles, cfg); // 1 hot → 0 cold
  return { ...body, temperature: floor + (base - floor) * t };
}

// True when the goal carries a machine-checkable stop condition (done_when). It's the AUTHORITATIVE
// auto-stop; without it the loop is self-judged (Gemma decides via goal_done). Either way an active
// goal LOOPS — this only selects which stop signal governs.
export function isAutonomous(s: GoalState): boolean {
  return !!s.doneWhen;
}

// The completion check Gemma must apply before declaring done. A user/Gemma-supplied `check` is the
// specific criteria ("re-render the SVG and confirm it reads as a pelican on a bike"); the default
// is the MINDSET-grounded "verify, don't assume" — no external advisor, Gemma checks its own work.
export const DEFAULT_CHECK =
  "verify the objective is genuinely achieved THIS turn — derive it, run/simulate it, or read the " +
  "real result — never assume you're done from memory or hope.";
export function checkText(s: GoalState): string {
  return s.check?.trim() || DEFAULT_CHECK;
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
  if (s.status === "active") {
    lines.push(`Cycle: ${s.cycle}/${s.maxCycles}`);
    if (ANNEAL_ON) {
      const band = currentBand(s);
      lines.push(`Phase: ${band} (temp ${temperature(Math.max(1, s.cycle), s.maxCycles, ANNEAL_CFG).toFixed(2)})`);
    }
  }
  if (isAutonomous(s) && last) {
    lines.push(`Last check: \`${last.cmd}\` exited ${last.code}` + (last.code === 0 ? "" : `:\n${clip(last.output)}`));
  }
  if (s.status === "active") {
    lines.push(`Completion check: ${checkText(s)}`);
    lines.push(`When it passes, call the goal_done tool${isAutonomous(s) ? " (it also runs done_when)" : ""} — invoke the tool, don't just say you're done, and never on assumption.`);
    if (ANNEAL_ON && isColdBand(currentBand(s))) {
      lines.push(`If the objective truly can't be fully met this cycle, call goal_conclude(outcome, summary) to land it honestly.`);
    }
  } else if (s.status === "concluded") {
    lines.push(`Status: concluded — ${s.outcome ?? "partial"}${s.summary ? `: ${s.summary}` : ""}`);
  } else {
    lines.push(`Status: ${s.status}${s.blockedReason ? ` — ${s.blockedReason}` : ""}`);
  }
  return lines.join("\n");
}

// The banded coaching + verification register (PRD §6.2 / §6.6a). One block per teacher phase: the
// nudge cools explore → consolidate → commit → decide, and the VERIFICATION ask cools with it —
// from "establish broadly" to "verify what the decision hinges on, flag the rest". The honesty floor
// (verified, or explicitly marked unverified) is constant across every band; only emphasis/triage
// changes. goal_conclude is offered ONLY in the cold bands (commit/decide), matching FR6's gate.
export function bandGuidance(band: Band, s: GoalState): string {
  const check = checkText(s);
  const done = (extra = "") =>
    `When the completion check genuinely passes, call goal_done — invoke the tool, never claim done on assumption.${extra}`;
  const cold =
    " If the objective truly can't be fully met, you may call goal_conclude(outcome, summary) to land it honestly rather than thrash.";
  switch (band) {
    case "explore":
      return (
        `Phase — explore. You have budget to range widely; don't lock in yet. Consider more than one ` +
        `approach, and question what you're assuming. Verification is cheap now: establish each claim ` +
        `before you lean on it — derive it, run/simulate it, or read the real source — and be skeptical ` +
        `of memory.\nCompletion check (must pass before you finish):\n${check}\n${done()}`
      );
    case "consolidate":
      return (
        `Phase — consolidate. You've explored; now converge. Pick the most promising line, deepen it, ` +
        `and start closing open threads — don't open new directions unless the current one is failing. ` +
        `Verify the claims this direction actually depends on.\nCompletion check (must pass before you ` +
        `finish):\n${check}\n${done()}`
      );
    case "commit":
      return (
        `Phase — commit. Budget is nearly spent; commit to your best result and finish it. Spend your ` +
        `remaining verification on what the outcome hinges on — derive/run/read; for anything you can't ` +
        `establish, state it explicitly as "unverified". Don't open new threads.\nCompletion check ` +
        `(must pass before you finish):\n${check}\n${done(cold)}`
      );
    case "decide":
      return (
        `Phase — DECIDE (final cycle). You cannot iterate further; land the best outcome you have now:\n` +
        `- If the objective is met: verify it (derive/run/read) and call goal_done.\n` +
        `- If it's only partially met: finish the solid parts, state plainly what is unverified or ` +
        `incomplete, and call goal_conclude(outcome="partial", summary=...).\n` +
        `- If it genuinely cannot be done: call goal_conclude(outcome="abandoned", summary=<the specific reason>).\n` +
        `Decide now — invoke a tool; never claim more than you verified.\nCompletion check:\n${check}`
      );
  }
}

// Push message: the north-star restated at the tail each cycle for drift correction (R6 shape),
// carrying the (now annealed) completion-check guidance so Gemma re-applies it — and, when the last
// goal_done was rejected, the reason, so the loop hands back a gradient instead of a bare "not yet".
// plan re-injects its own checklist, so lean. With annealing off (PI_GOAL_ANNEAL=0) it falls back to
// the original flat push, byte-for-byte, so the schedule is a pure addition.
export function buildContinue(s: GoalState, last?: LastCheck | null, doneError?: string | null): string {
  const header = ANNEAL_ON
    ? `Goal not yet met (cycle ${s.cycle}/${s.maxCycles}, phase: ${currentBand(s)}).`
    : `Goal not yet met (cycle ${s.cycle}/${s.maxCycles}).`;
  const parts = [header, `Objective: ${s.objective}`];
  if (last && last.code !== 0) parts.push(`Last check: \`${last.cmd}\` exited ${last.code}:\n${clip(last.output)}`);
  if (doneError) parts.push(`Your last goal_done was rejected — ${doneError}`);
  parts.push(
    ANNEAL_ON
      ? bandGuidance(currentBand(s), s)
      : `Keep working toward the goal. Completion check (must pass before you finish):\n${checkText(s)}\n` +
          `When that check actually passes, call the goal_done tool to finish — invoke the tool, don't just say you're done, and never claim done on assumption.`,
  );
  return parts.join("\n");
}

// Detect an aborted turn (the user hit Esc). pi finalizes the partial assistant message with
// stopReason "aborted"; the most recent assistant message in the run carries it. Shared signal with
// interrupt-notice (which latches the same thing) — here it tells the loop to YIELD, so Esc stops it.
export function lastAssistantStopReason(messages: unknown): string | undefined {
  const arr = Array.isArray(messages) ? messages : [];
  for (let i = arr.length - 1; i >= 0; i--) {
    const m = arr[i] as { role?: string; stopReason?: string };
    if (m?.role === "assistant") return m.stopReason;
  }
  return undefined;
}

// Durable terminal/cycle snapshot (R6 template). One current-state file per project,
// mirroring the STATE.md autonomous-loop convention.
export function buildSnapshot(s: GoalState, last?: LastCheck | null): string {
  const checkLine = last ? `\`${last.cmd}\` exited ${last.code}` : "—";
  const statusLine =
    s.status === "concluded"
      ? `Status: concluded — ${s.outcome ?? "partial"}${s.summary ? `: ${s.summary}` : ""}`
      : `Status: ${s.status}${s.blockedReason ? ` — ${s.blockedReason}` : ""}`;
  return [
    `# Goal status`,
    ``,
    `Objective: ${s.objective}`,
    statusLine,
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

// ---- reminder marker (shared convention; see grounding's ANCHOR note) -----------------------
// Wrap injected guidance so the model can tell it from the user's own words, and fold it into the
// trailing user turn (or append a fresh user message when the tail is a toolResult/assistant) so
// it never reads as a new user instruction. Marker bytes MUST match every other extension's.
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
  let state = freshGoal();
  let goalFile: string | null = null;
  let sessionDir: string | null = null;
  let sessionId = "session";
  let memDir: string | null = null;
  let cwd = process.cwd();
  let lastCheck: LastCheck | null = null;
  let lastDoneError: string | null = null; // why the last goal_done was rejected → fed back into the next push
  let inAgentEnd = false; // re-entrancy guard: process each agent_end at most once
  let userInterjected = false; // set when the user TYPES (input source "interactive") → loop yields
  let abortPending = false; // latched when the user ABORTS a turn (Esc) → loop yields, doesn't re-engage

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
          const terminal = loaded.status === "done" || loaded.status === "blocked" || loaded.status === "concluded";
          state = {
            objective: loaded.objective,
            doneWhen: typeof loaded.doneWhen === "string" ? loaded.doneWhen : null,
            check: typeof loaded.check === "string" ? loaded.check : null,
            maxCycles: Number.isFinite(loaded.maxCycles) ? loaded.maxCycles : DEFAULT_MAX_CYCLES,
            cycle: Number.isFinite(loaded.cycle) ? loaded.cycle : 0,
            status: terminal ? loaded.status : "active",
            blockedReason: loaded.blockedReason,
            outcome: loaded.outcome === "partial" || loaded.outcome === "abandoned" ? loaded.outcome : undefined,
            summary: typeof loaded.summary === "string" ? loaded.summary : undefined,
          };
        }
      }
    } catch {}
  });

  pi.registerTool({
    name: "goal_set",
    label: "Set Goal",
    description: "Set the objective and loop until done. check = what to verify before finishing (defaults to 'verify, don't assume'); done_when = optional shell cmd (exit 0 = done) for a machine stop; steps go in plan_set.",
    parameters: Type.Object({
      objective: Type.String({ description: "One-line objective, e.g. 'all unit tests pass'" }),
      check: Type.Optional(Type.String({ description: "What to verify before declaring done, e.g. 're-render the SVG and confirm it reads as a pelican on a bike'" })),
      done_when: Type.Optional(Type.String({ description: "Shell command; exit 0 = met (machine stop)" })),
      max_cycles: Type.Optional(Type.Number({ description: "Max loop cycles before giving up (default 20)" })),
    }),
    async execute(_id, params) {
      const objective = (params.objective ?? "").trim();
      if (!objective) return errResult(`goal_set needs an objective, e.g. goal_set(objective="all tests green", done_when="pytest -q").`);
      if (objective.length > MAX_OBJECTIVE_LEN) return errResult(`Objective too long (${objective.length} chars). Keep it under ${MAX_OBJECTIVE_LEN} — one line; break the work into steps with plan_set.`);
      const doneWhen = (params.done_when ?? "").trim() || null;
      const check = (params.check ?? "").trim() || null;
      let maxCycles = Number.isFinite(params.max_cycles) ? Math.floor(params.max_cycles as number) : DEFAULT_MAX_CYCLES;
      maxCycles = Math.max(1, Math.min(MAX_MAX_CYCLES, maxCycles));
      state = { objective, doneWhen, check, maxCycles, cycle: 0, status: "active" };
      lastCheck = null;
      lastDoneError = null;
      // ARM the loop — clear both yield latches so a stale interjection/abort from an earlier
      // (inactive) turn can't swallow this goal's first re-engagement. (message_end latches abort
      // regardless of goal status, and agent_end only clears it past the active-status guard.)
      userInterjected = false;
      abortPending = false;
      persist();
      snapshot();
      const mode = isAutonomous(state)
        ? `Autonomous: after each turn I run \`${doneWhen}\`; while it fails I re-engage you (up to ${maxCycles} cycles), and stop when it passes.`
        : `Self-judged: I re-engage you each turn (up to ${maxCycles} cycles) until the completion check passes and you call goal_done. (Stops on Esc, the moment you type, or on /goal clear.)`;
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
      if (remaining.length) unmet.push(`plan steps not done: ${remaining.join("; ")} (mark each with plan_check, or revise the plan)`);
      if (unmet.length) {
        lastDoneError = unmet.join("; "); // fed into the next re-engagement so the loop gives a reason, not a bare "not yet"
        return errResult(`Not done yet — fix these, then call goal_done again:\n- ${unmet.join("\n- ")}`);
      }
      state.status = "done";
      lastDoneError = null;
      persist();
      snapshot();
      return okResult(`Goal complete: ${state.objective}`);
    },
  });

  // The model-owned stop affordance (PRD FR6 / §6.3b). goal_done is the VERIFIED finish; goal_conclude
  // is the deliberate concession — "I've landed the best honest outcome and I'm stopping" — for when
  // the objective is only partially met or genuinely can't be done. It is GATED to the cold bands
  // (commit/decide): conceding early would be an escape hatch, so before the loop has spent most of
  // its budget the tool refuses and tells the model to keep working. This is what lets the loop end
  // on the model's own stated terms instead of a silent budget guillotine.
  pi.registerTool({
    name: "goal_conclude",
    label: "Conclude Goal",
    description:
      "Deliberately stop the goal with a stated outcome when it's only partially met or can't be done (NOT the verified-done path — that's goal_done). Allowed only once the loop is near its budget (commit/decide phase).",
    parameters: Type.Object({
      outcome: Type.Union([Type.Literal("partial"), Type.Literal("abandoned")], {
        description: "'partial' = some of the objective achieved; 'abandoned' = it genuinely can't be done",
      }),
      summary: Type.String({ description: "One line: what was achieved / what's unverified or blocking, and why you're stopping" }),
    }),
    async execute(_id, params) {
      if (!state.objective) return errResult(`No goal set — nothing to conclude.`);
      if (state.status !== "active") return okResult(`Goal already ${state.status}.`);
      // Gate to the cold tail. When annealing is off there are no bands, so allow it (the loop is
      // otherwise flat and the model still needs an honest exit).
      if (ANNEAL_ON && !isColdBand(currentBand(state))) {
        return errResult(
          `Too early to conclude — you're in the ${currentBand(state)} phase (cycle ${state.cycle}/${state.maxCycles}). ` +
            `Keep working; goal_conclude unlocks in the commit/decide phase near the budget. If it's truly met, call goal_done.`,
        );
      }
      const summary = (params.summary ?? "").trim();
      if (!summary) return errResult(`goal_conclude needs a one-line summary of what was achieved and why you're stopping.`);
      state.status = "concluded";
      state.outcome = params.outcome;
      state.summary = summary.slice(0, 500);
      lastDoneError = null;
      persist();
      snapshot();
      return okResult(`Goal concluded (${state.outcome}): ${state.summary}`);
    },
  });

  // Human entry point: `/goal <task>` (set + START working on it), `/goal` (show), `/goal clear`.
  pi.registerCommand("goal", {
    description: "Set a goal and start working on it now: /goal <task>. Also /goal (show), /goal clear. A done_when (for cross-turn auto-re-looping) is set via goal_set; steps via plan_set.",
    handler: async (args, ctx) => {
      const notify = (msg: string) => { try { (ctx as any)?.ui?.notify?.(msg); } catch {} };
      const a = (args ?? "").trim();
      if (a === "clear") {
        state = freshGoal(); lastCheck = null; lastDoneError = null; persist(); snapshot();
        notify("Goal cleared.");
        return;
      }
      if (!a) { notify(state.objective ? renderGoal(state, lastCheck) : "No goal set."); return; }
      // Set the north-star (concise) AND start the work: drive a turn with the FULL task text via
      // sendUserMessage (source "extension"), the same lever `pipe` uses. The agent_end driver then
      // re-engages each turn (self-judged loop) until Gemma checks the goal reached and calls
      // goal_done, or the cycle cap. It yields the instant you type (input source "interactive").
      state = { ...freshGoal(), objective: a.slice(0, MAX_OBJECTIVE_LEN) };
      lastDoneError = null;
      userInterjected = false; abortPending = false; // arm the loop (clear stale interjection/abort latches)
      persist(); snapshot();
      notify(`Goal set — working until reached (Esc or type to take over, /goal clear to stop): ${state.objective}`);
      pi.sendUserMessage(a, { deliverAs: "followUp" });
    },
  });

  // R1: the immutable objective goes into the byte-stable system prefix (one line +
  // a fixed instruction), so it is the always-present anchor and is paid once.
  pi.on("before_agent_start", async (event) => {
    if (!state.objective || state.status !== "active") return;
    const block = `## Goal (north-star)\n${state.objective}\nKeep working until reached. Completion check (must pass before you finish):\n${checkText(state)}\nWhen it passes, call the goal_done tool to finish — invoke the tool, never claim done on assumption.`;
    return { systemPrompt: `${event.systemPrompt}\n\n${block}` };
  });

  // R1: dynamic progress (cycle, last check) is injected each turn — folded into the trailing
  // user turn as a wrapped <reminder> (tail, never the prefix), so it doesn't impersonate the user.
  pi.on("context", async (event) => {
    if (!state.objective || state.status !== "active") return;
    return { messages: foldReminder(event.messages as Msg[], renderGoal(state, lastCheck)) };
  });

  // Channel B (PRD §6.4, opt-in): cool the model's actual sampling temperature as the loop runs.
  // The handler's return value replaces the outgoing request payload (sdk.ts onPayload →
  // emitBeforeProviderRequest); applyTempAnneal is the pure, fail-open core (shape-guarded, active-
  // goal-gated). Returning undefined leaves the request untouched.
  pi.on("before_provider_request", async (event) => {
    if (!ANNEAL_ON || !TEMP_ANNEAL_ON || !state.objective || state.status !== "active") return;
    try {
      const next = applyTempAnneal((event as { payload?: unknown }).payload, state);
      if (next !== (event as { payload?: unknown }).payload) return next; // only replace if we changed it
    } catch {
      /* fail open — never break a request over the temperature seam */
    }
  });

  // Yield to the human: when the user TYPES (input source "interactive"), the loop must not
  // re-engage after that turn — they've taken over. sendUserMessage drives turns with source
  // "extension", so the loop's own re-engagements (and the /goal kickoff) are NOT flagged. This is
  // what makes the cross-turn loop safe: it runs autonomously yet steps aside the instant you type.
  pi.on("input", async (event) => {
    if ((event as { source?: string }).source === "interactive") userInterjected = true;
    return { action: "continue" as const };
  });

  // Stop on Esc/abort (primary detect): pi finalizes the aborted assistant turn with stopReason
  // "aborted". Latch it here so the loop yields on the next agent_end. Without this, the loop
  // re-engaged straight through repeated Esc presses — the user could only escape by killing pi.
  pi.on("message_end", async (event) => {
    const m = (event as { message?: { role?: string; stopReason?: string } }).message;
    if (m?.role === "assistant" && m?.stopReason === "aborted") abortPending = true;
  });

  // The loop driver (push) — CROSS-TURN re-engagement, for ANY active goal. The stop signal differs:
  // autonomous goals stop when done_when passes; self-judged goals stop when Gemma calls goal_done
  // after applying the completion check. Either way it re-engages until stop or the cycle cap. It is
  // gated so a turn YOU started or aborted never re-engages (no hijack, and Esc actually stops it),
  // and re-entrancy-guarded so each agent_end is processed once.
  pi.on("agent_end", async (event) => {
    if (state.status !== "active" || !state.objective || inAgentEnd) return; // no/cleared goal → nothing to drive
    // Esc/abort during a goal turn is a STOP signal — yield like a human interjection so the loop
    // does not re-engage. Detect via the latch (message_end) OR a backstop scan of this run's
    // messages, so it fires regardless of which event a given pi build emits on abort.
    const aborted = abortPending || lastAssistantStopReason((event as { messages?: unknown }).messages) === "aborted";
    abortPending = false;
    if (aborted || userInterjected) { userInterjected = false; return; } // you drove/stopped this turn → yield
    inAgentEnd = true;
    try {
      // Autonomous: done_when is the authoritative machine stop — done even if Gemma forgot goal_done.
      if (isAutonomous(state)) {
        const code = await runDoneWhen();
        if (code === 0) { state.status = "done"; persist(); snapshot(); return; }
      }
      // Still active (self-judged Gemma hasn't called goal_done, or done_when still failing) →
      // re-engage for another cycle, restating the objective + completion check for drift correction.
      if (state.cycle < state.maxCycles) {
        state.cycle++; persist(); snapshot();
        const doneErr = lastDoneError; lastDoneError = null; // surface a rejected goal_done once, then clear
        pi.sendUserMessage(buildContinue(state, lastCheck, doneErr), { deliverAs: "followUp" });
      } else {
        // BLOCKED is durable, not silent — stop cleanly, never a silent runaway. By now the model has
        // already had its decide-phase turn (the last re-engagement used the decide band) and chose
        // neither goal_done nor goal_conclude, so this is the hard backstop, not a surprise cut (FR7).
        state.status = "blocked";
        state.blockedReason = ANNEAL_ON
          ? `cycle budget exhausted (${state.maxCycles}); the final decide-phase turn neither reached the goal nor called goal_conclude`
          : `cycle budget exhausted (${state.maxCycles}) without the goal being reached`;
        persist(); snapshot();
      }
    } finally {
      inAgentEnd = false;
    }
  });
}
