// anneal — the cooling schedule for the goal loop (PRD: docs/goal-annealing-prd.md).
//
// One pure schedule over the loop's own counter (state.cycle / state.maxCycles): a "temperature"
// that decays as cycles accumulate, and a band (the teacher register) selected from it. The goal
// extension feeds both into its tail push (buildContinue) so the nudge cools explore → consolidate
// → commit → decide, and — optionally, Channel B — into the model's sampling temperature.
//
// Design choices that matter (see PRD §6.1):
//   • BANDS are chosen by RESERVED CYCLE COUNTS, not by raw T. That keeps the arc sane for tiny
//     budgets (FR4): always ≥1 explore when maxCycles>1, always exactly one decide (the last cycle),
//     maxCycles=1 ⇒ pure decide. Raw T is only for display + Channel B.
//   • T is a normalized [0,1] decay (1 at cycle 0 → 0 at the budget end), monotone non-increasing.
//   • Everything here is PURE (no I/O, no Date/random) so the goal test.mjs can assert boundaries
//     directly and a resume replays identically. Config is read from env by a single helper.
//
// This module is goal-owned (PRD §6.6a): grounding stays untouched, so the schedule lives next to
// goal rather than in a shared extension dir. If we ever cool grounding's CHECK directly (§6.6b),
// promote this file to a shared location — the functions are already pure and dependency-free.

export type Band = "explore" | "consolidate" | "commit" | "decide";

export interface AnnealConfig {
  /** Fraction of the budget reserved for the cold tail (commit+decide). Default 0.25. */
  commitFraction: number;
  /** Upper bound on progress p=cycle/maxCycles for the explore band. Default 0.5. */
  exploreFraction: number;
  /**
   * Temperature decay shape for DISPLAY and Channel B (bands do NOT depend on this).
   *   • "cosine" (default) — the ML cosine-annealing curve T=(1+cos(πp))/2. HOLDS heat through the
   *     explore half, steepest drop mid-run, flattens cool into commit/decide. This is what tracks
   *     the band structure. (Classic geometric cooling is convex — it drops fastest at the START,
   *     which is backwards for a schedule meant to stay exploratory while there's budget.)
   *   • "linear" — straight ramp T=1−p (predictable; handy for tests/A-B).
   */
  shape: "cosine" | "linear";
  /** Channel B sampling-temperature range mapped from T∈[0,1] → [lo,hi]. */
  tempLo: number;
  tempHi: number;
}

export const DEFAULT_ANNEAL: AnnealConfig = {
  commitFraction: 0.25,
  exploreFraction: 0.5,
  shape: "cosine",
  tempLo: 0.3,
  tempHi: 1.0,
};

const clamp = (x: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, x));
const finite = (x: unknown, fallback: number): number => (typeof x === "number" && Number.isFinite(x) ? x : fallback);

// Reserved cold-tail length in cycles (commit band + the single decide cycle). ≥1 always.
export function reservedCommit(maxCycles: number, cfg: AnnealConfig = DEFAULT_ANNEAL): number {
  const m = Math.max(1, Math.floor(maxCycles));
  return Math.max(1, Math.ceil(m * clamp(cfg.commitFraction, 0, 1)));
}

// The teacher register for this cycle. Order matters — decide first (the last cycle is always a
// forced decision), then the reserved commit tail, then explore (guaranteed ≥ cycle 1 when the
// budget allows), then the consolidate middle.
//
//   maxCycles=1  → {1:decide}
//   maxCycles=2  → {1:explore, 2:decide}
//   maxCycles=3  → {1:explore, 2:consolidate, 3:decide}
//   maxCycles=20 → {1-10:explore, 11-15:consolidate, 16-19:commit, 20:decide}
//
// cycle is the loop's 1-based re-engagement counter (buildContinue fires at cycle ≥ 1). A cycle at
// or past the budget is decide (defensive ≥, in case the counter is nudged past the cap).
export function bandFor(cycle: number, maxCycles: number, cfg: AnnealConfig = DEFAULT_ANNEAL): Band {
  const m = Math.max(1, Math.floor(maxCycles));
  const c = Math.max(1, Math.floor(cycle));
  if (c >= m) return "decide";
  const reserved = reservedCommit(m, cfg);
  if (c > m - reserved) return "commit"; // inside the cold tail but not the decide cycle
  const exploreEnd = Math.max(1, Math.floor(m * clamp(cfg.exploreFraction, 0, 1)));
  if (c <= exploreEnd) return "explore"; // guarantees ≥1 explore cycle when m>1
  return "consolidate";
}

// True once the loop has entered the cold tail — the only window where conceding (goal_conclude)
// is allowed (PRD FR6): early concession would be an escape hatch, so we gate it to commit/decide.
export function isColdBand(band: Band): boolean {
  return band === "commit" || band === "decide";
}

// Normalized temperature in [0,1]: 1 at cycle 0, 0 at the budget end, monotone non-increasing.
// Used for DISPLAY and Channel B only (bands are reserved-count based, above). Cosine (default)
// holds heat early then drops; linear is the straight ramp.
export function temperature(cycle: number, maxCycles: number, cfg: AnnealConfig = DEFAULT_ANNEAL): number {
  const m = Math.max(1, Math.floor(maxCycles));
  const c = clamp(Math.floor(cycle), 0, m);
  const p = c / m; // progress in [0,1]
  if (cfg.shape === "linear") return clamp(1 - p, 0, 1);
  // cosine annealing: (1+cos(πp))/2 falls 1→0, concave early (holds heat), steepest at p=0.5.
  return clamp((1 + Math.cos(Math.PI * p)) / 2, 0, 1);
}

// Channel B (PRD §6.4): map the normalized T to an actual sampling temperature in [lo,hi].
// Hot loop start → hi (diverse/exploratory); cold end → lo (greedy/decisive).
export function samplingTemperature(cycle: number, maxCycles: number, cfg: AnnealConfig = DEFAULT_ANNEAL): number {
  const lo = cfg.tempLo;
  const hi = cfg.tempHi;
  const t = temperature(cycle, maxCycles, cfg);
  return clamp(lo + (hi - lo) * t, Math.min(lo, hi), Math.max(lo, hi));
}

// Read overrides from the environment (single I/O seam; the rest of the module is pure). Unknown /
// malformed values fall back to the default, so a typo never breaks the loop.
export function annealConfigFromEnv(env: Record<string, string | undefined> = process.env): AnnealConfig {
  const num = (key: string, fallback: number) => {
    const v = env[key];
    if (v == null || v === "") return fallback;
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  };
  const shape = env.PI_GOAL_ANNEAL_SHAPE === "linear" ? "linear" : DEFAULT_ANNEAL.shape;
  return {
    commitFraction: clamp(num("PI_GOAL_COMMIT_FRACTION", DEFAULT_ANNEAL.commitFraction), 0, 1),
    exploreFraction: clamp(num("PI_GOAL_EXPLORE_FRACTION", DEFAULT_ANNEAL.exploreFraction), 0, 1),
    shape,
    tempLo: finite(num("PI_GOAL_TEMP_LO", DEFAULT_ANNEAL.tempLo), DEFAULT_ANNEAL.tempLo),
    tempHi: finite(num("PI_GOAL_TEMP_HI", DEFAULT_ANNEAL.tempHi), DEFAULT_ANNEAL.tempHi),
  };
}

// Channel A on/off (default ON — pure-prompt, safe). Channel B on/off (default OFF — untyped seam).
export function annealEnabled(env: Record<string, string | undefined> = process.env): boolean {
  return env.PI_GOAL_ANNEAL !== "0" && env.PI_GOAL_ANNEAL !== "off";
}
export function tempAnnealEnabled(env: Record<string, string | undefined> = process.env): boolean {
  return env.PI_GOAL_TEMP_ANNEAL === "1" || env.PI_GOAL_TEMP_ANNEAL === "on";
}
