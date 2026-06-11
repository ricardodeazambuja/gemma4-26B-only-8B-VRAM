// Energy helpers: read Intel RAPL on Linux and convert tokens → Wh using a
// one-time calibration. Everything degrades to null/0 when RAPL is unavailable
// (non-Intel, no permissions, macOS) so the rest of stats still works. PLAN.md item 7.

import { join } from "node:path";
import { readFileSync, existsSync, writeFileSync, readdirSync } from "node:fs";

const RAPL_BASE = "/sys/class/powercap";

/** Sum the energy counters (microjoules) of all intel-rapl package domains.
 * Returns null if RAPL isn't readable. `base` is injectable for tests. */
export function readRaplEnergyUj(base = RAPL_BASE): number | null {
  if (!existsSync(base)) return null;
  let total = 0;
  let found = false;
  let dirs: string[] = [];
  try { dirs = readdirSync(base); } catch { return null; }
  for (const d of dirs) {
    if (!/^intel-rapl:\d+$/.test(d)) continue; // top-level package domains only
    const f = join(base, d, "energy_uj");
    try {
      const v = Number(readFileSync(f, "utf8").trim());
      if (Number.isFinite(v)) { total += v; found = true; }
    } catch {}
  }
  return found ? total : null;
}

export interface Calibration { jPerToken: number; note?: string; measuredW?: number; tokensPerSec?: number; }

export function calibrationPath(dir: string): string { return join(dir, "calibration.json"); }

export function loadCalibration(dir: string): Calibration | null {
  const p = calibrationPath(dir);
  if (!existsSync(p)) return null;
  try {
    const c = JSON.parse(readFileSync(p, "utf8"));
    return Number.isFinite(c.jPerToken) ? c : null;
  } catch { return null; }
}

export function saveCalibration(dir: string, c: Calibration): void {
  writeFileSync(calibrationPath(dir), JSON.stringify(c, null, 2));
}

/** Joules for N tokens at the calibrated rate. */
export function joulesForTokens(tokens: number, jPerToken: number): number {
  return tokens * jPerToken;
}

export function joulesToWh(j: number): number {
  return j / 3600;
}

/** Average package power (W) over a window, by sampling RAPL twice. Returns null
 * if RAPL is unavailable. `sampler` and `sleep` are injectable for tests. */
export async function sampleAveragePowerW(
  durationMs: number,
  sampler: () => number | null = readRaplEnergyUj,
  sleep: (ms: number) => Promise<void> = (ms) => new Promise((r) => setTimeout(r, ms)),
): Promise<number | null> {
  const e0 = sampler();
  if (e0 === null) return null;
  await sleep(durationMs);
  const e1 = sampler();
  if (e1 === null) return null;
  let deltaUj = e1 - e0;
  if (deltaUj < 0) deltaUj += Number.MAX_SAFE_INTEGER; // counter wrap (rare); best-effort
  const joules = deltaUj / 1e6;
  return joules / (durationMs / 1000);
}
