#!/usr/bin/env node
// One-time energy calibration for the stats extension (Linux + Intel RAPL).
//
// Usage:
//   1. Start a sustained Gemma generation in pi (e.g. ask it to write a long file)
//      and note its tokens/sec from llama.cpp's logs.
//   2. While it's generating, run:
//        node --experimental-strip-types stats/calibrate.mjs --project /path/to/project --tps 18
//   3. It samples package power over ~8 s and writes calibration.json with J/token.
//
// J/token = average_watts / tokens_per_second.  This is a deliberate simplification
// (one rate for prefill+decode); good enough to compare sessions, not a power meter.
import { sampleAveragePowerW, readRaplEnergyUj, saveCalibration } from "./energy.ts";
import { memoryDir } from "./core.ts";

const args = process.argv.slice(2);
const get = (flag) => { const i = args.indexOf(flag); return i >= 0 ? args[i + 1] : undefined; };

const project = get("--project") || process.cwd();
const tps = Number(get("--tps") || 0);
const seconds = Number(get("--seconds") || 8);

if (readRaplEnergyUj() === null) {
  console.error("RAPL not readable at /sys/class/powercap (need Intel CPU + read permission).");
  console.error("You can still set J/token by hand: write {\"jPerToken\": <value>} to");
  console.error(`  ${memoryDir(project)}/calibration.json`);
  process.exit(1);
}

console.log(`Sampling package power for ${seconds}s — keep Gemma generating…`);
const watts = await sampleAveragePowerW(seconds * 1000);
if (watts === null) { console.error("Sampling failed."); process.exit(1); }
console.log(`Average package power: ${watts.toFixed(1)} W`);

if (!tps || tps <= 0) {
  console.error("\nPass --tps <tokens/sec observed during generation> to compute J/token, e.g. --tps 18");
  process.exit(1);
}

const jPerToken = watts / tps;
const dir = memoryDir(project);
saveCalibration(dir, { jPerToken, measuredW: watts, tokensPerSec: tps, note: "package power / tokens-per-sec" });
console.log(`\nWrote ${dir}/calibration.json`);
console.log(`  jPerToken = ${jPerToken.toFixed(3)}  (${watts.toFixed(1)} W ÷ ${tps} tok/s)`);
console.log("/stats will now show estimated Wh.");
