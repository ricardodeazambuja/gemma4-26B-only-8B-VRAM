// Verify the embedding backend by driving semantic-memory's OWN embed() against
// it — so a green result means the extension works, not just that some server is up.
// Run: node --experimental-strip-types semantic-memory/verify-embeddings.mjs
//   env: PI_EMBED_URL (default Ollama OpenAI-compat), PI_EMBED_MODEL, PI_EMBED_TIMEOUT_MS
import { embed, defaultConfig } from "./embed.ts";
import { cosine } from "./search.ts";

const cfg = {
  ...defaultConfig(),
  url: process.env.PI_EMBED_URL || "http://127.0.0.1:11434/v1/embeddings",
  model: process.env.PI_EMBED_MODEL || "embeddinggemma",
  timeoutMs: Number(process.env.PI_EMBED_TIMEOUT_MS || 20000),
};

console.log(`Verifying embeddings via ${cfg.url} (model: ${cfg.model})`);

const v1 = await embed("the database schema lives in db/schema.sql", cfg);
if (!v1) {
  console.error("✗ embed() returned null — server unreachable, wrong URL/model, or bad response.");
  console.error("  Is the server running? Try:  curl " + cfg.url);
  process.exit(1);
}
console.log(`✓ got an embedding: ${v1.length} dims, sample [${v1.slice(0, 3).map((x) => x.toFixed(4)).join(", ")} …]`);

// Sanity: a related query should score higher than an unrelated one.
const vRelated = await embed("where is the database schema defined?", cfg);
const vUnrelated = await embed("what time does the train leave on tuesday?", cfg);
if (vRelated && vUnrelated) {
  const sRel = cosine(v1, vRelated);
  const sUnrel = cosine(v1, vUnrelated);
  console.log(`✓ similarity sanity: related=${sRel.toFixed(3)}  unrelated=${sUnrel.toFixed(3)}`);
  if (sRel <= sUnrel) {
    console.error("✗ related query did NOT score higher than unrelated — embeddings look wrong.");
    process.exit(1);
  }
}
console.log("\n✓ semantic-memory embeddings are working. Set in pi's environment:");
console.log(`    export PI_EMBED_URL=${cfg.url}`);
console.log(`    export PI_EMBED_MODEL=${cfg.model}`);
process.exit(0);
