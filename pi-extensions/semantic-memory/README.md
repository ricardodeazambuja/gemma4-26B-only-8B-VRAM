# semantic-memory

Cross-session memory with **no retrieval burden on the model**. Small models don't
reliably call a "search your memory" tool at the right moment — they can't query
for what they don't know they forgot. So the extension does the retrieval *for*
Gemma: it embeds each incoming user message and injects the most relevant past
facts at the tail of the context, automatically. PLAN.md item 5.

## Tools

- **`remember(fact)`** — save a durable one-line fact. Writes it to the curated
  `MEMORY.md` *and* the vector index.
- **`recall(query, k=3)`** — explicit semantic search (for when the model does know
  to look).
- **`forget(match)`** — remove a fact by substring, from both stores. Wrong facts
  are worse than missing ones, so deletion is first-class.

## The automatic parts (where the value is)

- **Auto-recall** (`context` hook): embeds the latest user message, cosine-searches
  the index, and injects the top 2 chunks (≥0.55 similarity) at the **tail**. No
  tool call, invisible to Gemma's tool-selection budget, cache-safe (rule R1).
- **Passive MEMORY.md injection** (`before_agent_start`): the curated file (capped
  at 1 KB) is appended to the system prompt. It's snapshotted at `session_start`
  and never re-read mid-session, so the text is byte-stable → the prompt prefix and
  KV cache hold.
- **Snapshot ingestion**: the `Task/Done/Next/Files` snapshots that `plan` writes
  before compaction are embedded into the index on `session_start`, then moved to
  `snapshots/ingested/` so they're only ingested once.

## Storage

`~/.pi/memory/<project-slug>/`:
- `MEMORY.md` — curated, human-editable, always injected.
- `chunks.jsonl` — one record per chunk; vector packed as base64 Float32 to stay
  compact. `decodeVector` copies into an aligned buffer (Buffer's pool offset isn't
  guaranteed 4-byte aligned).

## Index

Brute-force cosine (`search.ts`). At ≤ ~10⁴ chunks of 768-dim vectors this is
single-digit milliseconds in plain JS — **no ANN, no quantization, no Turbovec**
(rejected: alpha-stage, only pays off ≫10⁵ vectors). Isolated behind
`cosineSearch` so the index is swappable if the corpus ever grows.

## Embeddings — run a second llama-server

```bash
llama-server --embeddings -m embeddinggemma-Q8_0.gguf --port 8081
```

Configure via env (defaults shown):

| Var | Default |
|-----|---------|
| `PI_EMBED_URL` | `http://127.0.0.1:8081/v1/embeddings` |
| `PI_EMBED_MODEL` | `embeddinggemma` |
| `PI_EMBED_TIMEOUT_MS` | `4000` |

**Fail-soft:** if the server is down, `embed()` returns `null` and never throws —
`remember` stores facts without a vector (lazily indexable later), `recall` falls
back to substring search, and auto-recall quietly does nothing. Gemma's turn is
never broken by a missing embedding server.

## Test

```bash
node --experimental-strip-types semantic-memory/test.mjs
```

30 assertions covering the search math, store I/O (incl. the Float32 alignment
fix), both embedding-response shapes, server-down fallback, and the full
remember → auto-recall → forget → snapshot-ingest pipeline driven through a fake
embedding server.
