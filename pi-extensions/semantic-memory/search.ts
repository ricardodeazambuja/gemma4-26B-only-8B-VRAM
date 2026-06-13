// Brute-force cosine similarity. At the scale this runs (≤ ~10^4 chunks of
// EmbeddingGemma 768-dim vectors) this is single-digit milliseconds in plain JS —
// no ANN, no quantization, no Turbovec. Kept behind this module so the index can
// be swapped later without touching index.ts.

export interface Chunk {
  id: string;
  text: string;
  source: string;
  date: string;
  vector?: number[]; // absent when embedding was unavailable at write time
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function cosine(a: number[], b: number[]): number {
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

export interface Scored { chunk: Chunk; score: number; }

/** Top-k chunks by cosine to queryVec, above `threshold`. Chunks without a
 * vector are skipped (they can't be scored). */
export function cosineSearch(queryVec: number[], chunks: Chunk[], k: number, threshold = 0): Scored[] {
  const scored: Scored[] = [];
  for (const c of chunks) {
    if (!c.vector || !c.vector.length) continue;
    const score = cosine(queryVec, c.vector);
    if (score >= threshold) scored.push({ chunk: c, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

/** Fallback when no embedding is available: rank by overlap of lowercased word
 * tokens between query and chunk text. Crude but better than nothing. */
export function substringSearch(query: string, chunks: Chunk[], k: number): Scored[] {
  const terms = query.toLowerCase().split(/\W+/).filter((t) => t.length > 2);
  if (!terms.length) return [];
  const scored: Scored[] = [];
  for (const c of chunks) {
    const hay = c.text.toLowerCase();
    let score = 0;
    for (const t of terms) if (hay.includes(t)) score++;
    if (score > 0) scored.push({ chunk: c, score });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}
