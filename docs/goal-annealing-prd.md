# PRD: Annealed nudging for the `goal` loop and grounding checks

| | |
|---|---|
| **Status** | Draft |
| **Owner** | ricardodeazambuja |
| **Date** | 2026-06-14 |
| **Affects** | `pi-extensions/goal/`, `pi-extensions/grounding/`, a shared `anneal` schedule module, new `before_provider_request` seam |
| **Companion docs** | [`goal/README.md`](../pi-extensions/goal/README.md), [`grounding/README.md`](../pi-extensions/grounding/README.md), [`TECHNICAL.md`](./TECHNICAL.md) |

A design for turning the `goal` extension's flat re-engagement loop into an **annealed** one:
nudge the model like a patient teacher early (broad, exploratory), cool the nudge as cycles
accumulate, and — instead of cutting the work off at the budget — ramp into a phase that
**pushes the model to make and verify a final decision, then stop on its own**.

The *same* schedule also cools the **verification pressure**: early it presses for broad
verification; late it narrows to "verify only what's decision-critical, apply the standing flag-rule
to the rest, and decide." `grounding`'s `MINDSET` *and* `CHECK` stay fixed as the floor — `MINDSET`
already carries an always-on answer-time *establish-it-or-flag-it* rule — so annealing changes
*emphasis and effort-triage*, never the verification **bar**. Where that cooling register lives
(inside `goal`'s push, vs. `grounding`'s `CHECK`) is forced by a hard constraint: **pi shares no live
module state across extensions** (§6.6).

- [1. Problem & motivation](#1-problem--motivation)
- [2. Goals & non-goals](#2-goals--non-goals)
- [3. Background: how the loop works today](#3-background-how-the-loop-works-today)
- [4. The concept: annealing-inspired scheduling](#4-the-concept-annealing-inspired-scheduling)
- [5. Functional requirements](#5-functional-requirements)
- [6. Design sketch](#6-design-sketch)
- [7. Interactions & dependencies](#7-interactions--dependencies)
- [8. Configuration & defaults](#8-configuration--defaults)
- [9. Risks, edge cases & mitigations](#9-risks-edge-cases--mitigations)
- [10. Testing & validation](#10-testing--validation)
- [11. Observability](#11-observability)
- [12. Rollout / phasing](#12-rollout--phasing)
- [13. Success metrics](#13-success-metrics)
- [14. Open questions](#14-open-questions)
- [Appendix A: glossary](#appendix-a-glossary)
- [Appendix B: code references](#appendix-b-code-references)

---

## 1. Problem & motivation

`goal` drives a cross-turn loop: after each agent run it re-engages the model with the same
restated objective + completion check until the goal is met or a cycle budget is exhausted
(`buildContinue` → `sendUserMessage`, `goal/index.ts:115-124`, `405-408`). In parallel,
`grounding` re-injects the same act-now `CHECK` ("prove it before you answer") once per turn. Three
weaknesses:

1. **The nudge is flat.** Cycle 1 and cycle 19 get the same push — from *both* `buildContinue` and
   `CHECK`. A real mentor does not coach identically at the start of a task and at the deadline —
   early, you open the space; late, you force a decision. The loop has the data to do this
   (`state.cycle`, `state.maxCycles`) but spends none of it on *how* it nudges.

2. **Verification has one gear.** `CHECK` presses "establish every claim" at the same intensity on
   cycle 1 and cycle 19. Early that rigor is cheap and valuable; late it becomes
   **over-verification / analysis-paralysis** — the model keeps reading and re-checking instead of
   committing. Grounding has no notion of "you have verified enough; now decide."

3. **The stop is a guillotine, not a landing.** When `cycle` hits `maxCycles` the loop silently
   flips to `blocked` with `"cycle budget exhausted"` (`goal/index.ts:409-413`). The model is never
   told "this is your last chance — decide." Worse, the model has **no affordance to conclude on its
   own terms**: `goal_done` refuses unless `done_when` passes and every plan step is checked
   (`goal/index.ts:296-324`), so a model that has done reasonable-but-incomplete work cannot land
   the plane — it can only be cut off.

We want the loop to behave like a good teacher who also respects the clock: generous and
exploratory while there is time, increasingly directive as time runs out, verification effort that
narrows from "check everything" to "check what the decision hinges on," and — at the end —
**insisting the student commit to their best verified answer and stop**, rather than yanking the
work away mid-thought.

## 2. Goals & non-goals

**Goals**

- G1. A monotone "temperature" schedule `T(cycle, maxCycles)` that decays as the loop progresses,
  in a shared module both `goal` and `grounding` consume.
- G2. Nudge text (`buildContinue`) that shifts **explore → consolidate → commit** as `T` cools.
- G3. A **graceful terminal**: the last cycle(s) explicitly force a decision; the hard `blocked`
  backstop fires only *after* the model has had an explicit "decide now" turn.
- G4. A **model-owned stop affordance**: the model can conclude a goal as `done` / `partial` /
  `abandoned` with a one-line summary, instead of being cut off — gated so it can only concede
  near the end, never bail early.
- G5. (Optional, faithful-annealing channel) Cool the model's **actual sampling temperature** across
  cycles via the provider request.
- G6. Preserve grounding's correctness bar: the commit phase pushes the model to *decide*, never to
  *fabricate* — a committed answer is a verified answer (or one with its gaps explicitly flagged).
- **G7. Anneal the verification pressure on the same schedule:** hot = press for broad verification;
  cold = triage to decision-critical verification, apply MINDSET's standing flag-rule to the rest,
  and converge. This counters over-verification and reinforces `goal`'s `decide` band. (Realized in
  `goal`'s `buildContinue` — recommended — so `grounding` stays untouched; see §6.6.)
- **G8. Keep `grounding`'s `MINDSET` and `CHECK`, `ANCHOR`, and `goal`'s north-star as the fixed
  floor** — *not* annealed. `MINDSET`'s always-on rule is the verification bar; annealing only the
  dynamic push changes emphasis, never the bar.

**Non-goals**

- N1. Implementing true simulated annealing (Metropolis accept-worse-with-probability). See §4 —
  only the cooling *schedule* transfers; the loop has no notion of accepting worse states.
- N2. Replacing `plan`'s step tracking or `done_when`'s machine check. Those stay the authoritative
  signals; annealing changes *coaching tone* and *how the loop ends*, not the completion criteria.
- N3. Changing behavior for non-looping (advisory, pull-only) goals, or for ordinary non-goal turns.
  Annealing applies to the driven loop; `CHECK` outside an active goal keeps today's fixed behavior.
- **N4. Annealing `MINDSET` / `ANCHOR` / the north-star.** Explicitly rejected (§9): primarily it
  would weaken the standing verification floor at the deadline (the dangerous combo with "commit
  now"); secondarily it churns the prefix KV cache.

## 3. Background: how the loop works today

| Piece | Location | Role |
|---|---|---|
| `GoalState` | `goal/index.ts:30-38` | `objective`, `doneWhen`, `check`, `maxCycles`, `cycle`, `status`, `blockedReason` |
| `buildContinue` | `goal/index.ts:115-124` | The per-cycle re-engagement push (tail chokepoint we will anneal) |
| `agent_end` driver | `goal/index.ts:388-418` | Re-engages each cycle; flips to `blocked` at the cap |
| `goal_done` | `goal/index.ts:296-324` | The only finish path; rejects unless `done_when` + plan steps pass |
| `checkText` / `DEFAULT_CHECK` | `goal/index.ts:55-60` | Completion criterion; default is MINDSET's "verify, don't assume" |
| north-star prefix / status tail | `goal/index.ts:353-357`, `361-364` | Always-on objective (cached) + dynamic per-turn status |
| grounding `MINDSET` + `ANCHOR` (prefix) | `grounding/index.ts:123-125` | Byte-stable standing floor; appended to system prompt every call |
| grounding `CHECK` (tail) | `grounding/index.ts:132-140` | Act-now "prove it" reminder; folded once per turn at turn-start (the second tail chokepoint we will anneal) |

Two facts the design leans on:

- **The two tail reminders already co-fold into one turn.** `buildContinue` and `CHECK` both fire at
  loop re-engagement (each `buildContinue` is genuine text → `isTurnStart` true,
  `grounding/index.ts:110-118`), and `foldReminder` collapses them into a single user turn. So
  annealing them on one schedule anneals one coherent block.
- **An extension can rewrite the outgoing provider request** (enables G5/Channel B).
  `before_provider_request` hands the handler the request payload and uses its return value as the
  replacement (`sdk.ts:330-335` → `runner.ts:946-978`); for the llama.cpp server that payload carries
  `temperature`. Caveat: the payload is typed `unknown` (`types.ts:644-647`, result `:1018`) and is
  provider-shaped — an **untyped seam**, not a supported field.

## 4. The concept: annealing-inspired scheduling

Define a normalized progress `p = cycle / maxCycles ∈ [0, 1]` and a temperature `T` that decays from
hot (`T0`) to cold (`~0`) as `p → 1`. Everything else is a function of `T`.

**Budget-driven, not progress-driven.** `T` cools on *cycle count*, not on how close the goal is —
a goal 90%-done at cycle 3 is coached identically to one 10%-done at cycle 3. This is intentional:
it is exactly what enforces "you cannot work on a task forever." Proximity to *done* is already
handled by `done_when` / `goal_done`; the schedule's job is to bound *effort*, not to estimate
completion.

**Anneal the tail, not the cached prefix — and the floor never melts.** The byte-stable system-prefix
texts (`grounding`'s `MINDSET`/`ANCHOR` and `goal`'s north-star) are the *floor*: the invariant
*"establish it, or flag it unverified."* They must **not** anneal, for two reasons in priority order:
(a) **the honesty floor** — weakening the standing verify-rule precisely as the deadline nears, *while*
Channel A is also pushing "commit now," is the dangerous combination; that alone is sufficient; and
(b) **cache** — varying prefix text between cycles costs one extra prefix prefill *per cycle* (bounded
— it changes per cycle, not per call — but real on slow local hardware; see `TECHNICAL.md`). Note
`MINDSET` already enforces the verification **bar** on every call, answer-time included
(`grounding/index.ts:28-29`), so keeping it fixed loses nothing. All annealing lives in the **dynamic
tail**, where per-turn re-injection is already free; only the **act-now intensity** cools.

**Honesty about the metaphor.** Real simulated annealing's defining mechanism is the Metropolis
rule — accept a *worse* candidate with probability `exp(−ΔE/T)` to escape local minima. Our loop
never accepts/rejects worse states, so that part does **not** transfer (N1). What transfers is the
**cooling schedule**: a single decaying control that shifts behavior from exploration to commitment.
There is exactly one place the analogy becomes *literal* — the model's sampling temperature (G5/§6.4).
We treat that as a bonus channel, not the core.

Channels driven by one shared schedule:

- **Channel A — prompt pressure** (primary, reliable). The teacher register of `goal`'s
  `buildContinue` — which carries both the coaching push *and* (recommended) the annealed verification
  register, §6.6 — is a function of `T`. `grounding`'s fixed `CHECK` co-folds into the same user turn,
  so the model reads floor + cooling push as one block.
- **Channel B — sampling temperature** (optional, literal). The provider request's `temperature` is
  a function of `T`: diverse/exploratory early, greedy/decisive late. This matches the repo's
  measured finding that lower sampling temperature is the decisive lever for this model
  (see `mtp-benchmark.md` and the project's temperature notes).

## 5. Functional requirements

- **FR1 — Schedule.** A pure function `temperature(cycle, maxCycles, cfg) → T ∈ [0,1]`, monotone
  non-increasing in `cycle`, in a shared `anneal` module both extensions import. Pure ⇒ unit-testable
  with no I/O.
- **FR2 — Bands.** `T` maps to a phase ∈ {`explore`, `consolidate`, `commit`, `decide`} via
  configurable thresholds. The final cycle is always `decide`.
- **FR3 — Banded nudge.** `buildContinue` selects phase-appropriate coaching text. It keeps the
  existing payload (objective, last check, rejected-`goal_done` reason) and only varies the *framing*.
- **FR4 — Reserved commit tail.** Regardless of decay shape, at least the last `reservedCommit`
  cycles are `commit`/`decide`, and at least the first cycle is `explore` (so tiny budgets still
  anneal sensibly; `maxCycles = 1` ⇒ pure `decide`).
- **FR5 — Decision-forcing terminal.** When the loop reaches the `decide` band, the push explicitly
  states it is the last cycle, the model cannot iterate further, and it must land its best verified
  result and finish — or declare it cannot, with a specific reason.
- **FR6 — Model stop affordance.** The model can finalize via a concede path
  (`goal_done` extended, or a sibling tool — see §6.3) with `outcome ∈ {done, partial, abandoned}`
  and a one-line `summary`. Gated: concession (`partial`/`abandoned`) is honored only in the
  `commit`/`decide` bands; outside them `goal_done` enforces the normal gate so the model cannot bail
  early. `done` always runs the full verification gate.
- **FR7 — Backstop preserved.** The hard `blocked` stop still exists, but fires only *after* at least
  one `decide` turn, and records the model's last stated position when available (not a bare
  "budget exhausted").
- **FR8 — Honesty floor preserved (constant across all bands, goal *and* grounding).** No band may
  instruct the model to state unverified claims as fact. The cold/`decide` register changes *what to
  spend verification effort on*, never *whether to flag* — "verified, or explicitly marked unverified"
  is invariant (mirrors `MINDSET`).
- **FR9 — (Optional) Channel B.** Behind a flag, an active annealing goal sets the provider request's
  sampling temperature from `T`, clamped to a configured `[lo, hi]`, and **only** while such a goal is
  active — no leakage into ordinary turns, no silent clobbering of the user's base temperature.
- **FR10 — Annealed verification register.** The verification pressure cools with the schedule's band
  (FR2), on the same cadence as `buildContinue`: hot = press for broad verification; cold =
  decision-critical verification + apply MINDSET's standing flag-rule to the rest (§6.6). **Where this
  register lives — enriched into `goal`'s `buildContinue` (recommended) vs. `grounding`'s `CHECK`
  mutated in place — is the §6.6 / Q6 decision**, forced by the fact that pi shares no live module
  state across extensions (`loader.ts:331-340`). It reuses the shared `anneal` *schedule* — no second
  schedule. The register changes emphasis, never the bar (FR8).
- **FR11 — `MINDSET` (and `CHECK`) stay fixed.** `MINDSET`, `ANCHOR`, `goal`'s north-star — and, under
  the recommended §6.6 option, `grounding`'s `CHECK` — remain byte-stable and are **not** annealed
  (N4). Primary reason: preserve the **honesty floor** (never relax the standing verify-rule as the
  deadline nears). Secondary: the prefix KV cache (a bounded one-prefill-per-cycle cost). The only
  annealable surface is the dynamic push.
- **FR12 — Coupling & scope.** pi shares no live module state across extensions
  (`loader.ts:331-340`), so the band reaches the verification register either by `goal` owning it
  (§6.6a, recommended — no coupling) or by `grounding` reading `goal`'s published position (§6.6b).
  **Either way, with no active annealing goal behavior is byte-identical to today** — verification
  annealing is purely additive.

## 6. Design sketch

### 6.1 Schedule (shared module)

```
p   = cycle / maxCycles                      // progress in [0,1]
T   = T0 * pow(alpha, cycle)                 // geometric cooling (classic SA shape)
                                             // alpha chosen so T≈threshold at the commit boundary
```

Banding is by **reserved cycle counts** (budget-relative, robust for tiny `maxCycles`) rather than
raw `T`, with `T` still exported for Channel B and display:

```
reservedCommit = max(1, ceil(maxCycles * commitFraction))   // default commitFraction = 0.25
band:
  cycle == maxCycles            -> decide
  cycle  > maxCycles - reserved -> commit
  p      < exploreFraction      -> explore       // default exploreFraction = 0.5
  else                          -> consolidate
```

`temperature()` and `bandFor()` are **pure** and live in a shared `anneal` module (e.g.
`pi-extensions/anneal/`) imported by both `goal` and `grounding`, so neither owns the other's logic
and the existing `test.mjs` files can assert boundaries directly.

### 6.2 Banded `buildContinue` (Channel A — goal)

Same data, four framings (illustrative, to be finalized in a follow-up spec):

- **explore** — *Socratic.* "You have budget to explore. Don't lock in yet. What approaches haven't
  you tried? What are you assuming that you haven't checked (derived/run/read) this session?"
- **consolidate** — *directive.* "You've explored. Converge: pick the most promising line, deepen it,
  and start closing open threads. Don't open new directions unless the current one is failing."
- **commit** — *decisive.* "Budget is nearly spent. Commit to your best result. Verify it now —
  derive/run/read — then finish. Do not open new threads."
- **decide** — *decision-forcing.* "This is your last cycle; you cannot iterate further. Land the best
  outcome you have. If the objective is met, verify it and call `goal_done`. If it's partial, finish
  the solid parts, **explicitly flag what's unverified or incomplete**, and conclude as `partial`. If
  it genuinely can't be done, conclude as `abandoned` with the specific reason. Decide now."

### 6.3 Model stop affordance (FR6)

Two viable shapes; pick in §14:

- **(a) Extend `goal_done`** with optional `outcome` + `summary`. Default `outcome = done` keeps
  today's verified behavior. `partial`/`abandoned` are accepted **only** in `commit`/`decide` bands;
  they bypass the `done_when`/plan gate but record the summary and any remaining plan steps into the
  snapshot, then stop the loop (terminal status `done` with a `partial`/`abandoned` note, or a new
  `concluded` status).
- **(b) Sibling tool `goal_conclude(outcome, summary)`** honored only in the cold bands; `goal_done`
  stays strict. Cleaner separation, one more tool surface.

Either way this is the mechanism that satisfies "push the model to *decide* on stopping" rather than
cutting it off — the model is given a verb to end the work on its own stated terms.

### 6.4 Channel B — sampling-temperature annealing (optional)

A `before_provider_request` handler, active only while an annealing goal is `active`:

```
samplingTemp = lo + (hi - lo) * T          // T from §6.1, clamp to [lo, hi]
if (payload looks like the expected chat-completions body) payload.temperature = samplingTemp
return payload
```

Guardrails: shape-check the payload and **fail open** (leave it untouched if it isn't the expected
shape); never run when no annealing goal is active; clamp to `[lo, hi]` (defaults derived from the
project's temperature findings, e.g. `hi ≈ 1.0`, `lo ≈ 0.3`). This is the only literal-annealing
piece and rides an untyped seam, so it ships behind a flag (§8) and last (§12).

**Spike acceptance criterion.** The enabler is confirmed only at the *return path* (the returned
payload replaces the request body). The spike must verify the *field*: that a mutated
`payload.temperature` is honored **end-to-end on the wire** — i.e. `before_provider_request` runs
*after* the provider has merged its own `temperature` option into the body, so a downstream merge
cannot clobber the mutation. Acceptance = "the model samples at the annealed temperature," not "the
returned payload reaches transport."

### 6.5 Terminal flow (FR5/FR7)

```
agent_end (active goal, not aborted/interjected):
  if autonomous and done_when passes        -> status=done; stop
  else if model concluded (FR6)             -> status set by outcome; stop
  else if cycle < maxCycles                 -> cycle++; sendUserMessage(buildContinue@band)
  else                                      -> status=blocked (only here; after a decide turn),
                                               blockedReason carries last stated position
```

The decisive ramp is emergent: as `cycle → maxCycles` the band reaches `decide` *before* the hard
stop, so the model always gets at least one explicit "decide now" turn before `blocked`.

### 6.6 Where the annealed verification register lives (the coupling decision)

`grounding` contributes two texts: the byte-stable `MINDSET`/`ANCHOR` in the system prefix, and the
act-now `CHECK` folded into the trailing turn once per turn at turn-start
(`grounding/index.ts:123-140`). `MINDSET` already carries an **always-on, every-call answer-time**
establish-or-flag rule (`grounding/index.ts:28-29`: *"state nothing as fact you did not establish this
session, or mark it 'unverified'"*). **That clause is the verification *bar*, and it is fixed.**

So annealing does **not** lower the bar — `MINDSET` re-asserts it on every call. What we anneal is the
**emphasis and effort-triage** of the act-now push:

- **explore (hot)** — *maximal rigor.* "Establish each claim before you proceed — derive / run / read.
  Be skeptical of memory; verify broadly while you have the budget."
- **consolidate (warm)** — *focused.* "Verify the claims your current direction actually depends on."
- **commit / decide (cold)** — *triage, leaning on the standing rule.* "Spend remaining verification
  on what the decision hinges on; for everything else, apply MINDSET's standing rule — state it with
  an explicit 'unverified' flag — and move to your decision. Don't keep re-proving what you've
  established." (This *uses* MINDSET's always-on flag-rule; it does not relax it.)

The cold register counters grounding's own failure mode — over-verification / analysis-paralysis —
and *reinforces* `goal`'s `decide` band.

**Where does this register live? pi forces the choice.** Each extension is loaded with its own jiti
instance and `moduleCache:false` (`loader.ts:331-340`), so **a shared `anneal` module does not share
live state** — `goal` cannot hand `grounding` the current band in memory. The schedule *functions*
(`temperature()`/`bandFor()`, pure) can still be shared code; the *position* (`cycle`) cannot be shared
by import. That leaves two real options:

- **(a) `goal` owns the register — recommended.** Enrich `buildContinue`'s bands (§6.2) with the
  verification-triage language above; leave `grounding` **completely unchanged**. The two tail
  reminders co-fold into one turn anyway, so the model sees `CHECK` (fixed floor-enforcer) +
  `buildContinue` (annealed, now carrying the verification register) as one coherent block. This
  inverts the dependency the right way — the *optional* extension (`goal`) reaches in; the
  *foundational* one (`grounding`) is untouched — and needs **zero cross-extension coupling**.
- **(b) `grounding` cools `CHECK`'s own text.** To literally anneal `CHECK`, `grounding` must learn the
  band by reading `goal`'s published position (`goal-<sessionId>.json`, or a session entry), failing
  open to today's `CHECK` when absent. This is the only way to cool `CHECK`'s own bytes, but it makes
  the always-on extension depend on the optional one's on-disk format and adds a write→read ordering
  assumption (`goal` must `persist()` the new cycle in `agent_end` before the next turn's `context`
  hooks fire). Note this is *backwards* from the `goal`→`plan` precedent (there the optional reaches
  down; here the foundational reaches up).

**Recommendation: ship (a).** It satisfies the intent — verification pressure cooling from "prove
everything" to "prove what matters, flag the rest, decide" — without coupling `grounding` to `goal` at
all. `CHECK` and `MINDSET` both stay fixed (the floor); the cooling is entirely `goal`'s, active only
during a loop. Choose **(b)** only if cooling `CHECK`'s literal text is a hard requirement (Q6). Either
way, **with no active goal nothing changes** — verification annealing is purely additive (FR12).

## 7. Interactions & dependencies

- **grounding.** Under the recommended design (§6.6a) `grounding` is **unchanged**: `MINDSET` (prefix)
  and `CHECK` (tail) both stay the fixed floor — `MINDSET`'s always-on rule is the verification bar.
  The cooling verification register rides in `goal`'s `buildContinue`, which co-folds with `CHECK`
  into one user turn (each re-engagement is genuine text → `isTurnStart` true,
  `grounding/index.ts:110-118`), so floor + cooling push reach the model as one block. The cold
  register counters over-verification and reinforces `goal`'s `decide` band; FR8 keeps it from ever
  melting the establish-or-flag floor. (Option §6.6b would instead have `grounding` read `goal`'s
  published band to cool `CHECK`'s own text — available, not recommended.)
- **plan.** `goal_done` verifies plan steps (`readPlanRemaining`, `goal/index.ts:71-80`). The concede
  path (FR6) must *record* unfinished steps in the summary rather than silently pass the gate.
- **provider payload (Channel B).** The request body is shared by all turns; the handler must be
  strictly gated to an active annealing goal and must not persist changes beyond the turn.
- **autonomous vs self-judged mode.** The coaching bands (Channel A, both injectors) and Channel B
  apply to **both** loop modes. The decision-forcing terminal (FR5) and concede affordance (FR6) are
  primarily a *self-judged* construct: there the model's judgment is the stop signal, so forcing it to
  decide is the whole point. In **autonomous** mode (`isAutonomous` — `done_when` set,
  `goal/index.ts:48-50`) `done_when` remains the authoritative `done`: FR6's `done` runs the full gate
  (which includes `done_when`), so at budget end with `done_when` still failing the model cannot
  declare victory — its only conclusions are `partial` / `abandoned`. I.e. the autonomous concede *is*
  the budget-exhaustion abandon path, now carrying a model-stated reason instead of a bare "exhausted."
  Whether to cool the nudge at all when a machine check governs is an open question (§14).

## 8. Configuration & defaults

All defaults sane; overridable. v1 keeps the surface minimal.

| Knob | Default | Where | Notes |
|---|---|---|---|
| goal annealing on/off | on (Channel A) | env `PI_GOAL_ANNEAL` / `goal_set` | Channel A is safe to default on |
| `CHECK` annealing on/off | on | env `PI_GROUND_ANNEAL` | shares the schedule; **no-op without an active goal** |
| `commitFraction` | 0.25 | cfg | reserved commit tail = `ceil(maxCycles * f)` |
| `exploreFraction` | 0.5 | cfg | explore band upper bound on `p` |
| schedule shape | geometric | cfg | `linear` alternative for testing |
| Channel B on/off | **off** | env `PI_GOAL_TEMP_ANNEAL` | untyped seam → opt-in |
| temp range `[lo, hi]` | `[0.3, 1.0]` | env | clamp for Channel B |

Per-goal overrides can ride on `goal_set` (e.g. an optional `anneal` object) if needed; not required
for v1.

## 9. Risks, edge cases & mitigations

| Risk | Mitigation |
|---|---|
| Cold band pushes premature commitment to a *wrong* answer | Honesty floor stays dominant (FR8); `decide` requires verified-or-flagged; cold band only reached near budget end |
| **Annealing `MINDSET` would weaken the honesty floor at the deadline (and churn the prefix cache)** | **Rejected (N4/FR11): floor stays byte-stable — primary reason honesty (don't relax verify while pushing "commit"), cache a bounded secondary; only the dynamic push anneals** |
| **Over-verification / analysis-paralysis (model keeps re-checking, never commits)** | **Cold verification register triages to decision-critical verification + explicit flags (§6.6); reinforces `goal`'s `decide` band** |
| **Cold register melts the honesty floor (model stops flagging, asserts from memory)** | **`MINDSET`'s always-on rule is the fixed bar (FR8/FR11); cold band changes *what to spend effort on*, never *whether to flag*** |
| **Cross-extension coupling / dependency direction** | **pi shares no live module state (`loader.ts:331-340`); recommended §6.6a has `goal` own the register so `grounding` is untouched — zero coupling. Option (b) reads `goal`'s file and fails open to baseline (FR12)** |
| Channel B clobbers user's base temperature / leaks into normal turns | Strict gate on active annealing goal; clamp; fail-open shape guard; opt-in flag |
| Tiny budgets (`maxCycles` 1–3) collapse the schedule | Reserved-tail math (FR4): always ≥1 explore (when budget>1) and ≥1 decide; `maxCycles=1` ⇒ pure decide |
| Model ignores "last cycle, decide" and still doesn't conclude | Hard `blocked` backstop retained (FR7); durable, not silent |
| Concede path becomes an early-exit hatch | Gate concession to `commit`/`decide` bands only; `done` keeps full verification |
| Untyped provider payload shape drifts across pi/provider versions | Shape-check + fail-open; covered by Channel B being opt-in and shipped last |

## 10. Testing & validation

- **Unit — shared `anneal` module:** `temperature()` monotonic; band boundaries; reserved-tail math
  for `maxCycles ∈ {1,2,3,20}`. One test suite, consumed by both extensions.
- **Unit — `goal` (`goal/test.mjs`):** `buildContinue` emits the correct phase text per band; terminal
  fires a `decide` turn before `blocked`; concede path gated by band.
- **Unit — verification register:** under §6.6a, assert `buildContinue`'s bands carry the verification
  triage and `grounding` is untouched (`CHECK`/`MINDSET` unchanged). Under §6.6b, assert `CHECK` band
  selection matches the shared schedule, is byte-identical to today with no active goal (FR12), and
  fails open when the goal state file is missing/unreadable.
- **Integration (manual):** run `/goal` on a real task; observe the tone ramp across cycles in *both*
  `buildContinue` and `CHECK`, the verification narrowing from broad → decision-critical, the graceful
  decision-forcing finish, and (if Channel B on) the sampling-temp ramp. For image tasks use the
  pelican-on-a-bike case — remember the model must `read` the rendered PNG to judge it; the loop's
  honor-system check can't catch a hallucinated visual verdict.
- **Channel B:** assert the handler mutates `temperature` only for the expected payload shape and is a
  no-op with no active annealing goal.

## 11. Observability

- Extend `renderGoal` (`goal/index.ts:95-108`) and the durable snapshot to show the current band and
  `T` (and sampling temp if Channel B on), so a transcript/`goal-status.md` shows the anneal curve.
- Record the final `outcome` + `summary` in the snapshot so a terminated loop reads as a *decision*,
  not a cutoff.

## 12. Rollout / phasing

- **Phase 1 (Channel A):** the shared `anneal` schedule + banded `buildContinue` carrying the coaching
  push **and** the verification register (§6.6a) + decision-forcing terminal + model stop affordance.
  Pure-prompt, fully reliable, unit-tested; `grounding` stays unchanged. This delivers the entire
  user-visible behavior — "good teacher that narrows verification and makes you land it".
- **Phase 2 (Channel B):** sampling-temperature annealing behind `PI_GOAL_TEMP_ANNEAL`; A/B against
  Phase 1 to measure whether the literal temp ramp improves convergence quality.

## 13. Success metrics

- Loops that hit the budget end with an explicit model-stated `outcome` (`done`/`partial`/`abandoned`)
  instead of a silent "budget exhausted", in the large majority of runs.
- Fewer wasted late cycles (model thrashing / re-verifying after the useful work is done).
- Observable explore→commit arc in transcripts, in both the coaching push and the verification ask.
- (Phase 2) measured convergence-quality delta from the sampling-temp ramp vs Phase 1.

## 14. Open questions

1. **Stop affordance shape:** extend `goal_done` (§6.3a) vs. a sibling `goal_conclude` (§6.3b)?
2. **Band granularity:** four discrete bands (proposed) vs. a continuous `T`-driven push? Bands are
   simpler to test and read more like a teacher; continuous is "purer."
3. **Channel B range & default:** confirm `[lo, hi]` against fresh measurements for this model; keep
   it off by default for v1?
4. **Terminal status vocabulary:** reuse `blocked` for `abandoned`, or add a `concluded` status to
   distinguish "the model decided to stop" from "the loop ran out of road"?
5. **Autonomous-mode annealing:** should the schedule cool the nudge when a machine `done_when`
   governs the stop, or stay hot (keep pushing toward the check) and treat budget end purely as the
   abandon path? (See §7, autonomous vs self-judged.)
6. **Coupling mechanism (where the verification register lives).** pi loads each extension with its
   own jiti instance + `moduleCache:false` (`loader.ts:331-340`), so a shared `anneal` module does
   **not** share live state — the in-memory-singleton option is off the table. Options: **(a,
   recommended)** `goal` owns the register in `buildContinue`, `grounding` untouched, zero coupling;
   **(b)** `grounding` reads `goal`'s published position to cool `CHECK`'s own text — literally
   anneals `CHECK` but makes the foundational extension depend on the optional one's format + a
   write/read ordering. Pick (a) unless cooling `CHECK`'s own bytes is a hard requirement.
7. **Anneal `MINDSET` too?** Proposed **no** — primarily because cooling the *standing* verify-rule as
   the deadline nears, while also pushing "commit," is the dangerous combination (cache is a minor
   secondary). Confirm we only ever cool the *act-now* register, never the standing principle.
8. **`CHECK` outside a goal loop:** proposed to stay baseline (no budget/horizon ⇒ no schedule).
   Confirm additive-only is the desired contract.

---

## Appendix A: glossary

- **Cycle / budget** — `state.cycle` and `state.maxCycles`; the loop's iteration counter and cap.
- **Temperature `T`** — the abstract annealing control in `[0,1]` (this PRD), distinct from the
  model's *sampling* temperature (Channel B), which is *driven by* `T`.
- **Band / phase** — `explore` / `consolidate` / `commit` / `decide`; the coaching register selected
  by `T`.
- **Floor** — the byte-stable, cached, **un-annealed** texts (`MINDSET`, `ANCHOR`, `goal`'s
  north-star) carrying the invariant *"establish it, or flag it unverified."* Never melts.
- **Channel A / B** — prompt-pressure annealing (the dynamic push: `goal`'s `buildContinue`, carrying
  the annealed coaching **and** verification register; §6.6a) vs. sampling-temperature annealing.

## Appendix B: code references

| Ref | Location |
|---|---|
| `buildContinue` (goal tail chokepoint) | `pi-extensions/goal/index.ts:115-124` |
| `agent_end` driver + hard `blocked` at cap | `pi-extensions/goal/index.ts:388-418`, `409-413` |
| `goal_done` verification gate | `pi-extensions/goal/index.ts:296-324` |
| `GoalState` | `pi-extensions/goal/index.ts:30-38` |
| `checkText` / `DEFAULT_CHECK` (MINDSET-grounded) | `pi-extensions/goal/index.ts:55-60` |
| `renderGoal` (status surface) | `pi-extensions/goal/index.ts:95-108` |
| coupling precedent: `goal` reads `plan`'s state | `pi-extensions/goal/index.ts:71-80` |
| grounding `MINDSET` + `ANCHOR` (cached prefix) | `pi-extensions/grounding/index.ts:19-44`, `123-125` |
| grounding `CHECK` (annealed tail chokepoint) | `pi-extensions/grounding/index.ts:49-54`, `132-140` |
| grounding `isTurnStart` / `foldReminder` | `pi-extensions/grounding/index.ts:110-118`, `78-91` |
| `before_provider_request` payload is mutable | `packages/coding-agent/src/core/sdk.ts:330-335`, `.../extensions/runner.ts:946-978` |
| payload typed `unknown` (untyped seam) | `packages/coding-agent/src/core/extensions/types.ts:644-647`, `:1018` |
| `appendEntry` (in-session state option) | `packages/coding-agent/src/core/extensions/types.ts:1229-1230` |
