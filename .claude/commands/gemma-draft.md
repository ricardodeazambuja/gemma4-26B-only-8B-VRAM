---
description: Ask the local Gemma 4 draft tier for a cheap first pass (text draft, or OCR/describe an image) before spending big-model effort
argument-hint: <task — include image paths to read them locally>
allowed-tools: Bash(.claude/spec/draft.sh:*)
---

The user deliberately invoked the local Gemma draft tier for this task:

> $ARGUMENTS

Gemma's draft (or status) follows:

!`.claude/spec/draft.sh "$ARGUMENTS"`

Treat the output above as a DRAFT from the cheap local tier, not authoritative: verify it,
keep what is right, fix what is wrong, then answer the task. If it is an image transcript,
trust it as the image's content unless it contradicts other evidence (a raw Read is the
escape hatch). If the server was down or no draft was produced, just do the task normally.
