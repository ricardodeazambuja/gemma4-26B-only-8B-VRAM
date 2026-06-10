---
name: gemma-draft
description: Deliberately ask the local Gemma 4 draft tier first (cheap second opinion, OCR/describe an image, or a fast first pass) before spending Opus effort. Use when the user says "ask Gemma", "draft with the local model", "let the small model try first", "OCR this", or when a cheap first attempt would save tokens.
---

# Gemma draft tier (deliberate)

The hooks in `.claude/spec/` run Gemma *automatically*. This skill is the **manual** lever:
invoke the local model on purpose for a cheap first pass, a second opinion, or to read an image.

## How

The client is `.claude/spec/gemma.sh` (talks to llama-server on :8080). It exits non-zero and
prints nothing useful if the server is down — in that case, just proceed normally.

- Text draft / second opinion:
  ```bash
  .claude/spec/gemma.sh --system "You are a terse drafter." --max 256 "<the task>"
  ```
- OCR / describe an image WITHOUT spending Opus image tokens (preferred for any image):
  ```bash
  .claude/spec/gemma.sh --image path/to/pic.png "Describe and OCR this image."
  ```

## When to use it
- The user explicitly asks for the local model ("ask Gemma", "small model first").
- An image needs reading — always prefer `--image` over Reading the image yourself (G6: saves tokens).
- A quick draft/outline would cut Opus work — get Gemma's pass, then **verify and improve** it.

## How to treat the output
Gemma is the fast/cheap **draft tier**, not authoritative. Treat its output as a draft to
**verify**: keep what's right, fix what's wrong, discard if off. Never apply it blindly.
