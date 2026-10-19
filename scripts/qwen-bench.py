#!/usr/bin/env python3
"""qwen-bench.py — the quick smoke benchmark used for the Qwen bake-off (docs/qwen-bakeoff.md).

Runs three requests against whatever llama-server is listening (default port 8080):
  1. speed: a short coding prompt with thinking on, max_tokens=1500
     -> generation tokens/s from the server's own `timings`
  2. tool call: one `read` tool offered -> did the model emit a well-formed call?
  3. prefill: ~15.6K tokens (scripts/start.sh + scripts/run-server.sh), thinking off
     -> prompt-processing tokens/s (what pi's first turn costs)

Usage:  python3 scripts/qwen-bench.py [--port 8080] [--skip-prefill]
"""
import argparse
import json
import os
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def post(port, body, timeout=1800):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--skip-prefill", action="store_true")
    a = ap.parse_args()

    d, wall = post(a.port, {
        "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals, with a docstring and 3 doctests."}],
        "max_tokens": 1500,
    })
    t = d.get("timings", {})
    m = d["choices"][0]["message"]
    print("== speed: gen %.1f t/s, %d tokens in %.0fs (wall %.0fs)" % (
        t.get("predicted_per_second", 0), t.get("predicted_n", 0), t.get("predicted_ms", 0) / 1000, wall))
    print("   reasoning chars: %d | content chars: %d | finish: %s" % (
        len(m.get("reasoning_content") or ""), len(m.get("content") or ""), d["choices"][0]["finish_reason"]))

    tools = [{"type": "function", "function": {
        "name": "read", "description": "Read a file from disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    d, wall = post(a.port, {
        "messages": [{"role": "user", "content": "What is in the file scripts/run-qwen.sh? Use the tool."}],
        "tools": tools, "max_tokens": 1500,
    })
    c = d["choices"][0]
    calls = [tc["function"] for tc in (c["message"].get("tool_calls") or [])]
    print("== tool call: finish=%s calls=%s (%.0fs)" % (c["finish_reason"], json.dumps(calls), wall))

    if a.skip_prefill:
        return
    text = "".join(open(os.path.join(REPO, "scripts", f)).read() for f in ("start.sh", "run-server.sh"))
    d, wall = post(a.port, {
        "messages": [{"role": "user", "content": text + "\n\nIn one sentence, what does start.sh do?"}],
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    tm = d["timings"]
    print("== prefill: %d tokens at %.0f t/s (%.1fs); wall %.1fs" % (
        tm["prompt_n"], tm["prompt_per_second"], tm["prompt_ms"] / 1000, wall))
    print("   answer:", (d["choices"][0]["message"]["content"] or "")[:200])


if __name__ == "__main__":
    main()
