#!/usr/bin/env python3
"""Greedy decode speed of the server on scripts/run-flashmap.sh, with and without FlashMap.

Sends a fixed set of prompts (math, code, prose) to /v1/chat/completions with thinking
off and temperature 0, and prints per-prompt decode t/s, draft acceptance and the
generated text's sha1 (identical hashes across DRAFT=0/1 runs = lossless).

  python scripts/flashmap-bench.py [--url http://127.0.0.1:8080] [--max-tokens 512] [--json out.json]
"""
import argparse
import hashlib
import json
import urllib.request

PROMPTS = {
    "math": "How many positive whole-number divisors does 196 have? Explain step by step.",
    "gsm": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
           "in May. How many clips did Natalia sell altogether in April and May? Reason step by step.",
    "code": "Write a Python function that returns the n-th Fibonacci number using memoization, "
            "with a docstring and three doctests.",
    "code2": "Implement binary search over a sorted list in C, with a small main() that tests it.",
    "prose": "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a hash table works and what happens on a collision, for a beginner.",
}


def run(url, prompt, max_tokens):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--json")
    args = ap.parse_args()

    run(args.url, "Hi", 8)  # warmup
    rows = []
    print(f"{'prompt':8} {'tokens':>6} {'t/s':>7} {'drafted':>7} {'accepted':>8} {'acc%':>5} {'tok/step':>8}  sha1")
    for name, prompt in PROMPTS.items():
        out = run(args.url, prompt, args.max_tokens)
        t = out.get("timings", {})
        text = out["choices"][0]["message"]["content"]
        n = t.get("predicted_n", 0)
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        # verify steps = generated tokens - accepted drafts (each step emits one target token)
        steps = max(n - da, 1)
        row = {"prompt": name, "tokens": n, "tps": t.get("predicted_per_second", 0.0), "draft_n": dn,
               "draft_accepted": da, "tok_per_step": n / steps,
               "sha1": hashlib.sha1(text.encode()).hexdigest()[:10], "text": text}
        rows.append(row)
        acc = 100.0 * da / dn if dn else 0.0
        print(f"{name:8} {n:6d} {row['tps']:7.1f} {dn:7d} {da:8d} {acc:5.1f} {row['tok_per_step']:8.2f}  {row['sha1']}")
    tot_n = sum(r["tokens"] for r in rows)
    tot_t = sum(r["tokens"] / r["tps"] for r in rows if r["tps"])
    print(f"{'ALL':8} {tot_n:6d} {tot_n / tot_t:7.1f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
