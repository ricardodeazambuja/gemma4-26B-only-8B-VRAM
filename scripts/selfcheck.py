#!/usr/bin/env python3
"""selfcheck.py — let VibeThinker test (and fix) its own code via an external loop.

VibeThinker is a reasoner, not an agent: it will NOT call pi/llama.cpp tools, even
with tool_choice="required" (it just thinks and boxes an answer). So we close the
loop OUTSIDE the model — this harness is the "tool":

    1. ask the model for a solution
    2. extract the code, RUN it against a verifier (oracle / tests / smoke run)
    3. if it fails, feed the concrete failure back and ask for a fix
    4. repeat until it passes or rounds run out

This plays to the model's strength (writing/repairing code from a concrete failing
case) instead of its weakness (deciding to call a tool).

A problem file (passed as the argument) defines:
    TASK : str                       # the prompt sent to the model
    ENTRY: str                       # function the solution must define (for a clear error)
    def verify(ns) -> (ok: bool, feedback: str)   # ns = the exec'd namespace

Usage:
    python scripts/selfcheck.py scripts/selfcheck_problems/distinct_substrings.py --rounds 4

Env: VT_URL (default http://127.0.0.1:8080/v1/chat/completions), VT_MODEL (vibethinker-1.5b),
     VT_MAXTOK (16000), VT_MEM_MB (2048 — RAM cap for the candidate subprocess).
"""
import argparse, json, os, re, subprocess, sys, urllib.request, importlib.util

SERVER  = os.environ.get("VT_URL",   "http://127.0.0.1:8080/v1/chat/completions")
MODEL   = os.environ.get("VT_MODEL", "vibethinker-1.5b")
MAXTOK  = int(os.environ.get("VT_MAXTOK", "32000"))   # this model reasons a LOT before it
MEM_MB  = int(os.environ.get("VT_MEM_MB", "2048"))

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def load_problem(path):
    spec = importlib.util.spec_from_file_location("problem", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_code(text):
    """All ```python blocks AFTER </think> joined (the model may split a class and
    its functions across blocks). Fall back to the last block if the join doesn't
    even compile (e.g. a stale-then-corrected pair)."""
    answer = text.split("</think>")[-1] if "</think>" in text else text
    code = _join_blocks(answer)
    if code is None and answer is not text:
        code = _join_blocks(text)   # truncated mid-think / never closed </think>: scan it all
    return code


def _join_blocks(segment):
    blocks = [b.strip() for b in CODE_RE.findall(segment)]
    if not blocks:
        return None
    joined = "\n\n".join(blocks)
    try:
        compile(joined, "<cand>", "exec")
        return joined
    except SyntaxError:
        return blocks[-1]


def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": MAXTOK,
                       "temperature": 0.6, "top_p": 0.95}).encode()
    req = urllib.request.Request(SERVER, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    ch = d["choices"][0]
    return (ch["message"].get("content") or "", ch.get("finish_reason"),
            d.get("usage", {}).get("completion_tokens"))


# --- candidate runner (own process: rlimit + timeout so a runaway can't kill the box) ---
def run_candidate(problem_path, code, timeout=30):
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--run", problem_path],
            input=code, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (False, f"Your code exceeded the {timeout}s time limit (too slow or an infinite loop). "
                       "Use a more efficient algorithm.")
    out = (proc.stdout or "").strip().splitlines()
    verdict = next((l for l in reversed(out) if l.startswith("__VERDICT__")), None)
    if verdict is None:
        # killed by the RAM cap / segfault / crashed before printing
        err = (proc.stderr or "").strip()[-300:]
        return (False, "Your code was killed before finishing — most likely it blew the memory limit "
                       f"(not O(n) space) or crashed. Tail of stderr: {err!r}")
    payload = json.loads(verdict[len("__VERDICT__"):])
    return (payload["ok"], payload["feedback"])


def _run_mode(problem_path):
    """Hidden mode: exec stdin code, run verify(), print a verdict line. Memory-capped."""
    try:
        import resource
        cap = MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass
    code = sys.stdin.read()
    problem = load_problem(problem_path)
    ns = {}
    try:
        exec(compile(code, "<candidate>", "exec"), ns)
    except Exception as e:
        print("__VERDICT__" + json.dumps({"ok": False,
              "feedback": f"Your code raised while defining it: {type(e).__name__}: {e}"}))
        return
    if getattr(problem, "ENTRY", None) and not callable(ns.get(problem.ENTRY)):
        print("__VERDICT__" + json.dumps({"ok": False,
              "feedback": f"You did not define a function named {problem.ENTRY}(...)."}))
        return
    try:
        ok, fb = problem.verify(ns)
    except Exception as e:
        ok, fb = False, f"The verifier hit an unexpected error running your code: {type(e).__name__}: {e}"
    print("__VERDICT__" + json.dumps({"ok": bool(ok), "feedback": fb}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    if args.run:
        _run_mode(args.problem)
        return

    problem = load_problem(args.problem)
    print(f">> model={MODEL}  rounds={args.rounds}  mem_cap={MEM_MB}MB\n>> TASK: {problem.TASK.strip()[:200]}...\n")
    messages = [{"role": "user", "content": problem.TASK}]

    for rnd in range(1, args.rounds + 1):
        print(f"===== round {rnd} =====")
        reply, finish, ntok = chat(messages)
        print(f"  generated {ntok} tokens (finish_reason={finish})")
        if finish == "length":
            print("  ⚠ hit the token cap mid-reasoning — raise VT_MAXTOK / server CTX if this recurs.")
        code = extract_code(reply)
        if not code:
            print("  no ```python code block found in the reply; nudging.")
            print(f"  reply tail: ...{reply[-160:].strip()!r}")
            messages += [{"role": "assistant", "content": reply[-1500:]},
                         {"role": "user", "content": "I could not find a Python code block. "
                          "Reply with the complete function in a single ```python block."}]
            continue
        ok, fb = run_candidate(args.problem, code)
        print(f"  verdict: {'PASS ✅' if ok else 'FAIL ❌'}")
        print(f"  feedback: {fb}\n")
        if ok:
            print("===== FINAL CODE (verified) =====")
            print(code)
            print(f"\n>> converged in {rnd} round(s).")
            return
        answer = reply.split("</think>")[-1].strip() if "</think>" in reply else reply
        messages += [
            {"role": "assistant", "content": answer[-4000:]},
            {"role": "user", "content": "I ran your code against a reference checker. " + fb +
             " Provide a corrected, complete function in a single ```python block."},
        ]

    print(f">> did NOT converge within {args.rounds} rounds (the last verdict above is where it stands).")


if __name__ == "__main__":
    main()
