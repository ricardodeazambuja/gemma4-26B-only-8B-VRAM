#!/usr/bin/env python3
"""Offline regression test for vibethinker-toolshim.py recovery.

Runs the shim's recover_tool_call() against REAL VibeThinker-3B outputs captured
from live pi sessions (vibethinker-toolshim-fixtures.json) plus a few synthetic
format checks. No server needed.

    python3 scripts/test-toolshim.py

Each fixture records what the model actually emitted and which tool a correct
recovery must yield. `xfail` fixtures are documented limitations (e.g. an `edit`
call the model wrote as malformed JSON) — recovery is allowed to return None.
"""
import importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ts", os.path.join(HERE, "vibethinker-toolshim.py"))
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

# Core pi tool schemas (name / properties / required) — what the shim sees in a request.
SPECS = [
    {"name": "bash",  "props": {"command", "timeout"},        "required": {"command"}},
    {"name": "write", "props": {"path", "content"},           "required": {"path", "content"}},
    {"name": "edit",  "props": {"path", "edits"},             "required": {"path", "edits"}},
    {"name": "read",  "props": {"path", "offset", "limit"},   "required": {"path"}},
    {"name": "ls",    "props": {"path"},                       "required": set()},
    {"name": "find",  "props": {"pattern", "path"},           "required": {"pattern"}},
    {"name": "grep",  "props": {"pattern", "path"},           "required": {"pattern"}},
]

# Synthetic format checks (the older dialects the shim already handled).
SYNTHETIC = [
    ('{"name":"bash","arguments":{"command":"ls"}}', "bash"),          # inline {name,arguments}
    ('{"function":"bash","arguments":{"command":"ls"}}', "bash"),      # pi 'function' key
    ('<function>bash</function> {"command":"ls"}', "bash"),            # <function>NAME</function>+args
    ('result: {"command":"echo hi"} ok', "bash"),                     # bare args -> infer by shape
    ('write it: {"path":"/x","content":"y"} now', "write"),           # bare args -> write
]


def main():
    passed = failed = 0
    fixtures = json.load(open(os.path.join(HERE, "vibethinker-toolshim-fixtures.json")))
    print("== real-capture fixtures ==")
    for i, fx in enumerate(fixtures):
        rec = ts.recover_tool_call(fx["text"], SPECS)
        got = rec[0] if rec else None
        if fx.get("xfail"):
            ok = True  # documented limitation: any outcome accepted, but note it
            note = "xfail (limitation)" + (f", recovered {got}" if got else ", None")
        else:
            ok = (got == fx["expected"])
            note = f"expect {fx['expected']:5} got {got}"
        print(f"  [{i}] {fx['format']:24} {'PASS' if ok else 'FAIL'}  {note}")
        passed += ok
        failed += (not ok)
    print("== synthetic format checks ==")
    for t, exp in SYNTHETIC:
        rec = ts.recover_tool_call(t, SPECS)
        got = rec[0] if rec else None
        ok = (got == exp)
        print(f"  {'PASS' if ok else 'FAIL'}  expect {exp:5} got {got}")
        passed += ok
        failed += (not ok)
    total = passed + failed
    print(f"\n{passed}/{total} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
