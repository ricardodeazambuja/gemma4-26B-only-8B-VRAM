"""Problem module for selfcheck.py: count distinct substrings.

verify() checks the model's function against a brute-force set() oracle on ~500
random + hand-picked strings, then a 100000-char scale/timing check that an
O(n^2) solution can't pass. Returns a concrete smallest-failing counterexample as
feedback so the model has something specific to fix.
"""
import random, time

TASK = (
    "Write a Python function count_distinct_substrings(s) that returns the number of "
    "distinct non-empty substrings of the string s. It must run in O(n) or O(n log n) "
    "time so it handles |s| up to 100000 (e.g. a suffix automaton or suffix array + LCP). "
    "Provide the complete, self-contained function in a single ```python code block."
)
ENTRY = "count_distinct_substrings"


def _brute(s):
    return len({s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)})


def verify(ns):
    f = ns[ENTRY]
    random.seed(20240619)
    cases = ["", "a", "aa", "aab", "abab", "aaaa", "banana", "mississippi", "abracadabra"]
    cases += ["".join(random.choice("abc") for _ in range(random.randint(1, 14))) for _ in range(500)]

    for s in cases:
        try:
            got = f(s)
        except Exception as e:
            return (False, f"On input s={s!r} your function raised {type(e).__name__}: {e}.")
        exp = _brute(s)
        if got != exp:
            return (False, f"Your function is INCORRECT. On input s={s!r} it returned {got}, "
                           f"but the correct number of distinct non-empty substrings is {exp}.")

    # scale + timing: an O(n^2) set-of-substrings solution OOMs/times out here.
    big = "".join(random.choice("abcde") for _ in range(100_000))
    t0 = time.time()
    r = f(big)
    dt = time.time() - t0
    if dt > 5.0:
        return (False, f"Correct on all {len(cases)} small cases, but TOO SLOW: {dt:.1f}s on a "
                       f"100000-char string (limit 5s). Switch to an O(n)/O(n log n) algorithm.")
    return (True, f"All {len(cases)} correctness cases passed; 100000-char case returned {r} in {dt:.2f}s.")
