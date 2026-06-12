#!/usr/bin/env bash
# scripts/benchmark-mtp.sh — measure decode throughput (and MTP draft acceptance)
# for the QAT 26B-A4B MoE with vs without Multi-Token Prediction, holding every
# other knob constant. One invocation = one variant = one server load.
#
# It drives a throwaway llama-server on a dedicated port via the native
# /completion endpoint (predicted_per_second is the decode tok/s) and scrapes the
# server log for the draft acceptance rate (the /completion JSON omits it).
#
# Why a fair A/B: MTP needs a little extra VRAM for the 0.25 GB draft head + its
# KV. To compare like-for-like, run BOTH baseline and MTP at the SAME NCMOE — the
# lowest NCMOE at which the MTP variant loads. Any expert layer MTP forces off the
# GPU is a real cost of MTP and should be charged to it.
#
# Env knobs (all optional):
#   NMAX=""          empty -> baseline (no MTP); N -> --spec-type draft-mtp --spec-draft-n-max N
#   NCMOE=27 CTX=32768 KVQUANT=q8_0   model placement / context / KV quant (match run-server.sh)
#   TEMP=0           0 = greedy (acceptance ceiling, lossless => identical output both sides)
#   NPREDICT=256 REPS=3 SEED=42       generation length, timed repetitions, RNG seed
#   PORT=8090        throwaway server port (kept off the real 8080)
#   LABEL=auto       result label; OUTDIR=/tmp/mtp_bench  results dir
#
# Output: a one-line summary on stdout + a TSV row appended to $OUTDIR/results.tsv.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="$REPO_ROOT/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
HEAD="$REPO_ROOT/models/gemma4-26b-a4b-qat/mtp-gemma-4-26B-A4B-it.gguf"
CUDA_BIN="$REPO_ROOT/vendor/llama.cpp/build/bin/llama-server"
CUDA_ENV="${CUDA_ENV:-llamacpp-cuda}"

NMAX="${NMAX:-}"; NCMOE="${NCMOE:-27}"; CTX="${CTX:-32768}"; KVQUANT="${KVQUANT:-q8_0}"
TEMP="${TEMP:-0}"; NPREDICT="${NPREDICT:-256}"; REPS="${REPS:-3}"; SEED="${SEED:-42}"
PORT="${PORT:-8090}"; OUTDIR="${OUTDIR:-/tmp/mtp_bench}"
mkdir -p "$OUTDIR"
PMIN="${PMIN:-}"   # empty -> llama.cpp default (0.0); N -> --spec-draft-p-min N (draft-confidence floor)
if [ -z "${LABEL:-}" ]; then
  if [ -z "$NMAX" ]; then LABEL="baseline-t${TEMP}"; else LABEL="mtp-n${NMAX}-t${TEMP}${PMIN:+-p${PMIN}}"; fi
fi
LOG="$OUTDIR/server-$LABEL.log"
PROMPT_FILE="${PROMPT_FILE:-$OUTDIR/prompt.txt}"
[ -f "$PROMPT_FILE" ] || { echo "ERROR: prompt file not found: $PROMPT_FILE"; exit 2; }

SPEC_ARGS=()
[ -n "$NMAX" ] && SPEC_ARGS=(--spec-draft-model "$HEAD" --spec-type draft-mtp --spec-draft-n-max "$NMAX")
# p_min: MTP head stops drafting once its top-token prob drops below this (common/speculative.cpp:706)
[ -n "$NMAX" ] && [ -n "$PMIN" ] && SPEC_ARGS+=(--spec-draft-p-min "$PMIN")

echo ">> [$LABEL] launching server: NCMOE=$NCMOE CTX=$CTX KV=$KVQUANT TEMP=$TEMP ${NMAX:+MTP n-max=$NMAX}${PMIN:+ p-min=$PMIN}"
setsid mamba run --no-capture-output -n "$CUDA_ENV" "$CUDA_BIN" \
  -m "$MODEL" --alias gemma-4-26b-a4b-qat \
  -ngl 99 --n-cpu-moe "$NCMOE" --no-mmap \
  -c "$CTX" -ctk "$KVQUANT" -ctv "$KVQUANT" -fa on --jinja \
  --temp "$TEMP" --top-p 0.95 --top-k 64 \
  "${SPEC_ARGS[@]}" \
  --host 127.0.0.1 --port "$PORT" > "$LOG" 2>&1 &
WRAP_PID=$!

teardown() {
  pkill -9 -f "llama-server.*--port $PORT" >/dev/null 2>&1 || true
  kill -9 -- "-$WRAP_PID" >/dev/null 2>&1 || true
  kill -9 "$WRAP_PID" >/dev/null 2>&1 || true
}
trap teardown EXIT

# Wait for the model to finish loading. llama-server returns 503 ("loading model")
# until ready, which curl --retry treats as a completed transfer; poll in Python
# instead (its time.sleep works; the shell `sleep` builtin is blocked here).
HC="$(PORT="$PORT" python3 - <<'PY'
import os,urllib.request,time
port=os.environ["PORT"]; deadline=time.time()+900
while time.time()<deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=5) as r:
            if r.status==200: print("200"); raise SystemExit
    except SystemExit: raise
    except Exception: pass
    time.sleep(2)
print("timeout")
PY
)"
if [ "$HC" != "200" ]; then
  echo ">> [$LABEL] SERVER FAILED TO BECOME READY ($HC). Last log lines:"
  tail -25 "$LOG"
  exit 1
fi
echo ">> [$LABEL] server ready; running $REPS timed request(s) of $NPREDICT tokens"

SUMMARY="$(PORT="$PORT" NPREDICT="$NPREDICT" REPS="$REPS" SEED="$SEED" TEMP="$TEMP" \
           LABEL="$LABEL" NCMOE="$NCMOE" CTX="$CTX" KVQUANT="$KVQUANT" NMAX="$NMAX" \
           PROMPT_FILE="$PROMPT_FILE" OUTDIR="$OUTDIR" python3 - <<'PY'
import json,os,statistics,urllib.request
port=os.environ["PORT"]; n=int(os.environ["NPREDICT"]); reps=int(os.environ["REPS"])
seed=int(os.environ["SEED"]); temp=float(os.environ["TEMP"]); label=os.environ["LABEL"]
prompt=open(os.environ["PROMPT_FILE"]).read().strip()
def call(npred):
    body=json.dumps({"prompt":prompt,"n_predict":npred,"temperature":temp,"seed":seed,
                     "cache_prompt":False,"top_p":0.95,"top_k":64}).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",data=body,
                               headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req,timeout=900))
call(96)  # warmup (not timed) — long enough to ramp GPU clocks off idle before timing
pps=[]
for i in range(reps):
    t=call(n)["timings"]; pps.append(t["predicted_per_second"])
    print(f"   run {i+1}/{reps}: {t['predicted_per_second']:.2f} tok/s "
          f"(predicted_n={t['predicted_n']})")
med=statistics.median(pps)
row="\t".join([label,os.environ["NCMOE"],os.environ["CTX"],os.environ["KVQUANT"],
               f"{temp}",os.environ.get("NMAX") or "-",f"{med:.2f}",
               f"{min(pps):.2f}",f"{max(pps):.2f}"])
open(os.path.join(os.environ["OUTDIR"],"results.tsv"),"a").write(row+"\n")
print(f"RESULT\t{label}\tmedian={med:.2f}\tmin={min(pps):.2f}\tmax={max(pps):.2f}")
PY
)"
echo "$SUMMARY"

# Draft acceptance lives only in the server log, not the /completion JSON.
if [ -n "$NMAX" ]; then
  echo ">> [$LABEL] draft-acceptance lines from server log:"
  grep -iE "draft|accept|n_drafted|spec" "$LOG" | tail -8 || echo "   (none found — inspect $LOG)"
fi
echo ">> [$LABEL] done."
