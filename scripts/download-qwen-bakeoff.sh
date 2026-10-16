#!/usr/bin/env bash
#
# download-qwen-bakeoff.sh — fetch + sha256-verify the GGUFs for the Qwen bake-off
# (feat/test-qwen-bakeoff). Three candidates, each with its vision projector:
#
#   qwen36   Qwen3.6-35B-A3B          unsloth UD-IQ4_XS        ~17.7 GB  MoE, 3B active
#   qwen38   Qwen3.8-27B (dense)      ISTA-DASLab GSQ-RCO      ~10.4 GB  IQ3_XXS + MTP head
#   distill  Qwen3.8-35B-A3B-Distill  empero-ai IQ4_XS         ~19.6 GB  3.6 base, 3.8 traces
#
# Each file is checked against its Hugging Face LFS sha256 (the API's lfs.oid).
# A GGUF can have the right size and a valid header and still be corrupt
# (see docs/TECHNICAL.md), so a mismatch deletes the file and exits non-zero.
# Re-running skips files that already verify.
#
# Usage:
#   bash scripts/download-qwen-bakeoff.sh              # all three
#   bash scripts/download-qwen-bakeoff.sh qwen38       # just one (qwen36 | qwen38 | distill)
#
# Overrides: ENV_NAME (env with the `hf` CLI, default llamacpp).
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-llamacpp}"

# name | HF repo | local dir (under models/) | files (space-separated)
MODELS=(
  "qwen36|unsloth/Qwen3.6-35B-A3B-GGUF|qwen36-35b-a3b|Qwen3.6-35B-A3B-UD-IQ4_XS.gguf mmproj-F16.gguf"
  "qwen38|ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF|qwen38-27b-gsq|Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf mmproj-Qwen3.8-27B-BF16.gguf"
  "distill|empero-ai/Qwen3.8-35B-A3B-Distill-GGUF|qwen38-35b-a3b-distill|Qwen3.8-35B-A3B-IQ4_XS.gguf mmproj-Qwen3.8-35B-A3B-F16.gguf"
)

WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(qwen36 qwen38 distill)

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH."; exit 1; }

# Expected sha256 of one file in a repo, from the HF API (LFS oid).
expected_sha() {
  local repo="$1" file="$2"
  curl -fsSL "https://huggingface.co/api/models/$repo/paths-info/main" \
       --data-urlencode "paths=$file" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[0]["lfs"]["oid"] if d and d[0].get("lfs") else "")'
}

FAIL=0
for want in "${WANT[@]}"; do
  found=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name repo dir files <<<"$entry"
    [ "$name" = "$want" ] || continue
    found=1
    dest="$REPO_ROOT/models/$dir"
    mkdir -p "$dest"
    for f in $files; do
      want_sha="$(expected_sha "$repo" "$f")"
      if [ -z "$want_sha" ]; then
        echo "ERROR: no sha256 from the HF API for $repo/$f"; FAIL=1; continue
      fi
      if [ -f "$dest/$f" ]; then
        echo ">> $name: $f exists, verifying ..."
        if [ "$(sha256sum "$dest/$f" | cut -d' ' -f1)" = "$want_sha" ]; then
          echo "   OK (already verified)"; continue
        fi
        echo "   sha256 mismatch on existing file, re-downloading"
        rm -f "$dest/$f"
      fi
      echo ">> $name: downloading $repo/$f -> models/$dir/"
      mamba run --no-capture-output -n "$ENV_NAME" hf download "$repo" "$f" --local-dir "$dest"
      got_sha="$(sha256sum "$dest/$f" | cut -d' ' -f1)"
      if [ "$got_sha" = "$want_sha" ]; then
        echo "   sha256 OK"
      else
        echo "ERROR: sha256 mismatch for $f"
        echo "   expected $want_sha"
        echo "   got      $got_sha"
        rm -f "$dest/$f"
        FAIL=1
      fi
    done
  done
  [ "$found" = 1 ] || { echo "ERROR: unknown model '$want' (use qwen36 | qwen38 | distill)"; FAIL=1; }
done

exit "$FAIL"
