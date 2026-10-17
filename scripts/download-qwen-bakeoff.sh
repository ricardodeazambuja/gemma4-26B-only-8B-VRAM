#!/usr/bin/env bash
#
# download-qwen-bakeoff.sh — fetch + sha256-verify the GGUFs for the Qwen bake-off
# (qwen_on_8gb_vram). Three candidates, each with its vision projector:
#
#   qwen36   Qwen3.6-35B-A3B          unsloth UD-IQ4_XS        ~17.7 GB  MoE, 3B active
#   qwen38   Qwen3.8-27B (dense)      ISTA-DASLab GSQ-RCO      ~10.4 GB  IQ3_XXS + MTP head
#   distill  Qwen3.8-35B-A3B-Distill  empero-ai IQ4_XS         ~19.6 GB  3.6 base, 3.8 traces
#
# Optional (not in the default set):
#   dflash   DFlash2 block-diffusion drafter for the 27B (z-lab, Q4_K_M, ~1.1 GB)
#   bonsai2  Ternary-Bonsai-2-27B PTQ1_0 (PrismML, ~5.95 GB + 0.63 GB vision) — Qwen3.8-27B
#            in ternary weights; needs the PrismML llama.cpp fork (docs/bonsai2-on-8gb.md)
#
# Reproducible by construction: every file is downloaded at a PINNED Hugging Face repo
# revision and checked against a PINNED sha256 (the exact bytes tested in
# docs/qwen-bakeoff.md). A GGUF can have the right size and a valid header and still be
# corrupt (see docs/TECHNICAL.md), so a mismatch deletes the file and exits non-zero.
# Re-running skips files that already verify. FOLLOW_MAIN=1 downloads the repos' current
# main instead and checks against the API's current LFS sha256 (newer, untested bytes).
#
# Needs the `hf` CLI (package huggingface_hub) INSIDE the conda env ENV_NAME — the
# script refuses an `hf` that merely happens to be on PATH, so results don't depend
# on what else is installed on the machine. Fix with:
#   mamba install -n llamacpp -c conda-forge huggingface_hub
#
# Usage:
#   bash scripts/download-qwen-bakeoff.sh              # all three
#   bash scripts/download-qwen-bakeoff.sh qwen38       # just one (qwen36 | qwen38 | distill | dflash | bonsai2)
#
# Overrides: ENV_NAME (env with huggingface_hub, default llamacpp), FOLLOW_MAIN=1.
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-llamacpp}"
FOLLOW_MAIN="${FOLLOW_MAIN:-0}"

# name | HF repo | pinned revision | local dir (under models/) | file:sha256 ...
MODELS=(
  "qwen36|unsloth/Qwen3.6-35B-A3B-GGUF|a483e9e6cbd595906af30beda3187c2663a1118c|qwen36-35b-a3b|Qwen3.6-35B-A3B-UD-IQ4_XS.gguf:649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3 mmproj-F16.gguf:8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f"
  "qwen38|ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF|d562806dbafae37109975e970aae91b43e73b440|qwen38-27b-gsq|Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf:63f29a2189a6b4cc31f81e093d3856ad293a5583114439f41ae1ed7af4093262 mmproj-Qwen3.8-27B-BF16.gguf:13cb7bebccbd04afc8f4090cb949ecf8937cdf7377c5799b1a0c594e7c0d3e16"
  "distill|empero-ai/Qwen3.8-35B-A3B-Distill-GGUF|b1f9d1dcc3de8aa867669b0ab919384aeeb9b8d5|qwen38-35b-a3b-distill|Qwen3.8-35B-A3B-IQ4_XS.gguf:b645af45431ef9b41f43cae51c9323b2d0ca84f23031d483d2105c643ae58d65 mmproj-Qwen3.8-35B-A3B-F16.gguf:4381cb5110074396c2c7b39221fffae0c31886aa95b674de8e103d97edf58b94"
  "bonsai2|prism-ml/Ternary-Bonsai-2-27B-gguf|b072e1d3b35a0a630cece372c2127528e0994386|bonsai2-27b|Ternary-Bonsai-2-27B-PTQ1_0.gguf:53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3 Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf:6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903"
  "dflash|z-lab/Qwen3.8-27B-DFlash2-GGUF|2d9571f8ce46e151f61c6499c99dee6079e1d610|qwen38-27b-gsq|Qwen3.8-27B-DFlash2-Q4_K_M.gguf:1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd"
)

WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(qwen36 qwen38 distill)

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found on PATH (install Miniforge)."; exit 1; }

# The hf CLI must come from ENV_NAME itself, not from some other install on PATH.
ENV_PREFIX="$(mamba run -n "$ENV_NAME" printenv CONDA_PREFIX 2>/dev/null || true)"
[ -n "$ENV_PREFIX" ] || { echo "ERROR: conda env '$ENV_NAME' not found. Create it: mamba create -y -n $ENV_NAME -c conda-forge huggingface_hub"; exit 1; }
HF_BIN="$ENV_PREFIX/bin/hf"
[ -x "$HF_BIN" ] || { echo "ERROR: no 'hf' CLI in env '$ENV_NAME' ($HF_BIN). Install: mamba install -y -n $ENV_NAME -c conda-forge huggingface_hub"; exit 1; }

# Current-main sha256 of one file (FOLLOW_MAIN=1 only), from the HF API (LFS oid).
main_sha() {
  local repo="$1" file="$2"
  curl -fsSL "https://huggingface.co/api/models/$repo/paths-info/main" \
       --data-urlencode "paths=$file" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d[0]["lfs"]["oid"] if d and d[0].get("lfs") else "")'
}

FAIL=0
for want in "${WANT[@]}"; do
  found=0
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name repo rev dir files <<<"$entry"
    [ "$name" = "$want" ] || continue
    found=1
    dest="$REPO_ROOT/models/$dir"
    mkdir -p "$dest"
    [ "$FOLLOW_MAIN" = 1 ] && rev=main
    for spec in $files; do
      f="${spec%%:*}"
      if [ "$FOLLOW_MAIN" = 1 ]; then
        want_sha="$(main_sha "$repo" "$f")"
        [ -n "$want_sha" ] || { echo "ERROR: no sha256 from the HF API for $repo/$f"; FAIL=1; continue; }
      else
        want_sha="${spec#*:}"
      fi
      if [ -f "$dest/$f" ]; then
        echo ">> $name: $f exists, verifying ..."
        if [ "$(sha256sum "$dest/$f" | cut -d' ' -f1)" = "$want_sha" ]; then
          echo "   OK (already verified)"; continue
        fi
        echo "   sha256 mismatch on existing file, re-downloading"
        rm -f "$dest/$f"
      fi
      echo ">> $name: downloading $repo@${rev:0:7}/$f -> models/$dir/"
      "$HF_BIN" download "$repo" "$f" --revision "$rev" --local-dir "$dest"
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
  [ "$found" = 1 ] || { echo "ERROR: unknown model '$want' (use qwen36 | qwen38 | distill | dflash | bonsai2)"; FAIL=1; }
done

exit "$FAIL"
