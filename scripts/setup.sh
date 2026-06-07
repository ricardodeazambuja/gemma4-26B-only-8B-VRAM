#!/usr/bin/env bash
#
# setup.sh — one-time setup: create the llama.cpp env, download the model, and
# (optionally) build the native CUDA backend.
#
# Pick your backend with BACKEND:
#   bash scripts/setup.sh                 # BACKEND=vulkan (default): env + model.
#                                         #   Works on any GPU/driver, no build. ~4-5 tok/s.
#   BACKEND=cuda bash scripts/setup.sh    # env + model + build llama.cpp against your
#                                         #   driver's CUDA (scripts/build-llama-cuda.sh).
#                                         #   ~5-6x faster (~25 tok/s here). Adds ~20 min.
#
# The conda env + model download are needed either way; CUDA just adds the source build.
#
# Other overrides: ENV_NAME, CUDA_BUILD (conda variant), MODEL_REPO, MODEL_FILE.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BACKEND="${BACKEND:-vulkan}"
ENV_NAME="${ENV_NAME:-llamacpp}"
# conda-forge llama.cpp build variant. cuda129 = CUDA 12.9, cuda130 = CUDA 13.0.
# Both include the Vulkan backend, which is what we actually run (see README:
# the CUDA kernels need a driver matching the build's CUDA version).
CUDA_BUILD="${CUDA_BUILD:-cuda129}"

MODEL_REPO="${MODEL_REPO:-unsloth/gemma-4-26B-A4B-it-qat-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
MODEL_DIR="$REPO_ROOT/models/gemma4-26b-a4b-qat"

command -v mamba >/dev/null 2>&1 || { echo "ERROR: 'mamba' not found. Install Miniforge (https://github.com/conda-forge/miniforge) first."; exit 1; }

# --- 1. create env ----------------------------------------------------------
if mamba env list | grep -qE "/${ENV_NAME}\$|^\s*${ENV_NAME}\s"; then
  echo ">> env '$ENV_NAME' already exists — skipping create"
else
  echo ">> creating env '$ENV_NAME' (llama.cpp ${CUDA_BUILD} + huggingface_hub) ..."
  mamba create -y -n "$ENV_NAME" -c conda-forge "llama.cpp=*=*${CUDA_BUILD}*" huggingface_hub
fi

# --- 2. download the GGUF ---------------------------------------------------
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
  echo ">> model already present: $MODEL_DIR/$MODEL_FILE"
else
  echo ">> downloading $MODEL_FILE (~14 GB) into $MODEL_DIR ..."
  if mamba run -n "$ENV_NAME" hf --help >/dev/null 2>&1; then
    mamba run --no-capture-output -n "$ENV_NAME" hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
  else
    mamba run --no-capture-output -n "$ENV_NAME" huggingface-cli download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
  fi
fi

# --- 3. sanity check: can llama.cpp see a GPU? ------------------------------
echo ">> devices visible to llama.cpp:"
mamba run -n "$ENV_NAME" llama-server --list-devices 2>/dev/null | sed 's/^/   /'

# --- 4. optional: build the native CUDA backend -----------------------------
if [ "$BACKEND" = "cuda" ]; then
  echo
  echo ">> BACKEND=cuda — building llama.cpp against your driver's CUDA (~20 min) ..."
  bash "$REPO_ROOT/scripts/build-llama-cuda.sh"
  RUN_HINT="BACKEND=cuda bash scripts/start.sh        # native CUDA backend (~5-6x faster)"
else
  RUN_HINT="bash scripts/start.sh                     # Vulkan backend (default)
     (for ~5-6x more speed later: BACKEND=cuda bash scripts/setup.sh)"
fi

cat <<EOF

>> Setup complete (backend: $BACKEND).
   Next:
     1) Register pi provider (once):  bash scripts/configure-pi.sh
     2) Start server + pi:            $RUN_HINT
EOF
