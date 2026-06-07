#!/usr/bin/env bash
#
# setup.sh — one-time setup: create the llama.cpp env and download the model.
#
# Creates a conda/mamba env containing llama.cpp (with both CUDA and Vulkan
# backends compiled in) plus huggingface_hub, then downloads the Gemma 4
# 26B-A4B QAT GGUF (~14 GB) into ./models/.
#
# Override defaults via env vars, e.g.:
#   ENV_NAME=mygemma CUDA_BUILD=cuda130 bash scripts/setup.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

cat <<EOF

>> Setup complete.
   Next:
     1) Start the server:   bash scripts/run-server.sh
     2) Point pi at it:     bash scripts/configure-pi.sh   (once)
                            pi --provider llamacpp --model gemma-4-26b-a4b-qat
EOF
