#!/usr/bin/env bash
#
# build-llama-cuda.sh — build llama.cpp from source against the CUDA version your
# NVIDIA driver supports, so the CUDA backend actually runs.
#
# Why this exists: the conda-forge llama.cpp binary is compiled for a recent CUDA
# (12.9+). On an older driver (e.g. 535 = CUDA 12.2) its kernels fail to load
# ("device kernel image is invalid"), which is why the default setup uses Vulkan.
# Building locally against the driver's own CUDA version gives a working — and
# usually faster — CUDA backend.
#
# It auto-detects:
#   * the max CUDA version your driver supports (nvidia-smi)  -> toolkit to install
#   * your GPU compute capability (nvidia-smi)                -> CUDA arch to target
#
# and installs a matching CUDA toolchain into a dedicated conda env (nothing
# system-wide is touched), then clones and builds llama.cpp.
#
# Overrides (env vars):
#   CUDA_VER   CUDA toolkit version to build against   (default: driver's max)
#   CUDA_ARCH  CMAKE_CUDA_ARCHITECTURES                (default: detected, e.g. 75)
#   ENV_NAME   conda env for toolchain + runtime       (default: llamacpp-cuda)
#   LLAMA_REF  llama.cpp git ref/tag to build          (default: latest b* release)
#   SRC_DIR    clone/build location                    (default: ./vendor/llama.cpp)
#   MODEL      gguf used for the post-build smoke test  (default: the project model)
#   JOBS       parallel build jobs                      (default: nproc)
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v mamba       >/dev/null 2>&1 || { echo "ERROR: mamba required (install Miniforge)."; exit 1; }
command -v nvidia-smi  >/dev/null 2>&1 || { echo "ERROR: nvidia-smi required (NVIDIA driver)."; exit 1; }
command -v git         >/dev/null 2>&1 || { echo "ERROR: git required."; exit 1; }

# --- detect driver CUDA + GPU arch -----------------------------------------
DRIVER_CUDA="$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
DET_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"

CUDA_VER="${CUDA_VER:-$DRIVER_CUDA}"
CUDA_ARCH="${CUDA_ARCH:-$DET_ARCH}"
ENV_NAME="${ENV_NAME:-llamacpp-cuda}"
SRC_DIR="${SRC_DIR:-$REPO_ROOT/vendor/llama.cpp}"
MODEL="${MODEL:-$REPO_ROOT/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
JOBS="${JOBS:-$(nproc)}"

[ -n "$CUDA_VER" ]  || { echo "ERROR: couldn't detect driver CUDA version; set CUDA_VER=X.Y."; exit 1; }
[ -n "$CUDA_ARCH" ] || { echo "ERROR: couldn't detect GPU arch; set CUDA_ARCH (e.g. 75 for Turing)."; exit 1; }

# Default to the latest release tag (tested), not master. sort -V handles bNNNN numerically.
if [ -z "${LLAMA_REF:-}" ]; then
  LLAMA_REF="$(git ls-remote --tags --refs https://github.com/ggml-org/llama.cpp 'refs/tags/b[0-9]*' 2>/dev/null \
                | sed 's#.*/##' | sort -V | tail -1)"
  LLAMA_REF="${LLAMA_REF:-master}"
fi

# nvcc needs a host gcc within range: gcc<=12 for CUDA 12.x, gcc<=11 for 11.x.
CUDA_MAJOR="${CUDA_VER%%.*}"
if [ "${CUDA_MAJOR:-12}" -ge 12 ]; then GCCV=12; else GCCV=11; fi

echo ">> driver supports CUDA up to : $DRIVER_CUDA"
echo ">> building against CUDA       : $CUDA_VER (conda-forge toolkit, host gcc $GCCV)"
echo ">> target GPU arch             : sm_$CUDA_ARCH"
echo ">> llama.cpp ref               : $LLAMA_REF"
echo ">> env / src / jobs            : $ENV_NAME / $SRC_DIR / $JOBS"
echo

# --- 1. toolchain + runtime env --------------------------------------------
if mamba env list | grep -qE "/${ENV_NAME}\$|^\s*${ENV_NAME}\s"; then
  echo ">> env '$ENV_NAME' exists — skipping create (delete it to rebuild the toolchain)"
else
  echo ">> creating env '$ENV_NAME' (cuda-toolkit $CUDA_VER + gcc $GCCV + cmake/ninja) ..."
  # Same package set as ../environment-build.yml, but with the CUDA version
  # auto-detected from the driver instead of pinned. Keep the two in sync.
  mamba create -y -n "$ENV_NAME" -c conda-forge \
    cuda-toolkit="$CUDA_VER" "cuda-version=$CUDA_VER" \
    "gxx_linux-64=$GCCV" "gcc_linux-64=$GCCV" \
    cmake ninja git make
fi

# Capture env paths WITHOUT a login shell (avoids bashrc stdout pollution).
ENV_PREFIX="$(mamba run -n "$ENV_NAME" printenv CONDA_PREFIX)"
# conda's compiler activation sets CXX to the prefixed g++; use it as nvcc's host compiler
HOST_CXX="$(mamba run -n "$ENV_NAME" printenv CXX 2>/dev/null || true)"
HOST_CXX="${HOST_CXX:-$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++}"
echo ">> env prefix     : $ENV_PREFIX"
echo ">> nvcc host c++   : $HOST_CXX"

# --- 2. fetch source --------------------------------------------------------
if [ -d "$SRC_DIR/.git" ]; then
  echo ">> updating source in $SRC_DIR ($LLAMA_REF)"
  git -C "$SRC_DIR" fetch --depth 1 origin "$LLAMA_REF" && git -C "$SRC_DIR" checkout -q FETCH_HEAD || true
else
  echo ">> cloning llama.cpp ($LLAMA_REF) into $SRC_DIR"
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp "$SRC_DIR" 2>/dev/null \
    || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC_DIR"
fi

# --- 3. configure (cheap — fails fast if the toolchain is wrong) ------------
BUILD_DIR="$SRC_DIR/build"
echo ">> configuring (GGML_CUDA=ON, arch=$CUDA_ARCH) ..."
mamba run --no-capture-output -n "$ENV_NAME" cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_RPATH="$ENV_PREFIX/lib" \
  -DCMAKE_INSTALL_RPATH="$ENV_PREFIX/lib" 2>&1 | tee "$SRC_DIR/configure.log" | grep -iE "cuda compiler|working (cxx|cuda)|host compiler|cuda arch|cublas|found cuda" || true
echo ">> (full configure output in $SRC_DIR/configure.log)"

# --- 4. build ---------------------------------------------------------------
echo ">> building (10-20 min) ..."
mamba run --no-capture-output -n "$ENV_NAME" cmake --build "$BUILD_DIR" -j "$JOBS" \
  --target llama-server llama-cli llama-bench

BIN="$BUILD_DIR/bin/llama-server"
[ -x "$BIN" ] || { echo "ERROR: build did not produce $BIN"; exit 1; }

# --- 5. smoke test: actually DECODE (a kernel must load) --------------------
# NOTE: --list-devices is NOT enough — the broken conda build passed that and only
# crashed at compute. We run a tiny generation to prove the CUDA kernels load.
echo ">> verifying the CUDA kernels actually run (tiny decode) ..."
if [ -f "$MODEL" ]; then
  SMOKE_LOG="$SRC_DIR/smoketest.log"
  # Write to a file first (no `grep -q` in the pipe: it would SIGPIPE llama-bench
  # mid-run and, with `set -o pipefail`, falsely fail a working build).
  set +e
  mamba run --no-capture-output -n "$ENV_NAME" "$BUILD_DIR/bin/llama-bench" \
    -m "$MODEL" -ngl 99 --n-cpu-moe "${SMOKE_NCMOE:-22}" -n 8 -p 8 > "$SMOKE_LOG" 2>&1
  set -e
  # A completed benchmark prints a result row like "... | 44.18 ± 3.63". If the
  # CUDA kernels were invalid the run aborts before any such row appears.
  if grep -qE '\|[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*±' "$SMOKE_LOG"; then
    echo ">> SMOKE TEST PASSED — CUDA kernels load and run."
    grep -E 'gemma|backend|±' "$SMOKE_LOG" | tail -4 | sed 's/^/   /'
  else
    echo ">> SMOKE TEST FAILED — kernels did not run. See $SMOKE_LOG"
    echo "   (likely the wrong CUDA_ARCH or a CUDA newer than your driver supports)"
    exit 1
  fi
else
  echo ">> (model not found at \$MODEL — skipping decode test; run scripts/setup.sh to enable it)"
  mamba run -n "$ENV_NAME" "$BIN" --list-devices 2>/dev/null | sed 's/^/   /'
fi

cat <<EOF

>> Done. CUDA-built binaries: $BUILD_DIR/bin
   Run the server on the native CUDA backend:
     BACKEND=cuda bash scripts/run-server.sh
   (run-server.sh auto-detects this build and its env.)
EOF
