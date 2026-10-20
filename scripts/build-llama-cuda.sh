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
#   LLAMA_REF  llama.cpp git ref/tag/commit to build   (default: latest b* release)
#   LLAMA_REPO git URL to build from                    (default: ggml-org/llama.cpp;
#              e.g. a fork such as https://github.com/PrismML-Eng/llama.cpp)
#   PATCH      a patch file to `git apply` on top of LLAMA_REF before building
#              (default: none; e.g. patches/llama.cpp-dory-pr30306.patch with its base
#              commit as LLAMA_REF — see docs/dory-on-8gb.md)
#   SRC_DIR    clone/build location                    (default: ./vendor/llama.cpp)
#   MODEL      gguf used for the post-build smoke test  (default: the project model)
#   JOBS       parallel build jobs                      (default: nproc)
#   TARGETS    cmake targets to build                   (default: llama-server llama-cli
#              llama-bench; add llama-diffusion-cli for diffusion models)
#
set -euo pipefail

# -h / --help: print this script's header comment block and exit.
for _arg in "$@"; do case "$_arg" in
  -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \?//p}' "${BASH_SOURCE[0]}"; exit 0 ;;
esac; done

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
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
PATCH="${PATCH:-}"
if [ -n "$PATCH" ]; then
  PATCH="$(cd "$(dirname "$PATCH")" && pwd)/$(basename "$PATCH")"
  [ -f "$PATCH" ] || { echo "ERROR: PATCH file not found: $PATCH"; exit 1; }
fi
MODEL="${MODEL:-$REPO_ROOT/models/gemma4-26b-a4b-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf}"
JOBS="${JOBS:-$(nproc)}"
TARGETS="${TARGETS:-llama-server llama-cli llama-bench}"

[ -n "$CUDA_VER" ]  || { echo "ERROR: couldn't detect driver CUDA version; set CUDA_VER=X.Y."; exit 1; }
[ -n "$CUDA_ARCH" ] || { echo "ERROR: couldn't detect GPU arch; set CUDA_ARCH (e.g. 75 for Turing)."; exit 1; }

# Default to the latest release tag (tested), not master. sort -V handles bNNNN numerically.
if [ -z "${LLAMA_REF:-}" ]; then
  LLAMA_REF="$(git ls-remote --tags --refs "$LLAMA_REPO" 'refs/tags/b[0-9]*' 2>/dev/null \
                | sed 's#.*/##' | sort -V | tail -1)"
  LLAMA_REF="${LLAMA_REF:-master}"
fi

# nvcc needs a host gcc within range: gcc<=12 for CUDA 12.x, gcc<=11 for 11.x.
CUDA_MAJOR="${CUDA_VER%%.*}"
if [ "${CUDA_MAJOR:-12}" -ge 12 ]; then GCCV=12; else GCCV=11; fi

echo ">> driver supports CUDA up to : $DRIVER_CUDA"
echo ">> building against CUDA       : $CUDA_VER (conda-forge toolkit, host gcc $GCCV)"
echo ">> target GPU arch             : sm_$CUDA_ARCH"
echo ">> llama.cpp ref               : $LLAMA_REF  ($LLAMA_REPO)"
[ -n "$PATCH" ] && echo ">> patch                       : $PATCH"
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
# Always end up on exactly LLAMA_REF (tag, branch or commit) — fail loudly rather than
# silently building whatever was already checked out or the default branch's HEAD.
if [ ! -d "$SRC_DIR/.git" ]; then
  echo ">> cloning $LLAMA_REPO into $SRC_DIR"
  mkdir -p "$(dirname "$SRC_DIR")"
  git init -q "$SRC_DIR"
  git -C "$SRC_DIR" remote add origin "$LLAMA_REPO"
fi
echo ">> fetching $LLAMA_REF into $SRC_DIR"
git -C "$SRC_DIR" remote set-url origin "$LLAMA_REPO"
git -C "$SRC_DIR" fetch --depth 1 origin "$LLAMA_REF" \
  || { echo "ERROR: could not fetch '$LLAMA_REF' from $LLAMA_REPO"; exit 1; }
git -C "$SRC_DIR" checkout -q FETCH_HEAD
echo ">> source at $(git -C "$SRC_DIR" rev-parse HEAD)"

# Optional patch on top of LLAMA_REF. A re-run finds it already applied (the checkout above
# keeps working-tree changes) and skips it; anything else that doesn't apply cleanly fails.
if [ -n "$PATCH" ]; then
  if git -C "$SRC_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo ">> patch already applied: $(basename "$PATCH")"
  elif git -C "$SRC_DIR" apply --check "$PATCH"; then
    git -C "$SRC_DIR" apply "$PATCH"
    echo ">> applied patch: $(basename "$PATCH")"
  else
    echo "ERROR: $PATCH does not apply to $LLAMA_REF (wrong LLAMA_REF, or $SRC_DIR has local changes)"
    exit 1
  fi
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
  --target $TARGETS

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
