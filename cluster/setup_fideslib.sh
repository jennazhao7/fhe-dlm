#!/bin/bash
# One-time, FRONT-END ONLY (compute nodes have no outbound network).
# Fetches and builds FIDESlib + its patched OpenFHE + gpu/fides_e0 into
# $FHEDLM_ROOT/third_party (git-ignored, /groups fs -- not $HOME).
# No GPU is needed to compile; the GPU job (cluster/job_gpu_fides.sh) only runs.
#
# sm_75: FIDESlib's default arch list starts at sm_80 (80-real;86;89;90;100;120),
# i.e. Turing is not an advertised target. The code has no sm_80-only
# features left (its memcpy_async paths are commented out, no kernel opts in
# to >48 KB shared memory). Checked off-cluster on 2026-10-05: FIDESlib
# 593ad73 + its patched OpenFHE build cleanly with FIDESLIB_ARCH=75, CUDA
# 12.4.131, gcc 12.4; the binary carries sm_75 SASS only; fides_e0's matvec
# and boot code paths were validated on OpenFHE's CPU fallback at N=2^12.
# Whether the kernels *run* correctly on sm_75 is what the GPU job checks.
#
# Needs: CUDA >= 12.4 (12.0 fails: nvcc 12.0 rejects FIDESlib's
# std::source_location defaults), gcc 11-13, CMake >= 3.25.2, OpenMP.
# Override module names if CRC's differ (`module avail cuda gcc cmake`). If
# CRC has no CUDA >= 12.4 module, a user-space one works (what the off-cluster
# check used):  micromamba create -p $TP/cuda124 -c conda-forge cuda-nvcc=12.4 \
#   cuda-cudart-dev=12.4 cuda-driver-dev=12.4 cuda-nvtx-dev=12.4 cuda-cccl=12.4
# then CUDA_MODULE=none PATH=$TP/cuda124/bin:$PATH bash cluster/setup_fideslib.sh
#
#   bash cluster/setup_fideslib.sh            # ~1 h at -j8
source ~/.bashrc
set -euo pipefail
FHEDLM_ROOT="${FHEDLM_ROOT:-/groups/tjung/jzhao7/fhe-dlm}"
TP="$FHEDLM_ROOT/third_party"
FIDES_REV="${FIDES_REV:-593ad73c4df1b9998a22fc64b47362760d4c5a01}"  # 2.1.3, 2026-09-21
OFHE_TAG="fideslib-ref-v1.5.1.1"                                     # what 2.1.3 patches
JOBS="${JOBS:-8}"

[ "${CUDA_MODULE:-}" = none ] || module load "${CUDA_MODULE:-cuda/12.4}" || true
[ "${GCC_MODULE:-}" = none ] || module load "${GCC_MODULE:-gcc/12}" || true
command -v nvcc >/dev/null || { echo "FATAL: no nvcc; set CUDA_MODULE"; exit 1; }
nv=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
awk -v v="$nv" 'BEGIN{split(v,a,"."); exit !(a[1]>12 || (a[1]==12 && a[2]>=4))}' \
  || { echo "FATAL: nvcc $nv < 12.4 (see header)"; exit 1; }
CUDA_PATH="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
# FIDESlib's CMakeLists hard-codes CMAKE_CXX_COMPILER=g++, so the g++ on PATH
# must be one nvcc accepts.
echo "nvcc $(nvcc --version | tail -1) at $CUDA_PATH; $(g++ --version | head -1); $(cmake --version | head -1)"
cmake_ok=$(cmake --version | head -1 | awk '{split($3,v,"."); print (v[1]>3 || (v[1]==3 && v[2]>=25))}')
if [ "$cmake_ok" != 1 ]; then
  echo "cmake < 3.25: installing a recent one into the fhedlm env"
  conda activate fhedlm && python -m pip install -q "cmake>=3.25.2"
fi

mkdir -p "$TP" && cd "$TP"
[ -d FIDESlib ] || git clone https://github.com/CAPS-UMU/FIDESlib.git
cd FIDESlib && git fetch -q && git checkout -q "$FIDES_REV"

# Patched OpenFHE (what deps/build.sh does, without its `submodule --remote`).
cd deps
rm -rf openfhe-src && git clone -q https://github.com/openfheorg/openfhe-development openfhe-src
cd openfhe-src && git checkout -q "$OFHE_TAG" && git apply ../fideslib-ref-1.5.1.1.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TP/openfhe-fides" \
      -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
nice cmake --build build -j"$JOBS" --target install

cd "$TP/FIDESlib"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DFIDESLIB_ARCH=75 -DCUDA_PATH="$CUDA_PATH" \
      -DOPENFHE_INSTALL_PREFIX="$TP/openfhe-fides" -DFIDESLIB_INSTALL_PREFIX="$TP/fideslib" \
      -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF
nice cmake --build build -j"$JOBS" --target install

cmake -S "$FHEDLM_ROOT/gpu/fides_e0" -B "$FHEDLM_ROOT/gpu/fides_e0/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$TP/fideslib;$TP/openfhe-fides" -DCUDAToolkit_ROOT="$CUDA_PATH"
cmake --build "$FHEDLM_ROOT/gpu/fides_e0/build" -j"$JOBS"
echo "=== built: $FHEDLM_ROOT/gpu/fides_e0/build/fides_e0"
