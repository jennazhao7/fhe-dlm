#!/bin/bash
# One-time, FRONT-END ONLY (needs network for pip). Multi-GPU FIDESlib for the
# E5 bootstrap (HANDOFF §10): at N=2^17, depth 43, [4,4] the bootstrap does not
# fit one 24 GB RTX 6000 (full slots: 12.3 GB of C2S/S2C plaintexts alone,
# before 354 MB/key rotation keys). FIDESlib splits limbs and keys across
# devices when built with NCCL; the single-GPU build compiled that path out
# because CMake found no NCCL.
#
# NCCL comes from the nvidia-nccl-cu13 wheel installed with --target into
# third_party/nccl-cu13, NOT into the fhedlm env: the env's torch ships
# nvidia-nccl-cu12, and both wheels write nvidia/nccl/lib/libnccl.so.2.
# The result installs into third_party/fideslib-mgpu; third_party/fideslib
# (single GPU, used by job_gpu_fides.sh) is left untouched.
#
#   bash cluster/setup_fideslib_mgpu.sh          # ~15-20 min at -j8
source ~/.bashrc
set -euo pipefail
FHEDLM_ROOT="${FHEDLM_ROOT:-/groups/tjung/jzhao7/fhe-dlm}"
TP="$FHEDLM_ROOT/third_party"
ENV=/groups/tjung/jzhao7/conda-envs/fhedlm
NCCL_VER="${NCCL_VER:-2.32.3}"
JOBS="${JOBS:-8}"

module load "${CUDA_MODULE:-cuda/13.2.1}"
CUDA_PATH="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
conda activate "$ENV"   # cmake < 4 from pip (see setup_fideslib.sh)

PIP_CACHE_DIR=/groups/tjung/jzhao7/pip-cache python -m pip install -q --upgrade \
  --target "$TP/nccl-cu13" "nvidia-nccl-cu13==$NCCL_VER"
NCCL_H=$(find "$TP/nccl-cu13" -name nccl.h | head -1)
NCCL_SO=$(find "$TP/nccl-cu13" -name "libnccl.so.2" | head -1)
[ -n "$NCCL_H" ] && [ -n "$NCCL_SO" ] || { echo "FATAL: nccl.h / libnccl.so.2 not found in the wheel"; exit 1; }
NCCL_INC=$(dirname "$NCCL_H"); NCCL_LIB=$(dirname "$NCCL_SO")
ln -sf libnccl.so.2 "$NCCL_LIB/libnccl.so"   # find_library wants the unversioned name
echo "NCCL $NCCL_VER: include $NCCL_INC, lib $NCCL_LIB"

cd "$TP/FIDESlib"
[ "${CLEAN:-0}" = 1 ] && rm -rf build-mgpu   # default: reuse objects, relink only
cmake -B build-mgpu -DCMAKE_BUILD_TYPE=Release -DFIDESLIB_ARCH=75 -DCUDA_PATH="$CUDA_PATH" \
      -DOPENFHE_INSTALL_PREFIX="$TP/openfhe-fides" -DFIDESLIB_INSTALL_PREFIX="$TP/fideslib-mgpu" \
      -DFIDESLIB_COMPILE_TESTS=OFF -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
      -DCMAKE_LIBRARY_PATH="$NCCL_LIB" \
      -DCMAKE_CUDA_FLAGS="-I$NCCL_INC" -DCMAKE_CXX_FLAGS="-I$NCCL_INC" \
      -DCMAKE_EXE_LINKER_FLAGS="-L$NCCL_LIB -Wl,-rpath,$NCCL_LIB" 2>&1 | tee build-mgpu.cmake.log
grep -q "Found NCCL" build-mgpu.cmake.log || { echo "FATAL: CMake did not find NCCL"; exit 1; }
nice cmake --build build-mgpu -j"$JOBS" --target install

cmake -S "$FHEDLM_ROOT/gpu/fides_e0" -B "$FHEDLM_ROOT/gpu/fides_e0/build-mgpu" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$TP/fideslib-mgpu;$TP/openfhe-fides" -DCUDAToolkit_ROOT="$CUDA_PATH" \
      -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" -DNCCL_LIB_DIR="$NCCL_LIB"
cmake --build "$FHEDLM_ROOT/gpu/fides_e0/build-mgpu" -j"$JOBS"
echo "=== built: $FHEDLM_ROOT/gpu/fides_e0/build-mgpu/fides_e0 (NCCL lib dir $NCCL_LIB)"
