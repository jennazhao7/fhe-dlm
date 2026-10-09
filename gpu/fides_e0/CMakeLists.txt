cmake_minimum_required(VERSION 3.25.2)
# E0 GPU feasibility bench on FIDESlib. Built against an installed FIDESlib
# (cluster/setup_fideslib.sh), same pattern as FIDESlib's examples/bootstrap.
# gpu_encode.cu reaches into FIDESlib's installed internal headers (CKKS/*.cuh)
# and OpenFHE's, so it is compiled with nvcc for the same arch as FIDESlib.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 75)          # Quadro RTX 6000 (CRC jung_gpu)
project(fides_e0 LANGUAGES CXX CUDA)

find_package(fideslib REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp --diag-suppress=3133,611 -Wno-deprecated-gpu-targets")

# OpenFHE as installed for FIDESlib (third_party/openfhe-fides), found next to
# the FIDESlib prefix on CMAKE_PREFIX_PATH.
find_path(OPENFHE_INC openfhe/core/math/dftransform.h PATHS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES include REQUIRED)
set(OPENFHE_INCS ${OPENFHE_INC}/openfhe ${OPENFHE_INC}/openfhe/core ${OPENFHE_INC}/openfhe/pke
                 ${OPENFHE_INC}/openfhe/binfhe ${OPENFHE_INC}/openfhe/cereal)

add_executable(fides_e0 src/fides_e0.cpp src/gpu_encode_host.cpp src/gpu_encode.cu)
set_source_files_properties(src/fides_e0.cpp src/gpu_encode_host.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(fides_e0 PRIVATE ${OPENFHE_INCS})
target_link_libraries(fides_e0 PRIVATE fideslib::fideslib CUDA::cudart)
