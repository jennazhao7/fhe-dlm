cmake_minimum_required(VERSION 3.25.2)
# E0 GPU feasibility bench on FIDESlib. Built against an installed FIDESlib
# (cluster/setup_fideslib.sh), same pattern as FIDESlib's examples/bootstrap.
set(CMAKE_CXX_STANDARD 20)
project(fides_e0 LANGUAGES CXX)

find_package(fideslib REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

add_executable(fides_e0 src/fides_e0.cpp)
set_source_files_properties(src/fides_e0.cpp PROPERTIES LANGUAGE CXX)
target_link_libraries(fides_e0 PRIVATE fideslib::fideslib CUDA::cudart)
