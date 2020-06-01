cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MSVC_Generate_Profiling OFF)
set(SIMD_ISA_MSVC "AVX" CACHE STRING "SIMD Optimization Architecture for MSVC")
set(SIMD_ISA_GCClang "avx" CACHE STRING "SIMD Optimization Architecture for GCC or Clang")
set(CUDA_ISA "35" CACHE STRING "CUDA Instruction Set Architecture version you're targeting")

project(FM32 LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(Libs/imgui)
include_directories(Libs/imgui/examples)
include_directories(${SDL2_INCLUDE_DIRS})

add_library(FM32Kernel
    FM32Kernel.cuh
    FM32Kernel.cu
)

target_link_libraries(FM32Kernel ${CUDA_LIBRARY})

set_target_properties(FM32Kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(FM32
    main.cpp
    InitStuffs.h
    InitStuffs.cpp

    Libs/rtmidi/RtMidi.h
)

target_link_libraries(FM32
    ${CUDA_LIBRARY}
    ${SDL2_LIBRARIES}
    FM32Kernel
)

set_target_properties(FM32 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(FM32 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_${CUDA_ISA},code=sm_${CUDA_ISA}>)

if (MSVC)
    target_compile_options(FM32 PRIVATE /arch:${SIMD_ISA_MSVC})
    if (MSVC_Generate_Profiling)
        target_link_options(FM32 PRIVATE /PROFILE)
    endif()
else()
    target_compile_options(FM32 PRIVATE -m${SIMD_ISA_GCClang} -ffast-math)
endif()

