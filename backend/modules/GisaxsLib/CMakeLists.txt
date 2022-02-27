cmake_minimum_required (VERSION 3.18)

project (GisaxsLib LANGUAGES CXX)

add_library (GisaxsLib STATIC dummy.cpp)

if (CMAKE_CUDA_COMPILER)
    add_subdirectory(GisaxsCudaLib)
    target_link_libraries ( GisaxsLib PUBLIC GisaxsCudaLib)
    target_compile_definitions(GisaxsCudaLib PUBLIC CUDA_GPUS_AVAILABLE=1)
endif()

add_subdirectory(GisaxsCpuLib)
target_link_libraries ( GisaxsLib PUBLIC GisaxsCpuLib)