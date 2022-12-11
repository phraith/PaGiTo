cmake_minimum_required(VERSION 3.18)

project(GisaxsLib LANGUAGES CXX CUDA)

if (MSVC)
    set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
            $(VC_IncludePath);
            $(WindowsSDK_IncludePath);
            ${CMAKE_SOURCE_DIR}/modules/Common/inc/common;
            ${CMAKE_SOURCE_DIR}/modules/Common/inc/cuda_types;
            ${CMAKE_SOURCE_DIR}/modules/GisaxsLib/inc;
            )
endif (MSVC)

#creating library
add_library(GisaxsCudaLib
        STATIC
        inc/gpu/util/conversion_helper.h src/gpu/util/conversion_helper.cpp src/gpu/core/random_generator.cpp inc/gpu/core/random_generator.h)

target_sources(GisaxsCudaLib
        PRIVATE
        src/gpu/core/gisaxs_functions.cu
        src/gpu/core/gisaxs_functions_v2.cu
        src/gpu/core/gpu_qgrid.cu
        src/gpu/ff/cylinder.cu
        src/gpu/ff/sphere.cu
        src/gpu/ff/trapezoid.cu
        src/gpu/util/test.cu
        src/gpu/util/util.cu

        src/gpu/core/stream.cpp
        src/gpu/core/stream_provider.cpp
        src/gpu/core/event.cpp
        src/gpu/core/event_provider.cpp
        src/gpu/core/gpu_information.cpp
        src/gpu/core/gpu_device.cpp

        src/gpu/core/gpu_memory_provider_v2.cpp
        src/gpu/core/gpu_device_v2.cpp
        inc/gpu/core/gpu_device_v2.h

        inc/gpu/core/gpu_qgrid.h
        inc/gpu/core/gpu_memory_provider_v2.h
        inc/gpu/core/gisaxs_functions.h
        inc/gpu/core/gisaxs_functions_v2.h
        inc/gpu/ff/shape.h
        inc/gpu/ff/sphere.h
        inc/gpu/ff/cylinder.h
        inc/gpu/ff/trapezoid.h
        inc/gpu/util/cuda_numerics.h
        inc/gpu/util/test.h
        inc/gpu/util/util.h

        inc/gpu/core/stream_provider.h
        inc/gpu/core/stream.h
        inc/gpu/core/gpu_device.h
        inc/gpu/core/gpu_helper.h
        inc/gpu/core/gpu_information.h
        inc/gpu/core/gpu_memory_block.h
        inc/gpu/core/event.h
        )

target_compile_options(GisaxsCudaLib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        #--ptxas-options=-v
        --generate-line-info
        >)

file(GLOB_RECURSE GISAXS_CUDA_FILES "${CMAKE_CURRENT_LIST_DIR}/src/*.cu" "${CMAKE_CURRENT_LIST_DIR}/inc/*.h" "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${GISAXS_CUDA_FILES})

target_link_libraries(GisaxsCudaLib
        PUBLIC
        CommonLib
        PRIVATE
        CUDA::curand
        CUDA::cudart
        CUDA::cuda_driver
        )

target_include_directories(GisaxsCudaLib
        PUBLIC
        inc
        )


set_target_properties(GisaxsCudaLib
        PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        )