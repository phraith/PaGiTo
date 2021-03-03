cmake_minimum_required (VERSION 3.18)

project (GisaxsModelingFramework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(CheckLanguage)
check_language(CUDA)

find_package(Boost 1.62 REQUIRED COMPONENTS random)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    #set(CMAKE_CUDA_ARCHITECTURES 52)
    set(CMAKE_CUDA_ARCHITECTURES 75)

    find_package(CUDAToolkit)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(PAGMO_WITH_EIGEN3 "Use pagmo2 with Eigen3 " ON)
add_subdirectory(extern/pagmo2) 
add_subdirectory(extern/serialization)

add_subdirectory(modules/Common)
add_subdirectory(modules/ModelSimulator)
add_subdirectory(modules/ModelFitter)
add_subdirectory(modules/GisaxsLib)

if (CMAKE_CUDA_COMPILER)
    add_subdirectory(tests)
endif()
 
add_subdirectory(apps/ModelSimulator)
add_subdirectory(apps/ModelFitter)



set_target_properties(pagmo gtest gmock uninstall gtest_main gmock_main PROPERTIES FOLDER HiddenTargets)
set_target_properties(clang-format clang-format-diff clang-format-check libzmq libzmq-static PROPERTIES FOLDER HiddenTargets)
set_target_properties(capnp capnp_tool capnpc capnp-json kj kj-async SerializedGisaxsLib PROPERTIES FOLDER HiddenTargets)