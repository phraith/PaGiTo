cmake_minimum_required (VERSION 3.18)

project (GisaxsModelingFramework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
include(CheckLanguage)
check_language(CUDA)



#print_all_variables()

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    #set(CMAKE_CUDA_ARCHITECTURES 52)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

add_subdirectory(extern/serialization)

add_subdirectory(modules/Common)
add_subdirectory(modules/ConnectionUtility)
add_subdirectory(modules/ModelSimulator)
add_subdirectory(modules/ModelFitter)
add_subdirectory(modules/GisaxsLib)

if (CMAKE_CUDA_COMPILER)
    add_subdirectory(tests)
endif()

#add_subdirectory(apps/ModelFitter)
add_subdirectory(apps/ModelingFramework)