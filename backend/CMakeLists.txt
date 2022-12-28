cmake_minimum_required (VERSION 3.18)

project (GisaxsModelingFramework)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(CodeCoverage)
append_coverage_compiler_flags()
setup_target_for_coverage_lcov(
            NAME "GisaxsCoverage"
            DEPENDENCIES GisaxsTests  
            EXECUTABLE ./tests/GisaxsTests
            BASE_DIRECTORY "${PROJECT_SOURCE_DIR}/src"
            EXCLUDE "/usr*" "/opt*" "/src/tests*") 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
include(CheckLanguage)
check_language(CUDA)
enable_language(CXX)
get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
message("ALL LANGUAGES: ${LANGUAGES}")
foreach(LANG ${LANGUAGES})
    message("${LANG} compiler is \"${CMAKE_${LANG}_COMPILER_ID}\"")
endforeach()

#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)

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

#add_subdirectory(extern/serialization)

add_subdirectory(modules/Common)
add_subdirectory(modules/ConnectionUtility)
add_subdirectory(modules/ModelSimulator)
add_subdirectory(modules/ModelFitter)
add_subdirectory(modules/GisaxsLib)

#if (CMAKE_CUDA_COMPILER)
    add_subdirectory(tests)
    # add_subdirectory(coverage)
#endif()

#add_subdirectory(apps/ModelFitter)
add_subdirectory(apps/ModelingFramework)