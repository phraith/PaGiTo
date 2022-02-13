include(CheckLanguage)
check_language(CUDA)

find_package(nlohmann_json REQUIRED)

file(GLOB_RECURSE COMMON_LIB_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${COMMON_LIB_FILES})

add_library(CommonLib STATIC inc/parameter_definitions/transformation_container.h)

target_sources(
        CommonLib
        PRIVATE src/common/uuid_generator.cpp
        src/common/beam_configuration.cpp
        src/common/detector.cpp
        src/common/experimental_model.cpp
        src/common/fitting_parameter.cpp
        src/common/layer.cpp
        src/common/device.cpp
        src/common/qgrid.cpp
        src/common/image_data.cpp
        src/common/simulation_description.cpp
        src/common/sample.cpp
        src/parameter_definitions/experimental_setup.cpp
        src/parameter_definitions/experimental_setup.cpp
        src/parameter_definitions/detector_setup.cpp
        src/parameter_definitions/scattering_setup.cpp
        src/parameter_definitions/substrate_setup.cpp
        src/common/unitcell_v2.cpp
        src/common/flat_unitcell.cpp
        src/common/unitcell_utility.cpp
        inc/common/unitcell_utility.h
        inc/common/flat_unitcell.h
        inc/common/unitcell_v2.h
        inc/parameter_definitions/substrate_setup.h
        inc/parameter_definitions/scattering_setup.h
        inc/parameter_definitions/detector_setup.h
        inc/parameter_definitions/experimental_setup.h
        inc/common/beam_configuration.h
        inc/common/detector.h
        inc/common/experimental_model.h
        inc/common/fitting_parameter.h
        inc/common/layer.h
        inc/common/sample.h
        inc/common/simulation_description.h
        inc/common/device.h
        src/common/timer.cpp
        inc/common/uuid_generator.h
        inc/common/image_data.h
        inc/common/qgrid.h
        inc/common/timer.h
        inc/common/standard_defs.h
        inc/common/standard_constants.h)

target_include_directories(CommonLib PUBLIC inc)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    target_link_libraries(CommonLib PRIVATE CUDA::cudart CUDA::cuda_driver nlohmann_json
            nlohmann_json::nlohmann_json)

    target_include_directories(CommonLib PUBLIC inc/cuda_types
            ${CUDAToolkit_INCLUDE_DIRS})
else ()
    target_include_directories(CommonLib PUBLIC inc/host_types)
endif ()
