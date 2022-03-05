find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)

file(GLOB_RECURSE MODEL_SIMULATOR_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${MODEL_SIMULATOR_FILES})

find_path(CPPZMQ_INCLUDE_DIR NAMES zmq.hpp)
message(STATUS "cppzmq dir: ${CPPZMQ_INCLUDE_DIR}")

add_library(
        ModelSimulatorLib STATIC
        src/core/model_simulator_v2.cpp
        src/core/simulation_interval.cpp
        src/util/distribution_functions.cpp
        src/util/hardware_information.cpp
        src/util/utility.cpp
        src/util/sim_result.cpp

        # Header for IDE support###

        inc/core/model_simulator_v2.h
        inc/core/simulation_interval.h
        inc/util/cpu_information
        inc/util/distribution_functions.h
        inc/util/hardware_information.h
        inc/util/utility.h
        inc/util/sim_result.h)

if (UNIX)

    target_sources(ModelSimulatorLib
            PRIVATE src/util/linux/linux_cpu_information.cpp)
elseif (WIN32)
    target_sources(ModelSimulatorLib
            PRIVATE src/util/windows/win_cpu_information.cpp)
endif ()

target_link_libraries(
        ModelSimulatorLib
        PUBLIC CommonLib SerializedGisaxsLib GisaxsLib ConnectionUtility nlohmann_json
        nlohmann_json::nlohmann_json
        PRIVATE cppzmq)

if (UNIX)

    target_link_libraries(ModelSimulatorLib PRIVATE stdc++fs)
elseif (WIN32)
    target_link_libraries(ModelSimulatorLib PRIVATE bcrypt)

endif ()

target_include_directories(ModelSimulatorLib PUBLIC inc)
