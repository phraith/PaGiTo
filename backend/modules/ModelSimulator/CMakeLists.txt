find_package(amqpcpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB_RECURSE MODEL_SIMULATOR_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${MODEL_SIMULATOR_FILES})


add_library(
        ModelSimulatorLib STATIC
        src/core/model_simulator_v2.cpp
        src/core/simulation_interval.cpp
        src/util/hardware_information.cpp
        src/util/utility.cpp
        src/util/sim_result.cpp

        # Header for IDE support###

        inc/core/model_simulator_v2.h
        inc/core/simulation_interval.h
        inc/util/cpu_information
        inc/util/hardware_information.h
        inc/util/utility.h
        inc/util/sim_result.h)

if (UNIX)
    target_sources(ModelSimulatorLib
            PRIVATE src/util/linux/cpu_information.cpp)
elseif (WIN32)
    target_sources(ModelSimulatorLib
            PRIVATE src/util/windows/cpu_information.cpp)
endif ()

target_link_libraries(
        ModelSimulatorLib
        PUBLIC CommonLib GisaxsLib RabbitMq nlohmann_json spdlog::spdlog
        nlohmann_json::nlohmann_json
        PRIVATE amqpcpp)

if (UNIX)

    target_link_libraries(ModelSimulatorLib PRIVATE stdc++fs)
elseif (WIN32)
    target_link_libraries(ModelSimulatorLib PRIVATE bcrypt)

endif ()

target_include_directories(ModelSimulatorLib PUBLIC inc)
