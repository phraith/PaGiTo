find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB_RECURSE CONNECTION_UTILITY_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${CONNECTION_UTILITY_FILES})

add_library(ConnectionUtility STATIC src/client.cpp inc/client.h)
add_compile_definitions(ZMQ_BUILD_DRAFT_API)
target_sources(ConnectionUtility
        PRIVATE
        src/broker.cpp
        src/majordomo_utility.cpp
        src/service_information.cpp
        src/worker_information.cpp
        src/worker_request_handler.cpp
        src/worker.cpp
        inc/service.h
        inc/worker.h
        inc/worker_request_handler.h
        inc/worker_information.h
        inc/service_information.h
        inc/majordomo_utility.h
        inc/broker.h
        )

target_include_directories(ConnectionUtility
        PUBLIC
        inc
        )

target_link_libraries(ConnectionUtility
        PRIVATE
        cppzmq
        nlohmann_json
        nlohmann_json::nlohmann_json
        spdlog::spdlog
        )