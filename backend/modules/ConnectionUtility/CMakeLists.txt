find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)

file(GLOB_RECURSE CONNECTION_UTILITY_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${CONNECTION_UTILITY_FILES})

add_library (ConnectionUtility STATIC)

target_sources( ConnectionUtility
                    PRIVATE
                        src/connector_v2.cpp

                        inc/connector_v2.h
)

target_include_directories (    ConnectionUtility
                                    PUBLIC 
                                        inc
                            )

target_link_libraries(ConnectionUtility
        PRIVATE
            cppzmq
            nlohmann_json
            nlohmann_json::nlohmann_json
)