find_package(nlohmann_json REQUIRED)

file(GLOB_RECURSE SERIALIZATION_UTILITY_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
             FILES ${SERIALIZATION_UTILITY_FILES})

add_library(SerializationUtility STATIC)

target_sources(
  SerializationUtility
  PRIVATE src/json_factory.cpp
          inc/json_factory.h)

target_include_directories(SerializationUtility PUBLIC inc)

target_link_libraries(
  SerializationUtility PRIVATE nlohmann_json nlohmann_json::nlohmann_json
                               CommonLib)
