include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.9.1
)
FetchContent_Declare(
    zmq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG        v4.3.3
)

FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG        v4.7.1
)

FetchContent_MakeAvailable(zmq)
FetchContent_MakeAvailable(cppzmq)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

file(GLOB_RECURSE MODEL_SIMULATOR_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${MODEL_SIMULATOR_FILES})

add_library (   ModelSimulatorLib 
                    STATIC
                      src/core/model_simulator.cpp
                      src/core/simulation_interval.cpp

                      src/util/distribution_functions.cpp 
                      src/util/hardware_information.cpp 
                      
                      src/util/sim_connector.cpp src/util/sim_publisher.cpp src/util/utility.cpp
                      src/util/sim_result.cpp 

                      #### Header for IDE support###
                      inc/core/model_simulator.h
                      inc/core/simulation_interval.h

                      inc/util/cpu_information inc/util/distribution_functions.h
                       
                      inc/util/hardware_information.h
                      
                      inc/util/sim_connector.h inc/util/sim_publisher.h inc/util/utility.h
                      inc/util/sim_result.h
)


if (UNIX)

  target_sources( ModelSimulatorLib
                    PRIVATE
                      src/util/linux/linux_cpu_information.cpp
  )
elseif(WIN32)
  target_sources( ModelSimulatorLib
                    PRIVATE
                      src/util/windows/win_cpu_information.cpp
  )
endif()

target_link_libraries(  ModelSimulatorLib
                            PUBLIC
                              CommonLib
                              SerializedGisaxsLib
                              GisaxsLib
                            PRIVATE
                              nlohmann_json::nlohmann_json
                              cppzmq
                      )

if (UNIX)

    target_link_libraries(  ModelSimulatorLib
                                PRIVATE 
                                    stdc++fs
                         )
elseif (WIN32)
  target_link_libraries(    ModelSimulatorLib
                              PRIVATE 
                                bcrypt
  )

endif()

target_include_directories (    ModelSimulatorLib
                                    PUBLIC 
                                        inc
                            )