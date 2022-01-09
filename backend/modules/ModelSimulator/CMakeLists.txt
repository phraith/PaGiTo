find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)

file(GLOB_RECURSE MODEL_SIMULATOR_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${MODEL_SIMULATOR_FILES})

add_library (   ModelSimulatorLib 
                    STATIC
                      src/core/model_simulator.cpp
                      src/core/simulation_interval.cpp

                      src/util/distribution_functions.cpp 
                      src/util/hardware_information.cpp 
                      
                      src/util/sim_connector.cpp
                      src/util/sim_publisher.cpp
                      src/util/utility.cpp
                      src/util/sim_result.cpp
        ../ConnectionUtility/src/connector_v2.cpp

                      #### Header for IDE support###
                      inc/core/model_simulator.h
                      inc/core/simulation_interval.h
        ../ConnectionUtility/inc/connector_v2.h

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
                              nlohmann_json
                              nlohmann_json::nlohmann_json
                            PRIVATE
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