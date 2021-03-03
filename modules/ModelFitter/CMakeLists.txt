
file(GLOB_RECURSE MODEL_FITTER_FILES "src/*.cpp" "inc/*.h")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${MODEL_FITTER_FILES})


#creating executable
add_library(ModelFitterLib STATIC)

target_sources(ModelFitterLib
                PRIVATE
                  src/core/connector.cpp
                  src/core/fitting_description.cpp
                  src/core/fitting_result.cpp
                  src/core/model_fitter.cpp
                  src/core/publisher.cpp

                  inc/core/connector.h
                  inc/core/fitting_description.h
                  inc/core/fitting_result.h
                  inc/core/model_fitter.h
                  inc/core/publisher.h     
)

target_link_libraries(  ModelFitterLib 
                            PUBLIC
                              ModelSimulatorLib
                              SerializedGisaxsLib
                              CommonLib
                            PRIVATE
                              pagmo       
                              gtest
                              gtest_main
                              nlohmann_json::nlohmann_json
                              cppzmq
                      )

if (WIN32)
  target_link_libraries(    ModelFitterLib
                              PRIVATE 
                                bcrypt
  )
endif (WIN32)

target_include_directories (    ModelFitterLib
                                    PUBLIC 
                                        inc
                            )