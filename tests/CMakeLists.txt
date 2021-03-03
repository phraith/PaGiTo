include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)

FetchContent_MakeAvailable(googletest)

add_executable(GisaxsTests)

target_sources(GisaxsTests
                    PRIVATE
                        main.cpp
                        src/test.cpp
                        src/test_memory_provider.cpp
                        src/test_model_simulator.cpp)

target_link_libraries(GisaxsTests
                            PRIVATE
                                GisaxsCudaLib
                                ModelSimulatorLib
                                ModelFitterLib
                                gtest
                                gtest_main)

install(TARGETS GisaxsTests
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)                        