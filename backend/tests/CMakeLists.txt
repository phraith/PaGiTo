find_package(GTest REQUIRED)

add_executable(GisaxsTests src/test_cmaes.cpp)

target_sources(GisaxsTests
                    PRIVATE
                        main.cpp
                        src/test_model_simulator.cpp
                        src/test_cmaes.cpp)

target_link_libraries(GisaxsTests
                            PRIVATE
                                GisaxsLib
                                ModelSimulatorLib
                                GTest::gtest
                                GTest::gtest_main)

install(TARGETS GisaxsTests
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)                        