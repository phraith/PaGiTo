find_package(GTest REQUIRED)

get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
message("ALL LANGUAGES: ${LANGUAGES}")
foreach(LANG ${LANGUAGES})
    message("${LANG} compiler is \"${CMAKE_${LANG}_COMPILER_ID}\"")
endforeach()

include(CodeCoverage)

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

setup_target_for_coverage_lcov(
            NAME "coverage_html"
            EXECUTABLE ./GisaxsTests
            BASE_DIRECTORY "${CMAKE_SOURCE_DIR}/")

install(TARGETS GisaxsTests
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)                        