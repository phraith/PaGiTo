add_executable(ModelingFramework)

target_compile_features(ModelingFramework PRIVATE cxx_std_17)

target_sources  (ModelingFramework
                    PRIVATE
                        src/main.cpp
                )

target_link_libraries(ModelingFramework
        PRIVATE
            ConnectionUtility
            ModelSimulatorLib)

# Provide installation instructions
install(TARGETS ModelFitter
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)