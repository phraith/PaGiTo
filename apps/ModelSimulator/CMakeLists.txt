add_executable(ModelSimulator)

target_compile_features(ModelSimulator PRIVATE cxx_std_17)

target_sources  (ModelSimulator
                    PRIVATE
                        src/main.cpp
                )

target_link_libraries(ModelSimulator PRIVATE ModelSimulatorLib)

# Provide installation instructions
install(TARGETS ModelSimulator
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)