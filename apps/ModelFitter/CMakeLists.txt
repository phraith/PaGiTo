add_executable(ModelFitter)

target_compile_features(ModelFitter PRIVATE cxx_std_17)

target_sources  (ModelFitter
                    PRIVATE
                        src/main.cpp
                )

target_link_libraries(ModelFitter PRIVATE ModelFitterLib)

if(WIN32)
  add_custom_command(TARGET ModelFitter POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_BINARY_DIR}/extern/pagmo2/\$\(Configuration\)/pagmo.dll"
      $<TARGET_FILE_DIR:ModelFitter>)
endif()

# Provide installation instructions
install(TARGETS ModelFitter
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)