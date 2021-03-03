include(FetchContent)

FetchContent_Declare(
  capnproto
  GIT_REPOSITORY https://github.com/capnproto/capnproto.git
  GIT_TAG v0.8.0
)

FetchContent_GetProperties(capnproto)
if(NOT capnproto_POPULATED)
  FetchContent_Populate(capnproto)
  add_subdirectory(${capnproto_SOURCE_DIR} ${capnproto_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

capnp_generate_cpp(CAPNP_FITTING_SRCS CAPNP_FITTING_HDRS serialized_fitting_description.capnp)
capnp_generate_cpp(CAPNP_SIMULATION_SRCS CAPNP_SIMULATION_HDRS serialized_simulation_description.capnp)

add_library(SerializedGisaxsLib STATIC)

target_sources(SerializedGisaxsLib
                PRIVATE
                  ${CAPNP_FITTING_SRCS}
                  ${CAPNP_SIMULATION_SRCS}
                  ${CAPNP_FITTING_HDRS}
                  ${CAPNP_SIMULATION_HDRS}  
)

target_link_libraries(SerializedGisaxsLib
                        PUBLIC
                            CapnProto::capnp
                        )

target_include_directories(SerializedGisaxsLib
				PUBLIC
					${CMAKE_CURRENT_BINARY_DIR})
