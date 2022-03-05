cmake_minimum_required(VERSION 3.18)

project(GisaxsCpuLib)
#creating library
add_library(GisaxsCpuLib
        STATIC
        )

target_sources(GisaxsCpuLib
        PRIVATE
            src/cpu_device.cpp
            src/gisaxs_cpu_core.cpp
            inc/cpu_device.h
            inc/gisaxs_cpu_core.h
        )

target_link_libraries(GisaxsCpuLib
        PUBLIC
        CommonLib
        )

target_include_directories(GisaxsCpuLib
        PUBLIC
        inc
        )