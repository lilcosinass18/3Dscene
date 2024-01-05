add_library(noise noise.cpp include/noise/noise.hpp)
set_property(TARGET noise PROPERTY CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(noise PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS noise)