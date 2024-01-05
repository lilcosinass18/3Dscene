add_library(colorHistogram colorHistogram.cpp include/colorHistogram/colorHistogram.hpp)
set_property(TARGET colorHistogram PROPERTY CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(colorHistogram PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS colorHistogram)
