add_library(colorLines colorLines.cpp include/colorLines/colorLines.hpp)
set_property(TARGET colorLines PROPERTY CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(colorLines PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS colorLines)
