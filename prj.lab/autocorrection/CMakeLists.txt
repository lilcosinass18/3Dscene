add_library(autocorrection autocorrection.cpp include/autocorrection/autocorrection.hpp)
set_property(TARGET autocorrection PROPERTY CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(autocorrection PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS autocorrection)
