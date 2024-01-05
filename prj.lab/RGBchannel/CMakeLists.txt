add_library(RGBchannel RGBchannel.cpp include/RGBchannel/RGBchannel.hpp)
set_property(TARGET RGBchannel PROPERTY CXX_STANDARD 20)
include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(RGBchannel PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
install(TARGETS RGBchannel)
