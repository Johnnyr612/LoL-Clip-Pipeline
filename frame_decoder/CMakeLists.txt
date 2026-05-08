cmake_minimum_required(VERSION 3.20)
project(frame_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(AV REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil libswscale)

add_library(frame_decoder SHARED frame_decoder.cpp)
target_link_libraries(frame_decoder PRIVATE PkgConfig::AV)
target_compile_options(frame_decoder PRIVATE -O3)
