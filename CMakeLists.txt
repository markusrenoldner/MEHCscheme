cmake_minimum_required (VERSION 2.9)
project(MEHCscheme)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})


add_subdirectory(extern)
add_subdirectory(src)
