cmake_minimum_required (VERSION 2.9)
project(MEHCscheme)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(${PROJECT_NAME} ex1simple.cpp)

# add_subdirectory(extern)



target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem-4.5)

target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem-4.5)

target_link_libraries(${PROJECT_NAME} mfem)




