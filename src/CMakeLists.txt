

add_executable(${PROJECT_NAME} periodic.cpp)





target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem-4.5)

target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem-4.5)





# add also the utils file here:
add_library(UTILS utils.cpp)
target_include_directories(UTILS PUBLIC extern/mfem-4.5)
target_link_directories(UTILS PRIVATE extern/mfem-4.5)
target_link_libraries(UTILS mfem)
target_link_libraries(${PROJECT_NAME} UTILS)




target_link_libraries(${PROJECT_NAME} mfem)




