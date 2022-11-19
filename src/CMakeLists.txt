add_executable(${PROJECT_NAME} transpose1.cpp)



target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem-4.5)

target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem-4.5)

target_link_libraries(${PROJECT_NAME} mfem)




