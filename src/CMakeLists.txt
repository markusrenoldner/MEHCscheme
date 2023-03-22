


# periodic problem
#add_executable(${PROJECT_NAME} periodic-cons.cpp)
#add_executable(${PROJECT_NAME} periodic-conv.cpp)


# dirichlet problem
add_executable(${PROJECT_NAME} dirichlet-conv.cpp)
#add_executable(${PROJECT_NAME} dirichlet-decoupled-primal.cpp)
#add_executable(${PROJECT_NAME} dirichlet-decoupled-dual.cpp)


# other
#add_executable(${PROJECT_NAME} smallsys.cpp)
#add_executable(${PROJECT_NAME} glvis-forcing.cpp)
#add_executable(${PROJECT_NAME} ex1simple.cpp)
#add_executable(${PROJECT_NAME} ex5dirichlet.cpp)
#add_executable(${PROJECT_NAME} initcond_v2.cpp)
#add_executable(${PROJECT_NAME} initcond_v3.cpp)
#add_executable(${PROJECT_NAME} curlcurl.cpp)
#add_executable(${PROJECT_NAME} ex3simple.cpp)
#add_executable(${PROJECT_NAME} BCissue.cpp)
#add_executable(${PROJECT_NAME} dirichlet_primal.cpp)
#add_executable(${PROJECT_NAME} dirichlet_3.cpp)
#add_executable(${PROJECT_NAME} dirichlet_3DPC.cpp)
#add_executable(${PROJECT_NAME} dirichlet_3reduced.cpp)
#add_executable(${PROJECT_NAME} dirichlet_decoupled_FLS.cpp)
#add_executable(${PROJECT_NAME} periodic-dirichlet-test.cpp)





# links MFEM directory
target_include_directories(${PROJECT_NAME} PUBLIC extern/mfem-4.5)
target_link_directories(${PROJECT_NAME} PRIVATE extern/mfem-4.5)

# link utils file
add_library(UTILS utils.cpp)
target_include_directories(UTILS PUBLIC extern/mfem-4.5)
target_link_directories(UTILS PRIVATE extern/mfem-4.5)
target_link_libraries(UTILS mfem)

# link libraries
target_link_libraries(${PROJECT_NAME} UTILS)
target_link_libraries(${PROJECT_NAME} mfem)




