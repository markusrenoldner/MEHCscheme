# MEHCscheme
Mass, energy, helicity conserving, mimetic galerkin finite element discretisation based on [https://arxiv.org/abs/2104.13023](https://arxiv.org/abs/2104.13023) implemented in [https://mfem.org/](https://mfem.org/)

# CMake and Make
TODO (basically download and unpack mfem and glvis into the extern folder, then install it there as described on mfem.org; then run the build.sh file to cmake, the compile.sh file to make, the run.sh file to run and the clean.sh file to delete the mesh and solution files afterwards)

# folder structure
* /build will contain all files produced by cmake
* /extern contains the mfem (v4.5) and glvis (v4.2) library
* /playgound contains some files that are not necessary to run the main code
* /src contains the cpp files that implement the mehc scheme
* .sh are all shell scripts, that contain some handy commands for building, compiling, running and cleaning
