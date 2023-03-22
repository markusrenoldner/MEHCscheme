# MEHCscheme
A mass, energy, and helicty conserving dual-field Galerkin finite element discretization of the incompressible Navier-Stokes problem based on this paper [https://arxiv.org/abs/2104.13023](https://arxiv.org/abs/2104.13023) implemented in MFEM, see [https://mfem.org/](https://mfem.org/)

# CMake and Make
TODO (basically download and unpack mfem and glvis into the extern folder, then install it there as described on mfem.org; then run the build.sh file to cmake, the compile.sh file to make, the run.sh file to run and the clean.sh file to delete the mesh and solution files afterwards)

# folder structure
* /build will contain all files produced by cmake
* /extern contains the mfem (v4.5) and glvis (v4.2) library
* /out contains plots and data outputs
* /scripts contains some files necessary for plotting etc
* /src contains the cpp files that implement the mehc scheme
* .sh are all shell scripts, that contain some handy commands for building, compiling, running and cleaning

# run the scheme:
1. select the cpp file in src/CMakeLists.txt :
    * periodic-conv.cpp
    * periodic-cons.cpp
    * dirichlet-[placeholder].cpp
2. run ./build.sh
3. run ./compilerun.sh
4. produce a visualization using scripts/plots_[placeholder].ipynb
5. find the visualizsations in out/plots
6. find the raw data in out/rawdata