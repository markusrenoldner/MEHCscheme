# MEHCscheme

A mass, energy, and helicty conserving dual-field Galerkin finite element discretization of the incompressible Navier-Stokes problem based on this paper [https://arxiv.org/abs/2104.13023](https://arxiv.org/abs/2104.13023) implemented in MFEM, see [https://mfem.org/](https://mfem.org/)

## CMake and Make

- download and unpack mfem and glvis into the extern folder
- install it there as described on mfem.org
- run the `build.sh` file to cmake, the `compile.sh` file to make, the `run.sh` file to run 
- run `clean.sh` to delete the mesh and solution files afterwards


## file tree (simplified)

```
MEHCscheme
├── extern/                     # MFEM and glvis
│   ├── CMakeLists.txt
│   ├── mfem
│   └── glvis
├── build/                      # appears after building
├── out/                        # data output
├── scripts/                    # plot files
├── src/                        # main FEM code
│   ├── examples-edited/        # simplified MFEM examples
│   │   └── ex1simple.cpp       
│   ├── examples-mfem/          # original MFEM examples
│   │   └── ex1.cpp         
│   ├── CMakeLists.txt          # for build process
│   ├── curlcurl.cpp
│   ├── dirichlet-cons.cpp      # main simulation
│   ├── dirichlet-conv.cpp      # main simulation
│   └── utils.cpp               # helper functions
├── symbolic_math/              # mathematica code
├── build.sh                    
├── clean.sh                    
├── compile.sh
├── LICENSE
└── README.md
```

## run the scheme:

1. select the cpp file in src/CMakeLists.txt :
    * `periodic-conv.cpp`
    * `periodic-cons.cpp`
    * `dirichlet-[placeholder].cpp`
2. run ./build.sh
3. run ./compilerun.sh
4. produce a visualization using `scripts/*.ipynb`
5. find the visualizsations in out/plots
6. find the raw data in out/rawdata

## thesis PDF

see either of the following links

- https://people.math.ethz.ch/~hiptmair/StudentProjects/Renoldner.Markus/MScThesis.pdf
- https://repositum.tuwien.at/handle/20.500.12708/177634
- https://doi.org/10.34726/hss.2023.110820
