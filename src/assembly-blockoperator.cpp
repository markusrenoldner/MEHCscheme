#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// assembly options:
// 1) submatrices sind sparse und wie wouter
// 2) submatrices sind dense
// 3) blockmatrix (maybe works like blockoperator?)
// 4) blockoperator based on mfem example 5



int main(int argc, char *argv[]) {

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // FE spaces
    int order = 1;
    FiniteElementCollection *fec_RT = new RT_FECollection(order, dim);
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);

    // boundary conditions
    mfem::Array<int> RT_etdof;
    mfem::Array<int> ND_etdof;

    // Matrix M and -M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ND_etdof,M);
    M.Finalize();
    M*=100;
    
    // Matrix Nn
    mfem::BilinearForm blf_N(&RT);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(RT_etdof,N);
    N.Finalize();
    mfem::SparseMatrix Nn = N;
    Nn *= -1;
    Nn.Finalize();

    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    C.Finalize();
    mfem::SparseMatrix *CT = Transpose(C);
    CT->Finalize();

    // assemble blockoperator A
    int size_p = M.NumCols() + CT->NumCols();
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = ND.GetVSize();
    block_offsets[2] = RT.GetVSize();
    block_offsets.PartialSum();
    BlockOperator A(block_offsets);
    A.SetBlock(0,0, &M);
    A.SetBlock(0,1, CT);
    A.SetBlock(1,0, &C);
    A.SetBlock(1,1, &Nn);

    // rhs
    mfem::Vector b(size_p);
    mfem::Vector x(size_p);
    x = 0.924;
    A.Mult(x,b); // set b=A*x
    x = -1.1;
    
    // MINRES
    int iter = 200;
    int tol = 1e-4;
    mfem::MINRES(A, b, x, 0, iter, tol, tol); 

    // check if x=1
    for (int j = 0; j<size_p; j++) {
        std::cout << std::setprecision(1) << std::fixed;
        std::cout << x[j]<< "\n";
    }

    delete fec_RT;
    delete fec_ND;
}