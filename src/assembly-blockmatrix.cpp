#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace mfem;


// assembly options:
// ?? 1) sparse submatrices (wie wouter)
// ?? 2) dense submatrices
// -- 3) blockmatrix (maybe works like blockoperator?)
// ok 4) blockoperator (based on mfem example 5)


// variable names lower case and underscores: vel_vec
// function names upper case without underscores: AddMatrix


void PrintMatrix(mfem::Matrix &mat);
void PrintVector(mfem::Vector vec, int stride=1);



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

    
    std::cout << M.NumCols() << "\n";
    
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

    // assemble blockmatrix A
    int size_p = M.NumCols() + CT->NumCols();
    Array<int> block_offsets (3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = ND.GetVSize();
    block_offsets[2] = RT.GetVSize();
    block_offsets.PartialSum(); // =exclusive scan
    mfem::BlockMatrix A(block_offsets);
    A.SetBlock(0,0, &M);
    A.SetBlock(0,1, CT);
    A.SetBlock(1,0, &C);
    A.SetBlock(1,1, &Nn);

    // unknown dofs
    mfem::Array<int> u_dofs (M.NumCols());
    mfem::Array<int> z_dofs (CT->NumCols());
    std::iota(&u_dofs[0], &u_dofs[M.NumCols()],0);
    std::iota(&z_dofs[0], &z_dofs[CT->NumCols()],M.NumCols());

    // unknown vector
    mfem::Vector x(size_p); x = 1.5;
    mfem::GridFunction u(&ND);
    mfem::GridFunction z(&RT); 
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(z_dofs, z);

    // rhs
    mfem::Vector b(size_p); b = 0.0;
    mfem::Vector bsubv(ND.GetVSize()); bsubv = 0.0;
    M.AddMult(u,bsubv);
    CT->AddMult(z,bsubv,-1);
    b.AddSubVector(bsubv,0);
    
    // MINRES
    int iter = 2000;
    int tol = 1e-4;
    mfem::MINRES(A, b, x, 0, iter, tol, tol);

    // extract subvectors
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(z_dofs, z);

    // check 
    // mfem::Vector zz(size_p);
    // A.Mult(x,zz);
    // printvector(zz,1);

    delete fec_RT;
    delete fec_ND;
}