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




void printmatrix(mfem::Matrix &mat);
void printvector(mfem::Vector vec, int stride=1);



int main(int argc, char *argv[]) {

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // FE spaces
    int order = 1;
    FiniteElementCollection *fec_RT = new RT_FECollection(order, dim);
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_CG  = new H1_FECollection(order, dim);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace CG(&mesh, fec_CG);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND);
    mfem::GridFunction z(&RT); 
    mfem::GridFunction p(&CG);

    // boundary conditions
    mfem::Array<int> RT_etdof;
    mfem::Array<int> ND_etdof;
    mfem::Array<int> CG_etdof;

    // [M+R    CT     G]
    // [C      -N      ]
    // [GT             ] 

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

    // Matrix G and GT
    mfem::MixedBilinearForm blf_G(&CG, &ND);
    mfem::SparseMatrix G;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(CG_etdof,ND_etdof,G);
    G.Finalize();
    mfem::SparseMatrix *GT = Transpose(G);
    GT->Finalize();
    std::cout << "progress: assembled M,N,C,D,G\n";

    // update R1
    mfem::BilinearForm blf_Rp(&ND);
    mfem::SparseMatrix Rp;
    mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
    blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
    blf_Rp.Assemble();
    blf_Rp.FormSystemMatrix(ND_etdof,Rp);
    Rp.Add(1,M);
    Rp.Finalize();
    std::cout << "progress: updated Rp\n";


    // initialize system matrices
    Array<int> offsets_1 (4); // number of variables + 1
    offsets_1[0] = 0;
    offsets_1[1] = ND.GetVSize();
    offsets_1[2] = RT.GetVSize();
    offsets_1[3] = CG.GetVSize();
    offsets_1.PartialSum();

    // assemble blockmatrix A
    int size_p = M.NumCols() + CT->NumCols();
    Array<int> block_offsets (3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = ND.GetVSize();
    block_offsets[2] = RT.GetVSize();
    block_offsets.PartialSum(); // =exclusive scan
    mfem::BlockMatrix A(block_offsets);

    // update A1, A2
    A.SetBlock(0,0, &Rp); // includes M
    A.SetBlock(0,1, CT);
    A.SetBlock(1,2, &G);
    A.SetBlock(1,0, &C);
    A.SetBlock(1,1, &Nn);
    A.SetBlock(2,1, GT);

    // unknown dofs
    mfem::Array<int> u_dofs (M.NumCols());
    mfem::Array<int> z_dofs (CT->NumCols());
    std::iota(&u_dofs[0], &u_dofs[M.NumCols()],0);
    std::iota(&z_dofs[0], &z_dofs[CT->NumCols()],M.NumCols());

    // unknown vector
    mfem::Vector x(size_p); x = 1.5;
    // mfem::GridFunction u(&ND);
    // mfem::GridFunction z(&RT); 
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