#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// primal field discrete weak form:
// A * x = b
// [N+R   C   -D^T] [u]   [(N-R)*u - C*w + f]
// [C^T   -M    0 ] [w] = [        0        ]
// [D     0     0 ] [p]   [        0        ]
// 
// N = (sig_j , sig_i)
// R = (zeta x sig_j , sig_i)
// C = ...
// D
// M
// 


int main(int argc, char *argv[]) {

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/ref-square.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // FE spaces
    int order = 1;
    fec_H1 = new H1_FECollection(order, dim);
    fec_ND = new ND_FECollection(order, dim);
    fec_RT = new RT_FECollection(order, dim);
    fec_DG = new L2_FECollection(order, dim);
    FiniteElementSpace H1(&mesh, fec_H1);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace DG(&mesh, fec_DG);

    // unkowns and gridfunctions
    mfem::VectorGridFunction u(&ND);
    mfem::VectorGridFunction w(&RT);
    mfem::Gridfunction       p(&DG);

    mfem::VectorGridfunction v(&ND); // dual velocity
    mfem::VectorGridfunction zeta(&ND); // dual vorticity
    mfem::Gridfunction       q(&ND); // dual pressure

    // initial data
    // p = 0;
    p.ProjectCoefficient(0);
    // passt das?

    // boundary conditions
    mfem::Array<int> ess_tdof_list;
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    DG->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);    

    // RHS
    LinearForm b1(&H1);
    LinearForm b2(&ND);
    LinearForm b3(&RT);
    LinearForm b4(&DG);

    // Matrix M
    mfem::BilinearForm blf_M(RT);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ess_tdof_list,M);

    // Matrix N
    mfem::BilinearForm blf_N(ND);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(ess_tdof_list,N);

    // Matrix C
    mfem::MixedBilinearForm blf_C(ND, RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormSystemMatrix(ess_tdof_list,C);

    // Matrix D 
    // MixedScalarDivergenceIntegrator
    
    // Matrix G
    // MixedVectorGradientIntegrator

    // Matrix R1
    mfem::BilinearForm blf_R(ND);
    mfem::SparseMatrix R;
    mfem::VectorGridFuncCoeff zeta_gfcoeff(&zeta);
    blf_R.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(zeta_gfcoeff));
    blf_R.Assemble();
    blf_R.FormSystemMatrix(ess_tdof_list,R);

    // Matrix R2
    // MixedCrossProductIntegrator

    // big matrix
    mfem::SparseMatrix A;




    // assembly
    // line 266:
    // https://gitlab.com/WouterTonnon/semi-lagrangian-tools/-/blob/master/apps/Euler_1stOrder_NonCons.cpp
    // a.Assemble();
    // OperatorPtr A;
    // Vector B, X;
    // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
    // std::cout << "Size of linear system: " << A->Height() << std::endl;

    // solve
    // GSSmoother M((SparseMatrix&)(*A));
    // PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);

    // visuals
    // ofstream mesh_ofs("refined.mesh");
    // mesh_ofs.precision(8);
    // mesh.Print(mesh_ofs);
    // ofstream sol_ofs("sol.gf");
    // sol_ofs.precision(8);
    // x.Save(sol_ofs);

    // char vishost[] = "localhost";
    // int  visport   = 19916;
    // socketstream sol_sock(vishost, visport);
    // sol_sock.precision(8);
    // sol_sock << "solution\n" << mesh << x << flush;


    delete fec_H1;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;
}