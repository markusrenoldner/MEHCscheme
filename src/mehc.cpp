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
    FiniteElementCollection *fec_H1, *fec_ND, *fec_RT, *fec_DG;
    fec_H1 = new H1_FECollection(order, dim);
    fec_ND = new ND_FECollection(order, dim);
    fec_RT = new RT_FECollection(order, dim);
    fec_DG = new L2_FECollection(order, dim);
    FiniteElementSpace H1(&mesh, fec_H1);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace DG(&mesh, fec_DG);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND);
    mfem::GridFunction w(&RT);
    mfem::GridFunction p(&DG);
    mfem::GridFunction v(&ND); // dual velocity
    mfem::GridFunction zeta(&ND); // dual vorticity
    mfem::GridFunction q(&ND); // dual pressure

    // initial data
    // p = 0;
    // p.ProjectCoefficient(0);
    // passt das?

    // boundary conditions
    mfem::Array<int> H1_esstdof, ND_esstdof, RT_esstdof, DG_esstdof;
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    H1.GetEssentialTrueDofs(ess_bdr, H1_esstdof);
    ND.GetEssentialTrueDofs(ess_bdr, ND_esstdof);
    RT.GetEssentialTrueDofs(ess_bdr, RT_esstdof);
    DG.GetEssentialTrueDofs(ess_bdr, DG_esstdof);

    // RHS
    // TODO
    LinearForm b1(&H1);
    LinearForm b2(&ND);
    LinearForm b3(&RT);
    LinearForm b4(&DG);

    // Matrix M
    mfem::BilinearForm blf_M(&RT);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(RT_esstdof,M);

    // Matrix N
    mfem::BilinearForm blf_N(&ND);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(ND_esstdof,N);

    // Matrix C
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_esstdof,RT_esstdof,C);

    // Matrix D
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    mfem::SparseMatrix D;
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_D.Assemble();
    blf_D.FormRectangularSystemMatrix(RT_esstdof,DG_esstdof,D);
    
    std::cout << "hi1----------------------------\n";
    // Matrix G
    // mfem::MixedBilinearForm blf_G(&ND, &H1);
    // mfem::SparseMatrix G;
    // blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    // blf_G.Assemble();
    // blf_G.FormRectangularSystemMatrix(ND_esstdof,H1_esstdof,G);

    std::cout << "hi2----------------------------\n";
    // Matrix R
    // mfem::BilinearForm blf_R(&RT);
    // mfem::SparseMatrix R;
    // mfem::VectorGridFunctionCoefficient zeta_gfcoeff(&zeta);
    // blf_R.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(zeta_gfcoeff));
    // blf_R.Assemble();
    // blf_R.FormSystemMatrix(RT_esstdof,R);
    
    std::cout << "hi3----------------------------\n";

    // Matrix R_h (h ... half integer time step)
    // mfem::BilinearForm blf_R_h(&ND);
    // mfem::SparseMatrix R_h;
    // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
    // blf_R_h.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
    // blf_R_h.Assemble();
    // blf_R_h.FormSystemMatrix(ND_esstdof,R_h);
    
    std::cout << "hi4----------------------------\n";




    // big matrix
    mfem::SparseMatrix A;

    std::cout << "hi5----------------------------\n";



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