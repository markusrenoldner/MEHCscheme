#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// primal field at half integer time steps:
// A1 * x = b1
// [N+Rh   C   -D^T] [u]   [(N-Rh)*u - C*w + f]
// [C^T    -M    0 ] [w] = [         0        ]
// [D      0     0 ] [p]   [         0        ]
//
// dual field at integer time steps:
// A2 * y = b2
// [M+R   C^T    G] [v   ]   [(M-R)*u - C^T*w + f]
// [C     -N      ] [zeta] = [         0         ]
// [G^T           ] [q   ]   [         0         ]
// 
// submatrices:
// M = (tau_j , tau_i)
// N = (sig_j , sig_i)
// Rh = (zeta x sig_j , sig_i)
// R = (w x sig_j , sig_i)
// C = (nabla x tau_j , sig_i)
// G = (nabla gamma_j, tau_i)
// D = (nabla . sig_j , xi_i)


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

    // initial data TODO
    // p = 0;
    // p.ProjectCoefficient(0);
    // passt das?

    // boundary conditions
    // TODO proper BCs
    mfem::Array<int> H1_etdof, ND_etdof, RT_etdof, DG_etdof; // "essential true degrees of freedom"
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    H1.GetEssentialTrueDofs(ess_bdr, H1_etdof);
    ND.GetEssentialTrueDofs(ess_bdr, ND_etdof);
    RT.GetEssentialTrueDofs(ess_bdr, RT_etdof);
    DG.GetEssentialTrueDofs(ess_bdr, DG_etdof);

    // Matrix M
    mfem::BilinearForm blf_M(&RT);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(RT_etdof,M);

    // Matrix N
    mfem::BilinearForm blf_N(&ND);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(ND_etdof,N);

    // Matrix C
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);

    // Matrix D
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    mfem::SparseMatrix D;
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_D.Assemble();
    blf_D.FormRectangularSystemMatrix(RT_etdof,DG_etdof,D);
    std::cout << "---------------check1---------------\n";
    
    // Matrix G TODO
    // mfem::MixedBilinearForm blf_G(&ND, &H1);
    // mfem::SparseMatrix G;
    // blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    // blf_G.Assemble();
    // blf_G.FormRectangularSystemMatrix(ND_etdof,H1_etdof,G);
    std::cout << "---------------check2---------------\n";

    // Matrix R TODO
    // mfem::BilinearForm blf_R(&RT);
    // mfem::SparseMatrix R;
    // mfem::VectorGridFunctionCoefficient zeta_gfcoeff(&zeta);
    // blf_R.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(zeta_gfcoeff));
    // blf_R.Assemble();
    // blf_R.FormSystemMatrix(RT_etdof,R);
    std::cout << "---------------check3---------------\n";
    

    // Matrix Rh (h ... half integer time step) TODO
    // mfem::BilinearForm blf_Rh(&ND);
    // mfem::SparseMatrix Rh;
    // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
    // blf_Rh.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
    // blf_Rh.Assemble();
    // blf_Rh.FormSystemMatrix(ND_etdof,Rh);
    std::cout << "---------------check4---------------\n";

    // right hand side vector (A1*x=b1) TODO
    int systemsize = N.NumRows() + C.NumRows() + D.NumRows();
    mfem::Vector b1(systemsize);
    mfem::Vector b2(systemsize);
    // b1.AddSubVector(subvector,offset);
    // b2.AddSubVector(subvector,offset);

    // assemble big matrices
    // TODO add constant factors (reynolds, dt, ...)
    // TODO transpose C and D
    mfem::SparseMatrix A1(systemsize);
    mfem::SparseMatrix A2(systemsize);

        for (int r = 0; r < N.NumRows(); r++) { // rows of N or Rh

        mfem::Array<int> cols;
        mfem::Vector srow;
        N.GetRow(r, cols, srow);
        for (int c = 0; c < N.NumCols(); c++) { // cols of N
            A1.Add(r, cols[c], srow[c]); // add cols of N to A1
        }

        // cols.DeleteAll();
        // Rh.GetRow(r, cols, srow);
        // for (int c = 0; c < N.NumCols(); c++) { 
        //     A1.Add(r, cols[c], srow[c]);
        // }
    }
    std::cout << "---------------check6---------------\n";

    for (int r = 0; r < C.NumRows(); r++) {
        
        mfem::Array<int> cols;
        mfem::Vector srow;
        C.GetRow(r, cols, srow);

        // for ( ... ) { 
        //     A1.Add( ... );
        // }
    }
    std::cout << "---------------check7---------------\n";
    





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