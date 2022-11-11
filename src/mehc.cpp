#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;





// TODO put in separate file
void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix, int rowoffset, int coloffset) {
    for (int r = 0; r < submatrix.NumRows(); r++) {
        mfem::Array<int> cols;
        mfem::Vector srow;
        submatrix.GetRow(r, cols, srow);
        for (int c = 0; c < submatrix.NumCols(); c++) {
            matrix.Add(rowoffset + r, coloffset + cols[c], srow[c]);
        }
        cols.DeleteAll();
    }
}


int main(int argc, char *argv[]) {

    // mesh
    // const char *mesh_file = "extern/mfem-4.5/data/ref-square.mesh";
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
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
    mfem::GridFunction z(&ND); // dual vorticity (z...zeta)
    mfem::GridFunction q(&ND); // dual pressure
    
    // initial data
    // TODO 
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

    // Matrix M and -M
    mfem::BilinearForm blf_M(&RT);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(RT_etdof,M);
    mfem::SparseMatrix Mn = M;
    Mn *= -1;

    // Matrix N and -N
    mfem::BilinearForm blf_N(&ND);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(ND_etdof,N);
    mfem::SparseMatrix Nn = N;
    Nn *= -1;

    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    // TODO: is the transposition done correctly?
    mfem::SparseMatrix CT = C;
    mfem::Transpose(CT);
    // CT = *(mfem::Transpose(CT));
    // mfem::SparseMatrix CT = *(mfem::Transpose(C)); // dereferenced

    // Matrix D and DT
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    mfem::SparseMatrix D;
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_D.Assemble();
    blf_D.FormRectangularSystemMatrix(RT_etdof,DG_etdof,D);
    mfem::SparseMatrix DT = D;
    mfem::Transpose(DT);
    
    // Matrix G and GT
    mfem::MixedBilinearForm blf_G(&H1, &ND);
    mfem::SparseMatrix G;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    // blf_G.AddDomainIntegrator(new mfem::GradientIntegrator()); // equivalent?
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(ND_etdof,H1_etdof,G);
    mfem::SparseMatrix GT = G;
    mfem::Transpose(GT);

    // Matrix Rp (for primal field)
    // TODO: in 2D
    mfem::BilinearForm blf_Rp(&RT);
    mfem::SparseMatrix Rp;
    mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
    blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
    blf_Rp.Assemble();
    blf_Rp.FormSystemMatrix(RT_etdof,Rp);
    std::cout << "---------------check1---------------\n";
    
    // Matrix Rd (for dual field)
    // TODO: in 2D
    mfem::BilinearForm blf_Rd(&ND);
    mfem::SparseMatrix Rd;
    mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
    blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
    blf_Rd.Assemble();
    blf_Rd.FormSystemMatrix(ND_etdof,Rd);
    std::cout << "---------------check2---------------\n";

    // primal: A1*x=b1
    // [M+Rd   C^T    G] [u]   [(M-Rd)*u - C^T*z + f]
    // [C      -N      ] [w] = [          0         ]
    // [G^T            ] [p]   [          0         ]
    //
    // dual: A2*y=b2
    // [N+Rp   C   -D^T] [v]   [(N-Rp)*u - C*w + f]
    // [C^T    -M    0 ] [z] = [         0        ]
    // [D      0     0 ] [q]   [         0        ]
    
    // solution vectors x and y
    int systemsize = M.NumRows() + CT.NumRows() + G.NumRows();
    // int systemsize = N.NumRows() + C.NumRows() + D.NumRows(); // should be equal
    mfem::Vector x(systemsize);
    mfem::Vector y(systemsize);

    // assemble right hand side vectors b1 and b2
    mfem::Vector b1(systemsize);
    mfem::Vector b2(systemsize);
    mfem::Vector b1sub(M.NumRows());
    mfem::Vector b2sub(N.NumRows());
    mfem::Vector Mu(M.NumRows());
    mfem::Vector Rdu(M.NumRows());
    mfem::Vector CTz(M.NumRows());
    mfem::Vector Nu(N.NumRows());
    mfem::Vector Rpu(M.NumRows());
    mfem::Vector Cw(M.NumRows());
    std::cout << "---------------check3---------------\n";

    // TODO: extract u,w,p,v,z,q from x and y correctly and e.g. multiply like this: N.Mult(u,Nu)
    M.Mult (y,Mu);
    Rd.Mult(y,Rdu);
    CT.Mult(y,CTz);
    b1sub += Mu;
    b1sub += Rdu;
    b1sub += CTz;
    for (int j = 0; j < N.NumRows(); j++) {b1.Elem(j) = b1sub.Elem(j);}
    N.Mult(x,Nu);
    Rp.Mult(x,Rpu);
    CT.Mult(x,Cw);
    b2sub += Nu;
    b2sub += Rpu;
    b2sub += Cw;
    for (int j = 0; j < N.NumRows(); j++) {b2.Elem(j) = b2sub.Elem(j);}
    std::cout << "---------------check4---------------\n";


    // assemble big matrices A1 and A2
    // TODO add constant factors (reynolds, dt, ...)
    mfem::SparseMatrix A1(systemsize);
    mfem::SparseMatrix A2(systemsize);
    AddSubmatrix(M,  A1, 0, 0); // submatrix, matrix, rowoffset, coloffset
    AddSubmatrix(Rp, A1, 0, 0);
    AddSubmatrix(CT, A1, 0, M.NumCols());
    AddSubmatrix(G,  A1, 0, M.NumCols() + CT.NumCols());
    AddSubmatrix(C,  A1, M.NumRows(), 0);
    AddSubmatrix(GT, A1, M.NumRows() + C.NumRows(), 0);
    AddSubmatrix(Nn, A1, M.NumRows(), M.NumCols());
    A1.Finalize();
    std::cout << "---------------check5---------------\n";

    AddSubmatrix(N,  A2, 0, 0);
    AddSubmatrix(Rd, A2, 0, 0);
    AddSubmatrix(C,  A2, 0, N.NumCols());
    AddSubmatrix(DT, A2, 0, N.NumCols() + C.NumCols());
    AddSubmatrix(CT, A2, N.NumRows(), 0);
    AddSubmatrix(D,  A2, N.NumRows() + CT.NumRows(), 0);
    AddSubmatrix(Mn, A2, N.NumCols(), N.NumRows());
    A2.Finalize();
    std::cout << "---------------check6---------------\n";


    // solve
    // x = 0.;
    // TODO solve in loop
    // TODO reassemble Rp and Rd inside the loop and reassemble A1, A2
    // for (int t = 0 ; t < T ; t++) { ... }
    // GSSmoother M((SparseMatrix&)(*A));
    // PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);

    // wouter
    double tol = 1e-12;
    mfem::MINRES(A1, b1, x, 0, 10, tol * tol, tol * tol);

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
    
    std::cout << "---------------finished---------------\n";
}