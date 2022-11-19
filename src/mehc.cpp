#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// primal: A1*x=b1
// [M+Rd   C^T    G] [u]   [(M-Rd)*u - C^T*z + f]
// [C      -N      ] [w] = [          0         ]
// [G^T            ] [p]   [          0         ]
//
// dual: A2*y=b2
// [N+Rp   C   -D^T] [v]   [(N-Rp)*u - C*w + f]
// [C^T    -M    0 ] [z] = [         0        ]
// [D      0     0 ] [q]   [         0        ]


// TODO put in separate file
// TODO use SetSubMatrix() oder SetSubMatrixTranspose()
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

    std::cout << "---------------launch MEHC---------------\n";

    // mesh
    // const char *mesh_file = "extern/mfem-4.5/data/ref-square.mesh";
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    // for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // FE spaces
    // TODO call H1 CG and call L2 DG ?
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
    // TODO implement the ND0 and RT0 spaces

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
    // TODO proper BCs for the 6 spaces defined before
    mfem::Array<int> H1_etdof, ND_etdof, RT_etdof, DG_etdof; // "essential true degrees of freedom"
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    H1.GetEssentialTrueDofs(ess_bdr, H1_etdof);
    ND.GetEssentialTrueDofs(ess_bdr, ND_etdof);
    RT.GetEssentialTrueDofs(ess_bdr, RT_etdof);
    DG.GetEssentialTrueDofs(ess_bdr, DG_etdof);
    // TODO now use e.g. H1_etdof to set the essential boundary conditions
    // usually mfem has 1 syst matrix, here we have lots of submatrices => maybe BC are difficult to implement

    // Matrix M and -M
    // TODO: M sollte ND functions haben und N sollte RT haben
    // TODO: transpose all submatrices basically, siehe transpose.cpp oder doch nicht?
    // TODO: warum hat R so große einträge - feasiblity check von matrix einträgen
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(RT_etdof,M); // TODO formlinearsystem ? maybe not
    mfem::SparseMatrix Mn = M;
    Mn *= -1;

    // Matrix N and -N
    mfem::BilinearForm blf_N(&RT);
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
    
    // std::cout << "C:  " <<C.NumRows() <<" "<< C.NumCols() << "\n"; 
    // mfem::SparseMatrix CT(C.NumCols(), C.NumRows());
    // mfem::Transpose(CT);
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
    std::cout << "progress: M,N,C,D,G assembled\n";

    // size of primal and dual system (nr of unknowns)
    // TODO implement correct size
    int size_p = M.NumRows() + CT.NumRows() + G.NumRows();
    int size_d = N.NumRows() + CT.NumRows() + D.NumRows();
    int size_p_T = M.NumCols() + C.NumCols() + GT.NumCols();
    int size_d_T = N.NumCols() + C.NumCols() + DT.NumCols();

    std::cout << "size of primal system: " << size_p<<"\n";
    std::cout << "size of dual system:   " << size_d<<"\n";
    std::cout << "pt " << size_p_T<<"\n";
    std::cout << "dt " << size_d_T<<"\n";

    std::cout << "M:  " <<M.NumRows()  <<" "<< M.NumCols() << "\n"; 
    std::cout << "CT: " <<CT.NumRows() <<" "<< CT.NumCols() << "\n"; 
    std::cout << "G:  " <<G.NumRows()  <<" "<< G.NumCols() << "\n";
    std::cout << "N:  " <<N.NumRows()  <<" "<< N.NumCols() << "\n"; 
    std::cout << "C:  " <<C.NumRows()  <<" "<< C.NumCols() << "\n"; 
    std::cout << "DT: " <<DT.NumRows() <<" "<< DT.NumCols() << "\n"; 

    // initialize right hand side matrices and vectors
    mfem::Vector b1(size_p);
    mfem::Vector b2(size_d);
    mfem::Vector b1sub(M.NumRows());
    mfem::Vector b2sub(N.NumRows());
    mfem::Vector Mu(M.NumRows());
    mfem::Vector Rdu(M.NumRows());
    mfem::Vector CTz(M.NumRows());
    mfem::Vector Nu(N.NumRows());
    mfem::Vector Rpu(M.NumRows());
    mfem::Vector Cw(M.NumRows());

    // indices of unkowns in solution vectors x y
    mfem::Array<int> u_dofs;
    mfem::Array<int> w_dofs;
    mfem::Array<int> p_dofs;
    mfem::Array<int> v_dofs;
    mfem::Array<int> z_dofs;
    mfem::Array<int> q_dofs;
    for (int k = 0; k < M.NumCols(); ++k)                        { u_dofs.Append(k); }
    for (int k = M.NumCols(); k < M.NumCols()+CT.NumCols(); ++k) { w_dofs.Append(k); }
    for (int k = M.NumCols()+CT.NumCols(); k < size_p; ++k)      { p_dofs.Append(k); }
    for (int k = 0; k < N.NumCols(); ++k)                        { v_dofs.Append(k); }
    for (int k = N.NumCols(); k < N.NumCols()+C.NumCols(); ++k)  { z_dofs.Append(k); }
    for (int k = N.NumCols()+C.NumCols(); k < size_d; ++k)       { q_dofs.Append(k); }

    // solution vectors x and y
    mfem::Vector x(size_p);
    mfem::Vector y(size_d);
    x = 1.5;
    y = 1.;
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(w_dofs, w);
    x.GetSubVector(p_dofs, p);
    y.GetSubVector(v_dofs, v);
    y.GetSubVector(z_dofs, z);
    y.GetSubVector(q_dofs, q);

    // solve system in each time step
    std::cout << "---------------enter loop---------------\n";
    int T = 1;
    for (int t = 0 ; t < T ; t++) { 
        
        // Matrix Rp (for primal field)
        // TODO: Rp and Rd in 2D
        mfem::BilinearForm blf_Rp(&RT);
        mfem::SparseMatrix Rp;
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
        blf_Rp.Assemble();
        blf_Rp.FormSystemMatrix(RT_etdof,Rp);
        
        // Matrix Rd (for dual field)
        mfem::BilinearForm blf_Rd(&ND);
        mfem::SparseMatrix Rd;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
        blf_Rd.Assemble();
        blf_Rd.FormSystemMatrix(ND_etdof,Rd);
        std::cout << "progress: Rp,Rd assembled\n";
        
        // assemble A1, A2
        // TODO add constant (?) factors (reynolds, dt, ...)
        // TODO überschreibt Rp und Rd die matrix M? same für N
        mfem::SparseMatrix A1(size_p);
        mfem::SparseMatrix A2(size_d);
        AddSubmatrix(M,  A1, 0, 0); // submatrix, matrix, rowoffset, coloffset
        AddSubmatrix(Rp, A1, 0, 0); 
        AddSubmatrix(CT, A1, 0, M.NumCols());
        AddSubmatrix(G,  A1, 0, M.NumCols() + CT.NumCols());
        AddSubmatrix(C,  A1, M.NumRows(), 0);
        AddSubmatrix(GT, A1, M.NumRows() + C.NumRows(), 0);
        AddSubmatrix(Nn, A1, M.NumRows(), M.NumCols());
        A1.Finalize();
        AddSubmatrix(N,  A2, 0, 0);
        AddSubmatrix(Rd, A2, 0, 0);
        AddSubmatrix(C,  A2, 0, N.NumCols());
        AddSubmatrix(DT, A2, 0, N.NumCols() + C.NumCols());
        AddSubmatrix(CT, A2, N.NumRows(), 0);
        AddSubmatrix(D,  A2, N.NumRows() + CT.NumRows(), 0);
        AddSubmatrix(Mn, A2, N.NumCols(), N.NumRows());
        A2.Finalize();
        std::cout << "progress: A1,A2 assembled\n";

        // check some matrix dimensions
        // std::cout << A1.NumCols() << "\n";
        // int a = M.NumCols() + CT.NumCols() + G.NumCols();
        // int b = M.NumRows() + C.NumRows() + GT.NumRows();
        // std::cout << a << " "<<b<<"\n";
        // std::cout << CT.NumCols() << "\n";
        // std::cout << C.NumCols() << "\n";
        // std::cout << size_p<< "\n";
        // std::cout << M.NumCols() << "\n";
        // std::cout << CT.NumCols() << "\n";
        // std::cout << G.NumCols() << "\n";
        // std::cout << "---"<<Rd.GetRowColumns(4)[4] << "\n";
        // std::cout << "---"<<Rd.GetRowColumns(10)[35] << "\n";
        // std::cout << "---"<<Rp.GetRowColumns(10)[35] << "\n";

        // assemble b1, b2
        // TODO feasibility check
        M.Mult (u,Mu);
        Rd.Mult(u,Rdu);
        CT.Mult(z,CTz);
        b1sub = Mu;
        b1sub += Rdu;
        b1sub += CTz;
        for (int j = 0; j < N.NumRows(); j++) {b1.Elem(j) = b1sub.Elem(j);}
        N.Mult (u,Nu);
        Rp.Mult(u,Rpu);
        CT.Mult(w,Cw);
        b2sub = Nu;
        b2sub += Rpu;
        b2sub += Cw;
        for (int j = 0; j < N.NumRows(); j++) {b2.Elem(j) = b2sub.Elem(j);}
        std::cout << "progress: b1,b2 assembled\n";

        // check solution before solver
        // std::cout << "---"<<x[1] << "\n";
        // std::cout << "---"<<u[1] << "\n";
        // std::cout << "---"<<x.Elem(1) << "\n";
        // std::cout << "---"<<u.Elem(1) << "\n";
        
        // solve system
        double tol = 1e-6;
        int iter = 10000;
        mfem::MINRES(A1, b1, x, 0, iter, tol, tol); // primal: unkonwns at half integer time steps
        mfem::MINRES(A2, b2, y, 0, iter, tol, tol); // dual:   unkonwns at integer      time steps
        std::cout << "progress: MINRES\n";
        
        // extract solution values u,w,p,v,z,q from x,y
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(w_dofs, w);
        x.GetSubVector(p_dofs, p);
        y.GetSubVector(v_dofs, v);
        y.GetSubVector(z_dofs, z);
        y.GetSubVector(q_dofs, q);

        // check solution after solver
        // std::cout << "---"<<x[1] << "\n";
        // std::cout << "---"<<u[1] << "\n";
        // std::cout << "---"<<x.Elem(1) << "\n";
        // std::cout << "---"<<u.Elem(1) << "\n";
    }

    // visuals
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh.Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    u.Save(sol_ofs);

    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << u << flush;
    

    delete fec_H1;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;
    
    std::cout << "---------------finish MEHC---------------\n";
}