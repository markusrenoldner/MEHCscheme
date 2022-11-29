#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// TODO: umordnen der unbekannten, z ist primal, w ist dual

// primal: A1*x=b1
// [M+R    CT     G] [u]   [(M-R)*u - CT*z  + f]
// [C      -N      ] [w] = [         0         ]
// [GT             ] [p]   [         0         ]
//
// dual: A2*y=b2
// [N+R    C    -DT] [v]   [(N-R)*u - C*w + f]
// [CT     -M     0] [z] = [        0        ]
// [D      0      0] [q]   [        0        ]



// TODO put in separate file
// TODO use SetSubMatrix() oder SetSubMatrixTransp
// TODO use M.Add(1, Rp) and then addsubmatrix(M)ose()
// TODO check again using wouters file
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
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    // for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // FE spaces (CG \in H1, DG \in L2)
    int order = 1;
    FiniteElementCollection *fec_CG  = new H1_FECollection(order, dim);
    FiniteElementCollection *fec_ND  = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_RT  = new RT_FECollection(order, dim);
    FiniteElementCollection *fec_DG  = new L2_FECollection(order, dim);
    FiniteElementCollection *fec_ND0 = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_RT0 = new RT_FECollection(order, dim);
    FiniteElementSpace CG(&mesh, fec_CG);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace DG(&mesh, fec_DG);
    FiniteElementSpace ND0(&mesh, fec_ND0);
    FiniteElementSpace RT0(&mesh, fec_RT0);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND);
    mfem::GridFunction w(&RT);
    mfem::GridFunction p(&DG);
    mfem::GridFunction v(&ND0); 
    mfem::GridFunction z(&RT0); 
    mfem::GridFunction q(&CG);  
    
    // initial data
    // TODO 
    // p = 0;
    // p.ProjectCoefficient(0);

    // boundary conditions
    // TODO proper BCs
    // TODO now use e.g. H1_etdof to set the essential boundary conditions
    // TODO usually mfem has 1 syst matrix, here we have lots of 
    // submatrices => maybe BC are difficult to implement
    mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof, ND0_etdof, RT0_etdof;
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    CG.GetEssentialTrueDofs(ess_bdr, CG_etdof);
    ND.GetEssentialTrueDofs(ess_bdr, ND_etdof);
    RT.GetEssentialTrueDofs(ess_bdr, RT_etdof);
    DG.GetEssentialTrueDofs(ess_bdr, DG_etdof);
    ND0.GetEssentialTrueDofs(ess_bdr, ND0_etdof);
    RT0.GetEssentialTrueDofs(ess_bdr, RT0_etdof);
    std::cout << "progress: assembled BCs" << "\n";

    // Matrix M and -M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ND_etdof,M);
    M.Finalize();
    mfem::SparseMatrix Mn = M;
    Mn *= -1;
    Mn.Finalize();

    // Matrix Md and Mdn
    mfem::BilinearForm blf_Md(&ND0);
    mfem::SparseMatrix Md;
    blf_Md.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_Md.Assemble();
    blf_Md.FormSystemMatrix(ND0_etdof,Md); 
    //TODO FormLinearSystem... sets BC to right value, might only work for 1 single matrix
    //TODO FormSystemMatrix... does not set the BC to 0.. maybe? 
    Md.Finalize();
    mfem::SparseMatrix Mdn = Md;
    Mdn *= -1;
    Mdn.Finalize();

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

    // Matrix Nd
    mfem::BilinearForm blf_Nd(&RT0);
    mfem::SparseMatrix Nd;
    blf_Nd.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_Nd.Assemble();
    blf_Nd.FormSystemMatrix(RT0_etdof,Nd);
    Nd.Finalize();
    
    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    C.Finalize();
    mfem::SparseMatrix *CT = Transpose(C);
    CT->Finalize();

    // Matrix Cd and CdT
    mfem::MixedBilinearForm blf_Cd(&ND0, &RT0);
    mfem::SparseMatrix Cd;
    blf_Cd.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_Cd.Assemble();
    blf_Cd.FormRectangularSystemMatrix(ND0_etdof,RT0_etdof,Cd);
    Cd.Finalize();
    mfem::SparseMatrix *CdT = Transpose(Cd);
    CdT->Finalize();

    // Matrix Dd and DdT
    mfem::MixedBilinearForm blf_Dd(&RT0, &DG);
    mfem::SparseMatrix Dd;
    blf_Dd.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_Dd.Assemble();
    blf_Dd.FormRectangularSystemMatrix(RT0_etdof,DG_etdof,Dd);
    Dd.Finalize();
    mfem::SparseMatrix Ddn = Dd;
    Ddn *= -1;
    mfem::SparseMatrix *DdnT = Transpose(Ddn);
    DdnT->Finalize();

    // Matrix G and GT
    mfem::MixedBilinearForm blf_G(&CG, &ND);
    mfem::SparseMatrix G;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(CG_etdof,ND_etdof,G);
    G.Finalize();
    mfem::SparseMatrix *GT = Transpose(G);
    GT->Finalize();

    // std::cout << "M:  " <<M.NumRows()  <<" "<< M.NumCols() << "\n"; 
    // std::cout << "CT: " <<CT->NumRows() <<" "<< CT->NumCols() << "\n"; 
    // std::cout << "G:  " <<G.NumRows() <<" "<<  G.NumCols() << "\n"; 
    // std::cout << "C:  " <<C.NumRows() <<" "<<  C.NumCols() << "\n"; 
    // std::cout << "Nn: " <<Nn.NumRows() <<" "<<  Nn.NumCols() << "\n"; 
    // std::cout << "GT: " <<GT->NumRows() <<" "<<  GT->NumCols() << "\n"; 
    // std::cout << "N:  " <<N.NumRows()  <<" "<< N.NumCols() << "\n"; 
    // std::cout << "C:  " <<C.NumRows() <<" "<<  C.NumCols() << "\n"; 
    // std::cout << "DTn:"<<DTn->NumRows()<<" "<< DTn->NumCols() << "\n"; 
    // std::cout << "CT: " <<CT->NumRows() <<" "<< CT->NumCols() << "\n"; 
    // std::cout << "Mn: " <<Mn.NumRows() <<" "<<  Mn.NumCols() << "\n"; 
    // std::cout << "D:  " <<D.NumRows()  <<" "<< D.NumCols() << "\n"; 
    // std::cout << "progress2" << "\n";

    // size of primal and dual system (nr of unknowns)
    int size_p = M.NumCols() + CT->NumCols() + G.NumCols();
    int size_d = Nd.NumCols() + Cd.NumCols() + DdnT->NumCols();

    // initialize right hand side matrices and vectors
    mfem::Vector b1(size_p);
    mfem::Vector b1sub(M.NumRows());
    mfem::Vector Mu(M.NumRows());
    mfem::Vector Rdu(M.NumRows());
    mfem::Vector CTz(M.NumRows());
    mfem::Vector b2(size_d);
    mfem::Vector b2sub(Nd.NumRows());
    mfem::Vector Nu(Nd.NumRows());
    mfem::Vector Rpu(Nd.NumRows());
    mfem::Vector Cw(Nd.NumRows());

    // indices of unkowns in solution vectors x, y
    mfem::Array<int> u_dofs;
    mfem::Array<int> z_dofs;
    mfem::Array<int> p_dofs;
    mfem::Array<int> v_dofs;
    mfem::Array<int> w_dofs;
    mfem::Array<int> q_dofs;
    for (int k=0; k<M.NumRows(); ++k)                          { u_dofs.Append(k); }
    for (int k=M.NumRows(); k<M.NumRows()+CT->NumRows(); ++k)  { z_dofs.Append(k); }
    for (int k=M.NumRows()+CT->NumRows(); k<size_p; ++k)       { p_dofs.Append(k); }
    for (int k=0; k<Nd.NumRows(); ++k)                         { v_dofs.Append(k); }
    for (int k=Nd.NumRows(); k<Nd.NumRows()+Cd.NumRows(); ++k) { w_dofs.Append(k); }
    for (int k=Nd.NumRows()+Cd.NumRows(); k<size_d; ++k)       { q_dofs.Append(k); }

    // solution vectors x and y
    mfem::Vector x(size_p);
    mfem::Vector y(size_d);
    x = 1.5;
    y = 1.;
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(z_dofs, z);
    x.GetSubVector(p_dofs, p);
    y.GetSubVector(v_dofs, v);
    y.GetSubVector(w_dofs, w);
    y.GetSubVector(q_dofs, q);

    // time loop
    std::cout << "---------------enter loop---------------\n";
    int T = 1;
    for (int t = 0 ; t < T ; t++) { 
        
        // Matrix Rp
        // TODO: Rp and Rd in 2D
        // TODO: warum hat R so große einträge - feasiblity check von matrix einträgen
        mfem::BilinearForm blf_Rp(&ND);
        mfem::SparseMatrix Rp;
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
        blf_Rp.Assemble();
        blf_Rp.FormSystemMatrix(ND_etdof,Rp);
        Rp.Finalize();
        
        // Matrix Rd
        mfem::BilinearForm blf_Rd(&RT);
        mfem::SparseMatrix Rd;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
        blf_Rd.Assemble();
        blf_Rd.FormSystemMatrix(RT_etdof,Rd);
        Rd.Finalize();
        std::cout << "progress: Rp,Rd assembled\n";
        
        // assemble A1, A2
        // TODO add Rp and Rd and dont overwrite M and N with them
        // TODO add constant factors (reynolds, dt, ...)
        mfem::SparseMatrix A1(size_p);
        mfem::SparseMatrix A2(size_d);
        AddSubmatrix(M,   A1, 0, 0); // submatrix, matrix, rowoffset, coloffset
        AddSubmatrix(*CT, A1, 0, M.NumCols());
        AddSubmatrix(G,   A1, 0, M.NumCols() + CT->NumCols());
        AddSubmatrix(C,   A1, M.NumRows(), 0);
        AddSubmatrix(*GT, A1, M.NumRows() + C.NumRows(), 0);
        AddSubmatrix(Nn,  A1, M.NumRows(), M.NumCols());
        A1.Finalize();
        AddSubmatrix(Nd,    A2, 0, 0);
        AddSubmatrix(Cd,    A2, 0, N.NumCols());
        AddSubmatrix(*DdnT, A2, 0, N.NumCols() + C.NumCols());
        AddSubmatrix(*CdT,  A2, N.NumRows(), 0);
        AddSubmatrix(Mdn,   A2, N.NumCols(), CT->NumRows());
        AddSubmatrix(Dd,    A2, N.NumRows() + CT->NumRows(), 0);
        A2.Finalize();
        std::cout << "progress: A1,A2 assembled\n";

        // assemble b1, b2
        // TODO feasibility 
        // add Rd and Rp
        M.Mult (u,Mu);
        // Rd.Mult(u,Rdu);
        CT->Mult(z,CTz);
        b1sub = Mu;
        // b1sub += Rdu;
        b1sub += CTz;
        for (int j = 0; j < M.NumRows(); j++) {b1.Elem(j) = b1sub.Elem(j);}
        N.Mult (u,Nu);
        // Rp.Mult(u,Rpu);
        CT->Mult(w,Cw);
        b2sub = Nu;
        // b2sub += Rpu;
        b2sub += Cw;
        for (int j = 0; j < N.NumRows(); j++) {b2.Elem(j) = b2sub.Elem(j);}
        std::cout << "progress: b1,b2 assembled\n";

        // check solution before solver
        // TODO: can not print out all entries of N, Nn, DTn, D
        // for (int i = 0;     i<  .NumRows(); i++) {
        //     for (int j = 0; j<  .NumCols(); j++) {
        //         std::cout << std::setprecision(1) << std::fixed;
        //         std::cout <<   .Elem(i,j) << "  ";
        //     }
        //     std::cout <<"\n";
        // }
        std::cout << "progress: matrix test\n";

        // solve system
        double tol = 1e-3;
        int iter = 10000;
        // primal: unkonwns at half integer time steps
        // dual:   unkonwns at integer      time steps
        mfem::MINRES(A1, b1, x, 0, iter, tol, tol); 
        mfem::MINRES(A2, b2, y, 0, iter, tol, tol); 
        std::cout << "progress: MINRES\n";
        
        // extract solution values u,w,p,v,z,q from x,y
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);
        y.GetSubVector(v_dofs, v);
        y.GetSubVector(w_dofs, w);
        y.GetSubVector(q_dofs, q);

        // check solution after solver
        std::cout << "x[1]="<<x[1] << "\n";
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
    
    // free memory
    delete fec_CG;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;
    delete fec_ND0;
    delete fec_RT0;
    
    std::cout << "---------------finish MEHC---------------\n";
}