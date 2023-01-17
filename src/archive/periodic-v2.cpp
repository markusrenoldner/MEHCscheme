#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace mfem;
// using namespace std;



// outdated!













// MEHC scheme on periodic domain, like in the paper



// TODO: umordnen der unbekannten, z ist primal, w ist dual
// TODO: use periodic mesh and no BC (4 functionspaces)
// TODO: check conservation (not convergence) like in 5.1 with the given init cond.
// TODO: change p,d into 1,2

// primal: A1*x=b1
// [M+R    CT     G] [u]   [(M-R)*u - CT*z  + f]
// [C      -N      ] [w] = [         0         ]
// [GT             ] [p]   [         0         ]
//
// dual: A2*y=b2
// [N+R    C    -DT] [v]   [(N-R)*u - C*w + f]
// [CT     -M     0] [z] = [        0        ]
// [D      0      0] [q]   [        0        ]





// void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix, int rowoffset, int coloffset);
// void printvector(mfem::Vector vec, int stride=1);
// void printmatrix(mfem::Matrix &mat);



int main(int argc, char *argv[]) {

    std::cout << "---------------launch MEHC---------------\n";

    // TODO: use periodic mesh and no BC (4 functionspaces)
    // check conservation (not convergence) like in 5.1 with the given init cond.

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    // for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // FE spaces (CG \in H1, DG \in L2)
    int order = 1;
    FiniteElementCollection *fec_CG  = new H1_FECollection(order, dim);
    FiniteElementCollection *fec_ND  = new ND_FECollection(order, dim);
    FiniteElementCollection *fec_RT  = new RT_FECollection(order, dim);
    FiniteElementCollection *fec_DG  = new L2_FECollection(order, dim);
    FiniteElementSpace CG(&mesh, fec_CG);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace DG(&mesh, fec_DG);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND);
    mfem::GridFunction w(&RT);
    mfem::GridFunction p(&DG);
    mfem::GridFunction v(&ND); 
    mfem::GridFunction z(&RT); 
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
    mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof;
    // mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    // ess_bdr = 0;
    // CG.GetEssentialTrueDofs(ess_bdr, CG_etdof);
    // ND.GetEssentialTrueDofs(ess_bdr, ND_etdof);
    // RT.GetEssentialTrueDofs(ess_bdr, RT_etdof);
    // DG.GetEssentialTrueDofs(ess_bdr, DG_etdof);
    std::cout << "progress: assembled BCs" << "\n";

    // Matrix M and -M
    // TODO: benennung der matritzen primal/dual
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
    mfem::BilinearForm blf_Md(&ND);
    mfem::SparseMatrix Md;
    blf_Md.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_Md.Assemble();
    blf_Md.FormSystemMatrix(ND_etdof,Md); 
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
    mfem::BilinearForm blf_Nd(&RT);
    mfem::SparseMatrix Nd;
    blf_Nd.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_Nd.Assemble();
    blf_Nd.FormSystemMatrix(RT_etdof,Nd);
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
    mfem::MixedBilinearForm blf_Cd(&ND, &RT);
    mfem::SparseMatrix Cd;
    blf_Cd.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_Cd.Assemble();
    blf_Cd.FormRectangularSystemMatrix(ND_etdof,RT_etdof,Cd);
    Cd.Finalize();
    mfem::SparseMatrix *CdT = Transpose(Cd);
    CdT->Finalize();

    // Matrix Dd and DdT
    mfem::MixedBilinearForm blf_Dd(&RT, &DG);
    mfem::SparseMatrix Dd;
    blf_Dd.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_Dd.Assemble();
    blf_Dd.FormRectangularSystemMatrix(RT_etdof,DG_etdof,Dd);
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

    // system size
    int size_p = M.NumCols() + CT->NumCols() + G.NumCols();
    int size_d = Nd.NumCols() + Cd.NumCols() + DdnT->NumCols();

    // system matrices
    Array<int> offsets_p (3); // number of variables + 1
    offsets_p[0] = 0;
    offsets_p[1] = ND.GetVSize();
    offsets_p[2] = RT.GetVSize();
    offsets_p.PartialSum(); // =exclusive 
    Array<int> offsets_d (3);
    offsets_d[0] = 0;
    offsets_d[1] = ND.GetVSize();
    offsets_d[2] = RT.GetVSize();
    offsets_d.PartialSum();
    mfem::BlockMatrix A1(offsets_p);
    mfem::BlockMatrix A2(offsets_d);

    // initialize right hand side matrices and vectors
    mfem::Vector b1(size_p); 
    mfem::Vector b1sub(size_p);
    mfem::Vector b2(size_d); 
    mfem::Vector b2sub(size_p);
    
    // unknown dofs
    mfem::Array<int> u_dofs (M.NumCols());
    mfem::Array<int> z_dofs (CT->NumCols());
    mfem::Array<int> p_dofs (G.NumCols());
    mfem::Array<int> v_dofs (N.NumCols());
    mfem::Array<int> w_dofs (C.NumCols());
    mfem::Array<int> q_dofs (DdnT->NumCols());
    std::iota(&u_dofs[0], &u_dofs[M.NumCols()],0);
    std::iota(&z_dofs[0], &z_dofs[CT->NumCols()],M.NumCols());
    std::iota(&p_dofs[0], &p_dofs[G.NumCols()],M.NumCols()+CT->NumCols());
    std::iota(&v_dofs[0], &v_dofs[N.NumCols()],0);
    std::iota(&w_dofs[0], &w_dofs[C.NumCols()],N.NumCols());
    std::iota(&q_dofs[0], &q_dofs[DdnT->NumCols()],N.NumCols()+DdnT->NumCols());

    // solution vectors x and y
    // TODO: getsubvector ohne dofs
    mfem::Vector x(size_p); x = 1.5;
    mfem::Vector y(size_d); y = 1.5;
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
        // mfem::BilinearForm blf_Rp(&ND);
        // mfem::SparseMatrix Rp;
        // mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        // blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
        // blf_Rp.Assemble();
        // blf_Rp.FormSystemMatrix(ND_etdof,Rp);
        // Rp.Finalize();
        
        // Matrix Rd
        // mfem::BilinearForm blf_Rd(&RT);
        // mfem::SparseMatrix Rd;
        // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        // blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
        // blf_Rd.Assemble();
        // blf_Rd.FormSystemMatrix(RT_etdof,Rd);
        // Rd.Finalize();
        // std::cout << "progress: Rp,Rd assembled\n";

        

        
        // update A1, A2
        // TODO add Rp and Rd and dont overwrite M and N with them
        // TODO add constant factors (reynolds, dt, ...)
        A1.SetBlock(0,0, &M);
        A1.SetBlock(0,1, CT);
        A1.SetBlock(1,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &Nn);
        A1.SetBlock(2,1, GT);
        A2.SetBlock(0,0, &N);
        A2.SetBlock(0,1, &C);
        A2.SetBlock(1,2, DdnT);
        A2.SetBlock(1,0, CT);
        A2.SetBlock(1,1, &Mn);
        A2.SetBlock(2,1, &Dd);
        std::cout << "progress: A1,A2 assembled\n";

        // update b1, b2
        b1 = 0.0;
        M.Mult(u,b1sub);
        CT->AddMult(z,b1sub,-1);
        b1.AddSubVector(b1sub,0);
        b2 = 0.0;
        M.Mult(u,b1sub);
        CT->AddMult(z,b1sub,-1);
        b1.AddSubVector(b1sub,0);
        std::cout << "progress: b1,b2 assembled\n";

        // solve system
        // primal: unkonwns at half integer time steps
        // dual:   unkonwns at integer      time steps
        double tol = 1e-3;
        int iter = 10000;
        std::cout << "progress: start MINRES\n";
        // mfem::MINRES(A, b, x, 0, iter, tol, tol);
        // mfem::MINRES(A1, b1, x, 0, iter, tol, tol); 
        std::cout << "progress: start MINRES2\n";
        // mfem::MINRES(A2, b2, y, 0, iter, tol, tol); 
        std::cout << "progress: MINRES\n";
        
        
        // extract solution values u,w,p,v,z,q from x,y
        // TODO ohne dofs, siehe oben
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
    // std::ofstream mesh_ofs("refined.mesh");
    // mesh_ofs.precision(8);
    // mesh.Print(mesh_ofs);
    // std::ofstream sol_ofs("sol.gf");
    // sol_ofs.precision(8);
    // u.Save(sol_ofs);
    // char vishost[] = "localhost";
    // int  visport   = 19916;
    // socketstream sol_sock(vishost, visport);
    // sol_sock.precision(8);
    // sol_sock << "solution\n" << mesh << u << std::flush;
    
    // free memory
    delete fec_CG;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;

    
    std::cout << "---------------finish MEHC---------------\n";
    return 0;
}