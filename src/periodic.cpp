#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>


// MEHC scheme on periodic domain, like in the paper




// TODO: use periodic mesh and no BC (4 functionspaces)
// TODO: check conservation (not convergence) like in 5.1 with the given init cond.
// TODO: change p,d into 1,2, matrix names simpler
// TODO: are matrices and gridfunctions in right functionspaces? 


// primal: A1*x=b1
// [M+R    CT     G] [u]   [(M-R)*u - CT*z  + f]
// [C      -N      ] [z] = [         0         ]
// [GT             ] [p]   [         0         ]
//
// dual: A2*y=b2
// [N+R    C    -DT] [v]   [(N-R)*u - C*w + f]
// [CT     -M     0] [w] = [        0        ]
// [D      0      0] [q]   [        0        ]





void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix, int rowoffset, int coloffset);
void printvector(mfem::Vector vec, int stride=1);
void printvector2(mfem::Vector vec, int stride=1);
void printvector3(mfem::Vector vec, int stride=1, int start=0, int stop=0, int prec=3);
void printmatrix(mfem::Matrix &mat);



int main(int argc, char *argv[]) {

    std::cout << "---------------launch MEHC---------------\n";

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh";
    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    // for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // simulation parameters
    double Re = 1;
    double dt = 0.1;
    int timesteps = 3;

    // FE spaces (CG \in H1, DG \in L2)
    int order = 1;
    mfem::FiniteElementCollection *fec_CG  = new mfem::H1_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_ND  = new mfem::ND_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_RT  = new mfem::RT_FECollection(order, dim);
    mfem::FiniteElementCollection *fec_DG  = new mfem::L2_FECollection(order, dim);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace RT(&mesh, fec_RT);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND); u = 4.3;
    mfem::GridFunction z(&RT); z = 5.3; 
    mfem::GridFunction p(&CG); p = 6.3;
    mfem::GridFunction v(&RT); v = 7.3; 
    mfem::GridFunction w(&ND); w = 8.3;
    mfem::GridFunction q(&DG); q = 9.3;      

    // system size
    int size_p = u.Size() + z.Size() + p.Size();
    int size_d = v.Size() + w.Size() + q.Size();
    
    // initialize solution vectors
    mfem::Vector x(size_p);
    mfem::Vector y(size_d);
    x.SetVector(u,0);
    x.SetVector(z,u.Size());
    x.SetVector(p,u.Size()+z.Size());
    y.SetVector(v,0);
    y.SetVector(w,v.Size());
    y.SetVector(q,v.Size()+w.Size());

    // helper dofs
    mfem::Array<int> u_dofs (u.Size());
    mfem::Array<int> z_dofs (z.Size());
    mfem::Array<int> p_dofs (p.Size());
    mfem::Array<int> v_dofs (v.Size());
    mfem::Array<int> w_dofs (w.Size());
    mfem::Array<int> q_dofs (q.Size());
    std::iota(&u_dofs[0], &u_dofs[u.Size()], 0);
    std::iota(&z_dofs[0], &z_dofs[z.Size()], u.Size());
    std::iota(&p_dofs[0], &p_dofs[p.Size()], u.Size()+z.Size());
    std::iota(&v_dofs[0], &v_dofs[v.Size()], 0);
    std::iota(&w_dofs[0], &w_dofs[w.Size()], v.Size());
    std::iota(&q_dofs[0], &q_dofs[q.Size()], v.Size()+q.Size());
    std::cout << "progress: initialized unknowns\n";

    // boundary conditions
    mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof;

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
    mfem::SparseMatrix Mdt = M;
    Mdt *= 1/dt;

    // Matrix Md and Mdn
    // TODO merge with M
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

    // initialize system matrices
    mfem::Array<int> offsets_1 (4); // number of variables + 1
    offsets_1[0] = 0;
    offsets_1[1] = ND.GetVSize();
    offsets_1[2] = RT.GetVSize();
    offsets_1[3] = CG.GetVSize();
    offsets_1.PartialSum(); // exclusive scan
    mfem::Array<int> offsets_2 (4);
    offsets_2[0] = 0;
    offsets_2[1] = ND.GetVSize();
    offsets_2[2] = RT.GetVSize();
    offsets_2[3] = DG.GetVSize();
    offsets_2.PartialSum();
    mfem::BlockMatrix A1(offsets_1);
    mfem::BlockMatrix A2(offsets_2);
    std::cout << "progress: initialized system matrices\n";

    // initialize rhs
    mfem::Vector b1(size_p); 
    mfem::Vector b1sub(size_p);
    mfem::Vector b2(size_d); 
    mfem::Vector b2sub(size_p);
    std::cout << "progress: initialized RHS\n";

    // time loop
    double T;
    for (int i = 0 ; i < timesteps ; i++) {
        T = i*dt;
        std::cout << "---------------enter loop, t="<<T<<"-----------\n";

        // update R1
        mfem::BilinearForm blf_Rp(&ND);
        mfem::SparseMatrix Rp;
        mfem::SparseMatrix MR = M; // TODO make sure its a deep copy
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
        blf_Rp.Assemble();
        blf_Rp.FormSystemMatrix(ND_etdof,Rp);
        Rp.Add(1,MR);
        Rp.Finalize();

        // update R2
        mfem::BilinearForm blf_Rd(&RT);
        mfem::SparseMatrix Rd;
        mfem::SparseMatrix NR = N;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
        blf_Rd.Assemble();
        blf_Rd.FormSystemMatrix(RT_etdof,Rd);
        Rd.Add(1,NR);
        Rd.Finalize();

        // update A1, A2
        // TODO add constants, like Re
        A1.SetBlock(0,0, &MR);
        A1.SetBlock(0,1, CT);
        A1.SetBlock(1,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &Nn);
        A1.SetBlock(2,1, GT);
        A2.SetBlock(0,0, &NR);
        A2.SetBlock(0,1, &C);
        A2.SetBlock(1,2, DdnT);
        A2.SetBlock(1,0, CT);
        A2.SetBlock(1,1, &Mn);
        A2.SetBlock(2,1, &Dd);
        std::cout << "progress: updated system matrices\n";

        // update b1, b2
        // TODO if det(CT) is small => consequences?
        b1 = 0.0;
        b1sub = 0.0;
        M.AddMult(u,b1sub,1/dt);
        Rp.AddMult(u,b1sub,-1/2);
        CT->AddMult(z,b1sub,-1/(2*Re));
        b1.AddSubVector(b1sub,0);
        b2 = 0.0;
        b2sub = 0.0;
        N.AddMult(v,b2sub,1/dt);
        Rd.AddMult(v,b2sub,-1/2);
        C.AddMult(w,b2sub,-1/(2*Re));
        b2.AddSubVector(b2sub,0);
        std::cout << "progress: updated RHS\n";
        // printvector3(b1,20,0,0,6);
        
        // TODO check why this RHS doesnt produce x=...
        // TODO: 1) isolate problem 2) check other solver 3) precond.
        // mfem::Vector helper(size_p);
        // mfem::Vector newrhs(size_p);
        // mfem::Vector xxx(size_p);
        // xxx=1.299;
        // helper = 1.3;
        // A1.Mult(helper,newrhs);
        // double tol = 1e-3;
        // int iter = 20000;
        // mfem::MINRES(A1, newrhs, xxx, 0, iter, tol, tol); 
        // printvector3(xxx,1,0,20,6);
        
        // solve 
        double tol = 1e-3;
        int iter = 2000;
        // mfem::Vector drei(size_p); drei = 3.0;
        // A1.Mult(drei,b1);
        mfem::MINRES(A1, b1, x, 0, iter, tol, tol); 
        mfem::MINRES(A2, b2, y, 0, iter, tol, tol); 
        std::cout << "progress: MINRES\n";
        
        // extract solution values u,w,p,v,z,q from x,y
        // TODO: getsubvector ohne dofs
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);
        y.GetSubVector(v_dofs, v);
        y.GetSubVector(w_dofs, w);
        y.GetSubVector(q_dofs, q);

        // check solution after solver
        // printvector3(x,1,0,20,6);
    }

    // free memory
    delete fec_CG;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;

    
    std::cout << "---------------finish MEHC---------------\n";
}