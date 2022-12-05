#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace mfem;
// using namespace std;


// MEHC scheme on periodic domain, like in the paper



// TODO: umordnen der unbekannten, z ist primal, w ist dual
// TODO: use periodic mesh and no BC (4 functionspaces)
// TODO: check conservation (not convergence) like in 5.1 with the given init cond.
// TODO: change p,d into 1,2
// TODO: replace all M.NumCols() by u.Size()
// TODO: matrix names simpler
// TODO: are matrices and gridfunctions in right functionspaces? 

// primal: A1*x=b1
// [M+R    CT     G] [u]   [(M-R)*u - CT*z  + f]
// [C      -N      ] [w] = [         0         ]
// [GT             ] [p]   [         0         ]
//
// dual: A2*y=b2
// [N+R    C    -DT] [v]   [(N-R)*u - C*w + f]
// [CT     -M     0] [z] = [        0        ]
// [D      0      0] [q]   [        0        ]





void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix, int rowoffset, int coloffset);
void printvector(mfem::Vector vec, int stride=1);
void printvector2(mfem::Vector vec, int stride=1);
void printvector3(mfem::Vector vec, int stride=1, int start=0, int stop=0, int prec=3);
void printmatrix(mfem::Matrix &mat);



int main(int argc, char *argv[]) {

    std::cout << "---------------launch MEHC---------------\n";

    // TODO: use periodic mesh and no BC (4 functionspaces)
    // check conservation (not convergence) like in 5.1 with the given init cond.

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh";
    // const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
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
    mfem::GridFunction z(&RT); 
    mfem::GridFunction p(&CG);
    mfem::GridFunction v(&RT); 
    mfem::GridFunction w(&ND);
    mfem::GridFunction q(&DG);  
    
    // initial data
    // TODO 
    u = 4.3;
    z = 5.3;
    p = 6.3;
    v = 7.3;
    w = 8.3;
    q = 9.3;

    // boundary conditions
    mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof;

    // Matrix M and -M
    // TODO: benennung der matritzen primal/dual, bzw 1/2
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
    std::cout << "progress: assembled M,N,C,D,G\n";

    // system size
    int size_p = M.NumCols() + CT->NumCols() + G.NumCols();
    int size_d = Nd.NumCols() + Cd.NumCols() + DdnT->NumCols();

    // initialize solution vectors
    mfem::Vector x(size_p);
    mfem::Vector y(size_d);    
    x.SetVector(u,0);
    x.SetVector(z,M.NumCols());
    x.SetVector(p,M.NumCols()+CT->NumCols());
    y.SetVector(v,0);
    y.SetVector(w,Nd.NumCols());
    y.SetVector(q,Nd.NumCols()+Cd.NumCols());

    // initialize system matrices
    Array<int> offsets_1 (4); // number of variables + 1
    offsets_1[0] = 0;
    offsets_1[1] = ND.GetVSize();
    offsets_1[2] = RT.GetVSize();
    offsets_1[3] = CG.GetVSize();
    offsets_1.PartialSum(); // =exclusive scan
    Array<int> offsets_2 (4);
    offsets_2[0] = 0;
    offsets_2[1] = ND.GetVSize();
    offsets_2[2] = RT.GetVSize();
    offsets_2[3] = DG.GetVSize();
    offsets_2.PartialSum();
    mfem::BlockMatrix A1(offsets_1);
    mfem::BlockMatrix A2(offsets_2);

    // initialize rhs
    mfem::Vector b1(size_p); 
    mfem::Vector b1sub(size_p);
    mfem::Vector b2(size_d); 
    mfem::Vector b2sub(size_p);
    
    // helper dofs
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
    std::cout << "progress: initialized vectors\n";


    // time loop
    std::cout << "---------------enter loop---------------\n";
    int T = 1;
    for (int t = 0 ; t < T ; t++) {

        // update R1
        mfem::BilinearForm blf_Rp(&ND);
        mfem::SparseMatrix Rp;
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
        blf_Rp.Assemble();
        blf_Rp.FormSystemMatrix(ND_etdof,Rp);
        Rp.Add(1,M);
        Rp.Finalize();

        // update R2
        mfem::BilinearForm blf_Rd(&RT);
        mfem::SparseMatrix Rd;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
        blf_Rd.Assemble();
        blf_Rd.FormSystemMatrix(RT_etdof,Rd);
        Rd.Add(1,N);
        Rd.Finalize();
        std::cout << "progress: updated Rp,Rd\n";

        // update A1, A2
        A1.SetBlock(0,0, &Rp); // includes M
        A1.SetBlock(0,1, CT);
        A1.SetBlock(1,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &Nn);
        A1.SetBlock(2,1, GT);
        A2.SetBlock(0,0, &Rd); // includes N
        A2.SetBlock(0,1, &C);
        A2.SetBlock(1,2, DdnT);
        A2.SetBlock(1,0, CT);
        A2.SetBlock(1,1, &Mn);
        A2.SetBlock(2,1, &Dd);
        std::cout << "progress: updated A1,A2\n";

        // update b1, b2
        // TODO add constants, like Re
        // TODO det(CT) is small => consequences?
        b1 = 0.0;
        M.Mult(u,b1sub);
        Rp.AddMult(u,b1sub,-1);
        CT->AddMult(z,b1sub,-1);
        b1.AddSubVector(b1sub,0);
        b2 = 0.0;
        N.Mult(v,b2sub);
        Rd.AddMult(v,b2sub,-1);
        C.AddMult(w,b2sub,-1);
        b2.AddSubVector(b2sub,0);
        std::cout << "progress: updated b1,b2\n";

        // TODO check why this RHS doesnt produce x=
        // TODO: 1) isolate problem 2) check other solver 3) precond.
        mfem::Vector helper(size_p);
        mfem::Vector newrhs(size_p);
        mfem::Vector xxx(size_p);
        xxx=1.299;
        helper = 1.3;
        A1.Mult(helper,newrhs);
        double tol = 1e-3;
        int iter = 20000;
        mfem::MINRES(A1, newrhs, xxx, 0, iter, tol, tol); 
        // printvector3(xxx,1,0,20,6);
        
        // solve 
        // double tol = 1e-3;
        // int iter = 2000;
        // mfem::Vector drei(size_p); drei = 3.0;
        // A1.Mult(drei,b1);
        // mfem::MINRES(A1, b1, x, 0, iter, tol, tol); 
        // mfem::MINRES(A2, b2, y, 0, iter, tol, tol); 
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