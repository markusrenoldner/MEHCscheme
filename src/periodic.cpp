#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

// MEHC scheme on periodic domain, like in the paper

// primal: A1*x=b1
// [M_dt+R1  CT_Re    G] [u]   [(M_dt-R1)*u - CT_Re*z  + f]
// [C        N_n      0] [z] = [             0            ]
// [GT       0        0] [p]   [             0            ]
//
// dual: A2*y=b2
// [N_dt+R2  C_Re     DT_n] [v]   [(N_dt-R2)*u - C_Re*w + f]
// [CT       M_n      0   ] [w] = [            0           ]
// [D        0        0   ] [q]   [            0           ]

// attention: the systems are coupled
// z...vorticity of primal system, but corresponds to dual velocity v
// w...vorticity of dual system, but corresponds to primal velocity u
// u,z,p at half integer, and v,w,q at full integer time steps, hence:
// R1 depends on w and defined on full int time step, but part of primal syst
// R2 depends on z and defined on half int time step, but part of dual system






void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix,
                  int rowoffset, int coloffset);
void PrintVector(mfem::Vector vec, int stride=1);
void PrintVector2(mfem::Vector vec, int stride=1);
void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void PrintMatrix(mfem::Matrix &mat, int prec=2);
void u_0(const mfem::Vector &x, mfem::Vector &v);
void w_0(const mfem::Vector &x, mfem::Vector &w);



int main(int argc, char *argv[]) {

    std::cout << "---------------launch MEHC---------------\n";

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh"; 
    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    // for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // simulation parameters
    double Re_inv = 0.; // = 1/Re
    double dt = 1.;
    int timesteps = 2;

    // FE spaces (CG \in H1, DG \in L2)
    int order = 1;
    mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
    mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order,dim);
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order,dim);
    mfem::FiniteElementSpace CG(&mesh, fec_CG);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);
    mfem::FiniteElementSpace RT(&mesh, fec_RT);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND); // u = 4.3;
    mfem::GridFunction z(&RT); // z = 5.3;
    mfem::GridFunction p(&CG); // p = 6.3;
    mfem::GridFunction v(&RT); // v = 7.3;
    mfem::GridFunction w(&ND); // w = 8.3;
    mfem::GridFunction q(&DG); // q = 9.3;

    // initial condition
    mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
    mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0);
    u.ProjectCoefficient(u_0_coeff);
    v.ProjectCoefficient(u_0_coeff);
    z.ProjectCoefficient(w_0_coeff);
    w.ProjectCoefficient(w_0_coeff);

    // system size
    int size_1 = u.Size() + z.Size() + p.Size();
    int size_2 = v.Size() + w.Size() + q.Size();
    std::cout << "size1: " << size_1 << "\n"<<"size2: "<<size_2<< "\n";
    std::cout<< "size u/z/p: "<<u.Size()<<"/"<<z.Size()<<"/"<<p.Size()<<"\n";
    std::cout<< "size v/w/q: "<<v.Size()<<"/"<<w.Size()<<"/"<<q.Size()<<"\n";
    
    // initialize solution vectors
    mfem::Vector x(size_1);
    mfem::Vector y(size_2);
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
    std::iota(&q_dofs[0], &q_dofs[q.Size()], v.Size()+w.Size());
    std::cout << "progress: initialized unknowns\n";

    // boundary conditions
    // TODO: is this correct?
    // TODO: ex5: submatrices without BC, ex9: esstdof is empty
    // TODO: ex5,8,19: BC in blockmatrix problems
    // TODO: check if matrices have diagonal struct, or weird BC artefacts
    mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof;

    // Matrix M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M_dt;
    mfem::SparseMatrix M_n;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ND_etdof,M_n);
    M_dt = M_n;
    M_dt *= 1/dt;
    M_n *= -1.;
    M_dt.Finalize();
    M_n.Finalize();

    // Matrix N
    mfem::BilinearForm blf_N(&RT);
    mfem::SparseMatrix N_dt;
    mfem::SparseMatrix N_n;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
    blf_N.Assemble();
    blf_N.FormSystemMatrix(RT_etdof,N_n);
    N_dt = N_n;
    N_dt *= 1/dt;
    N_n *= -1.;
    N_dt.Finalize();
    N_n.Finalize();

    // Matrix C
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    mfem::SparseMatrix *CT;
    mfem::SparseMatrix C_Re;
    mfem::SparseMatrix CT_Re;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator()); //=(curl u,v)
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    CT = Transpose(C);
    C_Re = C;
    CT_Re = *CT; 
    C_Re *= Re_inv/2.;
    CT_Re *= Re_inv/2.;
    *CT *= Re_inv/2.;
    C *= Re_inv/2.;
    C.Finalize();
    CT->Finalize();
    C_Re.Finalize();
    CT_Re.Finalize();

    // Matrix D
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    mfem::SparseMatrix D;
    mfem::SparseMatrix *DT_n;
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator()); //=(div u,v)
    blf_D.Assemble();
    blf_D.FormRectangularSystemMatrix(RT_etdof,DG_etdof,D);
    DT_n = Transpose(D);
    *DT_n *= -1.;
    D.Finalize();
    DT_n->Finalize();

    // Matrix G
    mfem::MixedBilinearForm blf_G(&CG, &ND);
    mfem::SparseMatrix G;
    mfem::SparseMatrix *GT;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(CG_etdof,ND_etdof,G);
    GT = Transpose(G);
    G.Finalize();
    GT->Finalize();

    // initialize system matrices
    mfem::Array<int> offsets_1 (4);
    offsets_1[0] = 0;
    offsets_1[1] = u.Size();
    offsets_1[2] = z.Size();
    offsets_1[3] = p.Size();
    offsets_1.PartialSum(); // exclusive scan
    mfem::Array<int> offsets_2 (4);
    offsets_2[0] = 0;
    offsets_2[1] = v.Size();
    offsets_2[2] = w.Size();
    offsets_2[3] = q.Size();
    offsets_2.PartialSum();
    mfem::BlockMatrix A1(offsets_1);
    mfem::BlockMatrix A2(offsets_2);
    std::cout << "progress: initialized system matrices\n";

    // initialize rhs
    mfem::Vector b1(size_1);
    mfem::Vector b1sub(u.Size());
    mfem::Vector b2(size_2); 
    mfem::Vector b2sub(v.Size());
    std::cout << "progress: initialized RHS\n";
    
    // empty boundary DOF array for conservation tests
    mfem::Array<int> ess_tdof_list;
    
    // mass conservation 1
    // TODO implement mass conservation correctly (divergence of u as an inner product?)
    mfem::MixedBilinearForm mblf1(&ND,&CG);
    mfem::SparseMatrix mmat1;
    mblf1.AddDomainIntegrator(new mfem::MixedVectorWeakDivergenceIntegrator());//=(u,grad v)
    mblf1.Assemble();
    mblf1.FormRectangularSystemMatrix(ess_tdof_list,ess_tdof_list,mmat1);

    // mass conservation 2
    mfem::MixedBilinearForm mblf2(&DG,&RT);
    mfem::SparseMatrix mmat2;
    mblf2.AddDomainIntegrator(new mfem::MixedScalarWeakGradientIntegrator()); //=(u,div v)
    mblf2.Assemble();
    mblf2.FormRectangularSystemMatrix(ess_tdof_list,ess_tdof_list,mmat2);

    // energy conservation 1
    // TODO: fix conservation of E
    // TODO: energy, helicity of init cond not coherent with paper
    // TODO: use vectormassintegrator? or mixed...?
    // TODO: look for use of "innerproduct" in mfem examples
    mfem::BilinearForm eblf1(&ND);
    mfem::SparseMatrix emat1;
    eblf1.AddDomainIntegrator(new mfem::MixedVectorMassIntegrator()); //=(u,v)
    eblf1.Assemble();
    eblf1.FormSystemMatrix(ess_tdof_list,emat1);

    // energy conservation 2
    mfem::BilinearForm eblf2(&RT);
    mfem::SparseMatrix emat2;
    eblf2.AddDomainIntegrator(new mfem::MixedVectorMassIntegrator()); //=(u,v)
    eblf2.Assemble();
    eblf2.FormSystemMatrix(ess_tdof_list,emat2);

    // helicity conservation 1
    // TODO: fix conservation of H
    mfem::BilinearForm hblf1(&ND);
    mfem::SparseMatrix hmat1;
    hblf1.AddDomainIntegrator(new mfem::MixedVectorMassIntegrator()); //=(u,v)
    hblf1.Assemble();
    hblf1.FormSystemMatrix(ess_tdof_list,hmat1);

    // helicity conservation 2
    mfem::BilinearForm hblf2(&RT);
    mfem::SparseMatrix hmat2;
    hblf2.AddDomainIntegrator(new mfem::MixedVectorMassIntegrator()); //=(u,v)
    hblf2.Assemble();
    hblf2.FormSystemMatrix(ess_tdof_list,hmat2);

    // print
    std::cout << "div(u) = " << mmat1.InnerProduct(u,p) << "\n";
    std::cout << "div(v) = " << mmat2.InnerProduct(q,v) << "\n";
    std::cout << "    E1 = " << 1./2.*emat1.InnerProduct(u,u) << "\n";
    std::cout << "    E2 = " << 1./2.*emat2.InnerProduct(v,v) << "\n";
    std::cout << "    H1 = " << hmat1.InnerProduct(u,w) << "\n";
    std::cout << "    H2 = " << hmat2.InnerProduct(v,z) << "\n";








    // TODO: check: w2 =? curl u1; w2 =? curl u2; etc.

    // TODO check why Cu-Nz!=0 for init cond
    // mfem::Vector sol1 (z.Size());
    // C.Mult(u,sol1);
    // N_n.AddMult(z,sol1);
    // printvector3(sol1,1,10,50,15); 

    // TODO check why CTv-Mw=0 for init cond
    // mfem::Vector sol2 (w.Size());
    // CT->Mult(v,sol2);
    // M_n.AddMult(w,sol2);
    // printvector3(sol2,2,10,50,15); 










    // TODO: integrate explicit euler code

    // time loop
    double T;
    for (int i = 0 ; i < timesteps ; i++) {
        T = i*dt;
        std::cout << "---------------enter loop, t="<<T<<"-----------\n";
        
        // update R1
        mfem::BilinearForm blf_R1(&ND);
        mfem::SparseMatrix R1;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_R1.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_R1.Assemble();
        blf_R1.FormSystemMatrix(ND_etdof,R1);
        R1 *= 1./2.;
        R1.Finalize();

        // update R2
        mfem::BilinearForm blf_R2(&RT);
        mfem::SparseMatrix R2;
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_R2.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_R2.Assemble();
        blf_R2.FormSystemMatrix(RT_etdof,R2);
        R2 *= 1./2.;
        R2.Finalize();

        // M+R and N+R
        mfem::SparseMatrix MR = M_dt;
        mfem::SparseMatrix NR = N_dt;
        R1.Add(1,MR); 
        R2.Add(1,NR);
        MR.Finalize();
        NR.Finalize();

        // update A1, A2
        A1.SetBlock(0,0, &MR);
        A1.SetBlock(0,1, &CT_Re);
        A1.SetBlock(0,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &N_n);
        A1.SetBlock(2,0, GT);

        A2.SetBlock(0,0, &NR);
        A2.SetBlock(0,1, &C_Re);
        A2.SetBlock(0,2, DT_n);
        A2.SetBlock(1,0, CT);
        A2.SetBlock(1,1, &M_n);
        A2.SetBlock(2,0, &D);
        std::cout << "progress: updated system matrices\n";

        // update b1, b2
        b1 = 0.0;
        b1sub = 0.0;
        M_dt.AddMult(u,b1sub);
        R1.AddMult(u,b1sub,-1);
        CT_Re.AddMult(z,b1sub,-1);
        b1.AddSubVector(b1sub,0);
        b2 = 0.0;
        b2sub = 0.0;
        N_dt.AddMult(v,b2sub);
        R2.AddMult(v,b2sub,-1);
        C_Re.AddMult(w,b2sub,-1);
        b2.AddSubVector(b2sub,0);
        std::cout << "progress: updated RHS\n";

        // create symmetric system AT*A*x=AT*b
        mfem::TransposeOperator AT1 (&A1);
        mfem::TransposeOperator AT2 (&A2);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
        mfem::Vector ATb1 (size_1);
        mfem::Vector ATb2 (size_2);
        A1.MultTranspose(b1,ATb1);
        A2.MultTranspose(b2,ATb2);

        // solve 
        double tol = 1e-12;
        int iter = 10000;
        mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol); 
        mfem::MINRES(ATA2, ATb2, y, 0, iter, tol*tol, tol*tol); 
        std::cout << "progress: MINRES\n";

        // check residuum
        // mfem::Vector res1(size_1); res1=0.;
        // mfem::Vector res2(size_2); res2=0.;
        // A1.Mult(x,res1); A2.Mult(y,res2);
        // res1 -= b1; res2 -= b2;
        // printvector3(res1,1,0,20,15);
        // printvector3(res2,1,0,20,15);
        
        // extract solution values u,w,p,v,z,q from x,y
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);
        y.GetSubVector(v_dofs, v);
        y.GetSubVector(w_dofs, w);      
        y.GetSubVector(q_dofs, q);

        // check solution after solver
        // printvector3(x,1,0,20,6);
        // printvector3(y,1,0,20,6);

        // conserved quantities
        std::cout << "div(u) = " << mmat1.InnerProduct(u,p) << "\n";
        std::cout << "div(v) = " << mmat2.InnerProduct(q,v) << "\n";
        std::cout << "    E1 = " << 1./2.*emat1.InnerProduct(u,u) << "\n";
        std::cout << "    E2 = " << 1./2.*emat2.InnerProduct(v,v) << "\n";
        std::cout << "    H1 = " << hmat1.InnerProduct(u,w) << "\n";
        std::cout << "    H2 = " << hmat2.InnerProduct(v,z) << "\n";
        
    }
    std::cout << "---------------exit loop, t="<<T+dt<<"------------\n";

    // free memory
    delete fec_CG;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;

    std::cout << "---------------finish MEHC---------------\n";
}

void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
    double pi = 3.14159265358979323846;
    int dim = x.Size();

    // old: u0=(cos(2piz), sin(2piz), sin(2pix))
    // u0=(cos(piz), sin(piz), sin(pix))
    // periodic on [-1,1]^3
    returnvalue(0) = std::cos(pi*x(3));
    returnvalue(1) = std::sin(pi*x(3)); 
    returnvalue(2) = std::sin(pi*x(1));
}

void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
    double pi = 3.14159265358979323846;
    int dim = x.Size();

    // old: w0=(-2pi cos(2piz), -2pi cos(2pix) -2pi sin(2piz), 0) 
    // w0=(-pi cos(piz), -pi cos(pix) -pi sin(piz), 0) 
    // periodic on [-1,1]^3
    returnvalue(0) = -pi*std::cos(pi*x(3));
    returnvalue(1) = -pi*std::cos(pi*x(1)) -pi*std::sin(pi*x(3)); 
    returnvalue(2) = 0;
}