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






// void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix,
//                   int rowoffset, int coloffset);
// void PrintVector(mfem::Vector vec, int stride=1);
// void PrintVector2(mfem::Vector vec, int stride=1);
// void PrintVector3(mfem::Vector vec, int stride=1, 
//                   int start=0, int stop=0, int prec=3);
// void PrintMatrix(mfem::Matrix &mat, int prec=2);
void u_0(const mfem::Vector &x, mfem::Vector &v);
void w_0(const mfem::Vector &x, mfem::Vector &v);



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
    int timesteps = 10; 

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
    
    // helper vectors to compute some discrete objects
    mfem::Vector uold = u;
    mfem::Vector vold = v;
    mfem::Vector zold = z;
    mfem::Vector wold = w;
    mfem::Vector udiff(u.Size()); 
    mfem::Vector uavg (u.Size()); 
    mfem::Vector zavg (z.Size()); 
    mfem::Vector vdiff(v.Size()); 
    mfem::Vector vavg (v.Size()); 
    mfem::Vector wavg (w.Size()); 
    mfem::Vector wdiff (w.Size()); 

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

    // boundary conditions
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

    // initialize rhs
    mfem::Vector b1(size_1);
    mfem::Vector b1sub(u.Size());
    mfem::Vector b2(size_2); 
    mfem::Vector b2sub(v.Size());
    
    // empty boundary DOF array for conservation tests
    mfem::Array<int> ess_tdof_list;
    
    // conservation properties
    // mfem::Vector mass_vec1 (p.Size());
    // mfem::Vector mass_vec2 (q.Size());
    // GT->Mult(u,mass_vec1);
    // D.Mult(v,mass_vec2);
    // std::cout << "div(u) = " << mass_vec1.Norml2() << "\n";
    // std::cout << "div(v) = " << mass_vec2.Norml2() << "\n";
    // std::cout << "    E1 = " << -1./2.*blf_M.InnerProduct(u,u) << "\n";
    // std::cout << "    E2 = " << -1./2.*blf_N.InnerProduct(v,v) << "\n";
    // TODO fix helicity conservation
    std::cout << "    H1 = " << -1.*blf_M.InnerProduct(u,w) << "\n";
    std::cout << "    H2 = " << -1.*blf_N.InnerProduct(v,z) << "\n";

    // check eq 47b,46b
    // mfem::Vector vec_47b (z.Size()); vec_47b=0.;
    // C.Mult(u,vec_47b);
    // N_n.AddMult(z,vec_47b);
    // mfem::Vector vec_46b (w.Size()); vec_46b=0.;
    // CT->Mult(v, vec_46b);
    // M_n.AddMult(w,vec_46b);
    // std::cout << "---eq 46b, 47b---\n" << vec_47b.Norml2() << "\n";
    // std::cout << vec_46b.Norml2() << "\n";

    // eq 46a,47a
    // mfem::Vector vec_47a (u.Size()); vec_47a=0.;
    // mfem::Vector vec_46a (v.Size()); vec_46a=0.;

    // time loop
    double T;
    for (int i = 1 ; i <= timesteps ; i++) {
        T = i*dt;
        std::cout << "iter="<<i<<"-------------------------\n";
        
        // update R1
        mfem::MixedBilinearForm blf_R1(&ND,&ND);
        mfem::SparseMatrix R1;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_R1.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_R1.Assemble();
        blf_R1.FormRectangularSystemMatrix(ND_etdof,ND_etdof,R1);
        R1 *= 1./2.;
        R1.Finalize();

        // update R2
        mfem::MixedBilinearForm blf_R2(&RT,&RT);
        mfem::SparseMatrix R2;
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_R2.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_R2.Assemble();
        blf_R2.FormRectangularSystemMatrix(RT_etdof,RT_etdof,R2);
        R2 *= 1./2.;
        R2.Finalize();

        // M+R and N+R
        mfem::SparseMatrix MR = M_dt;
        mfem::SparseMatrix NR = N_dt;
        MR.Add(1,R1);
        NR.Add(1,R2);
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

        // conserved quantities
        // GT->Mult(u,mass_vec1);
        // D.Mult(v,mass_vec2);
        // std::cout << "div(u) = " << mass_vec1.Norml2() << "\n";
        // std::cout << "div(v) = " << mass_vec2.Norml2() << "\n";
        // std::cout << "    E1 = " << -1./2.*blf_M.InnerProduct(u,u) << "\n";
        // std::cout << "    E2 = " << -1./2.*blf_N.InnerProduct(v,v) << "\n";
        std::cout << "    H1 = " << -1.*blf_M.InnerProduct(u,w) << "\n";
        std::cout << "    H2 = " << -1.*blf_N.InnerProduct(v,z) << "\n";

        // eq 47b,46b
        // vec_47b=0.;
        // C.Mult(u,vec_47b);
        // N_n.AddMult(z,vec_47b);
        // vec_46b=0.;
        // CT->Mult(v, vec_46b);
        // M_n.AddMult(w,vec_46b);
        // std::cout << "---eq 47b, 46b---\n"<<vec_47b.Norml2() << "\n";
        // std::cout << vec_46b.Norml2() << "\n";
    
        // diff and avg values
        // TODO: fix 1/2 and 1/dt factors for udiff, uavg, ... use def above time loop
        udiff = u;
        udiff.Add(-1.,uold);
        uavg = u;
        uavg.Add(1.,uold);
        zavg = z;
        zavg.Add(1.,zold);
        vdiff = v;
        vdiff.Add(-1.,vold);
        vavg = v;
        vavg.Add(1.,vold);
        wavg = w;
        wavg.Add(1.,wold);
        wdiff = w;
        wdiff.Add(-1,wold);

        // eq 47a,46a
        // vec_47a = 0.;
        // M_dt.AddMult(udiff,vec_47a);
        // R1.AddMult(uavg,vec_47a); 
        // CT_Re.AddMult(zavg,vec_47a); 
        // G.AddMult(p,vec_47a);
        // vec_46a = 0.;
        // N_dt.AddMult(vdiff,vec_46a);
        // R2.AddMult(vavg,vec_46a);
        // C_Re.AddMult(wavg,vec_46a);
        // DT_n->AddMult(q,vec_46a);
        // std::cout <<"---eq 47a,46a---\n"<<vec_47a.Norml2() << "\n";
        // std::cout << vec_46a.Norml2() << "\n";

        // eq 27a,26a: all 4 terms for energy cons
        // std::cout<<"---eq 27a,26a term1---\n"<<-1.*M_n.InnerProduct(udiff,uavg);
        // std::cout << "\n"<<-1.*N_n.InnerProduct(vdiff,vavg)<<"\n";
        // std::cout<<"---eq 27a,26a term2---\n"<<R1.InnerProduct(uavg,uavg);
        // std::cout << "\n"<<R2.InnerProduct(vavg,vavg)<<"\n";
        // std::cout<<"---eq 27a,26a term3---\n"<<CT->InnerProduct(zavg,uavg);
        // std::cout << "\n"<<C.InnerProduct(wavg,vavg)<<"\n";
        // std::cout << "---eq 27a,26a term4---\n"<<G.InnerProduct(p,uavg);
        // std::cout << "\n"<<DT_n->InnerProduct(q,vavg)<<"\n";

        
        // helcitiy difference to prev timestep 

        // M_n.PrintInfo(std::cout);
        std::cout<<"---eq 32,33---\n"<< "delta H1a = "
        <<-1.*M_n.InnerProduct(udiff,w);
        std::cout << "\n"<<"delta H1b = "
        << -1.*M_n.InnerProduct(u,wdiff)<< "\n";
        // M_n.PrintInfo(std::cout);

        // TODO if this works, and euler not => euler is the problem

        // M_n.PrintInfo(std::cout);
        // std::cout << udiff.Norml2() << "\n";
        // std::cout << w.Norml2() << "\n";
        // std::cout << u.Norml2() << "\n";
 
        // double a = M_n.InnerProduct(udiff,w);
        // double b = M_n.InnerProduct(u,wdiff);
        // std::cout<<"---\n";

        // std::cout << udiff.Norml2() << "\n";
        // std::cout << w.Norml2() << "\n";
        // std::cout << u.Norml2() << "\n";
        // M_n.PrintInfo(std::cout);




        // TODO implement terms from vorticity cons proof

        // update old (previous time step) values for next time step
        uold = u;
        vold = v;
        zold = z;
        wold = w;
    }

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

    returnvalue(0) = std::cos(pi*x(3));
    returnvalue(1) = std::sin(pi*x(3)); 
    returnvalue(2) = std::sin(pi*x(1));
}

void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    int dim = x.Size();

    returnvalue(0) = -pi*std::cos(pi*x(3));
    returnvalue(1) = -pi*std::cos(pi*x(1)) -pi*std::sin(pi*x(3)); 
    returnvalue(2) = 0;
}