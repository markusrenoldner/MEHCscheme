#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"




// MEHC scheme for dirichlet problem
// essential BC at Hdiv and Hcurl of dual system only


// use wouters trick for the BC




struct Parameters {
    double Re_inv = 1; // = 1/Re 
    double dt     = 0.1;
    double tmax   = 1*dt;
    int ref_steps = 3;
    int init_ref  = 0;
    int order     = 1;
    std::string outputfile = "out/rawdata/periodic-conv-invisc.txt";
    const char* mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    double t;
};

void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void     u_0(const mfem::Vector &x, mfem::Vector &v);
void     w_0(const mfem::Vector &x, mfem::Vector &v);
void       f(const mfem::Vector &x, mfem::Vector &v); 


int main(int argc, char *argv[]) {

    // simulation parameters
    Parameters param;
    double Re_inv = param.Re_inv; 
    double dt     = param.dt;
    double tmax   = param.tmax;
    int ref_steps = param.ref_steps;
    int init_ref  = param.init_ref;
    int order     = param.order;
    std::string outputfile = param.outputfile;

    // output file 
    std::ofstream file;
    file.precision(6);
    file.open(outputfile);
    // file.open(outputfile, std::ios::app);

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        auto start = std::chrono::high_resolution_clock::now();

        // mesh
        const char *mesh_file = param.mesh_file;
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        int l;
        dt *= 0.5; // TODO
        for (l = 0; l<init_ref+ref_step; l++) {
            mesh.UniformRefinement();
        } 
        std::cout << "----------ref: " << ref_step << "----------\n";

        // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);
        mfem::FiniteElementSpace RT(&mesh, fec_RT);
        mfem::FiniteElementSpace CG(&mesh, fec_CG);

        // TODO
        // boundary arrays: contain indices of essential boundary DOFs
        // mfem::Array<int> ND_ess_tdof;
        // mfem::Array<int> RT_ess_tdof;
        // mfem::Array<int> ND_ess_tdof_0;
        // mfem::Array<int> RT_ess_tdof_0;
        // ND.GetBoundaryTrueDofs(ND_ess_tdof); 
        // RT.GetBoundaryTrueDofs(RT_ess_tdof); 
        // ND.GetBoundaryTrueDofs(ND_ess_tdof_0); 
        // RT.GetBoundaryTrueDofs(RT_ess_tdof_0); 

        // // concatenation of essdof arrays
        // mfem::Array<int> ess_dof1, ess_dof2;
        // ess_dof2.Append(RT_ess_tdof);
        // for (int i=0; i<ND_ess_tdof.Size(); i++) {
        //     ND_ess_tdof[i] += RT.GetNDofs() ;
        // }
        // ess_dof2.Append(ND_ess_tdof);

        // essdofs
        mfem::Array<int> RT_ess_tdof;
        mfem::Array<int> ND_ess_tdof;
        mfem::Array<int> ND_ess_tdof_0;
        mfem::Array<int> ess_dof2;
        ND.GetBoundaryTrueDofs(ND_ess_tdof); 
        ND.GetBoundaryTrueDofs(ND_ess_tdof_0); 
        RT.GetBoundaryTrueDofs(ess_dof2);
        RT.GetBoundaryTrueDofs(RT_ess_tdof);
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            ND_ess_tdof[i] += RT.GetNDofs() ;
        }
        ess_dof2.Append(ND_ess_tdof);

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND); //u = 4.3;
        mfem::GridFunction z(&RT); //z = 5.3;
        mfem::GridFunction p(&CG); p=0.; //p = 6.3;
        mfem::GridFunction v(&RT); //v = 3.;
        mfem::GridFunction w(&ND); //w = 3.; 
        mfem::GridFunction q(&DG); q=0.; //q = 9.3;

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        u.ProjectCoefficient(u_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w.ProjectCoefficient(w_0_coeff);

        // linearform for forcing term
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::LinearForm f1(&ND);
        mfem::LinearForm f2(&RT);
        f1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f1.Assemble();
        f2.Assemble();

        // boundary integral for primal reynolds term
        mfem::LinearForm lform_zxn(&ND);
        lform_zxn.AddBoundaryIntegrator(new mfem::VectorFEBoundaryTangentLFIntegrator(w_0_coeff)); // !!!
        lform_zxn.Assemble();
        lform_zxn *= -1.*Re_inv;

        // boundary integral für div-free cond
        mfem::LinearForm lform_un(&CG);
        lform_un.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(u_0_coeff));
        lform_un.Assemble();
        lform_un *= Re_inv;
        // lform_un *= -1.;

        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        int size_2 = v.Size() + w.Size() + q.Size();
        // std::cout<< "size1/u/z/p: "<<size_1<<"/"<<u.Size()<<"/"<<z.Size()<<"/"<<p.Size()<<"\n";
        // std::cout<< "size2/v/w/q/lam: "<<size_2<<"/"<<v.Size()<<"/"<<w.Size()<<"/"<<q.Size()<<"\n";
        
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

        // Matrix M
        mfem::BilinearForm blf_M(&ND);
        blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_M.Assemble();
        blf_M.Finalize();
        mfem::SparseMatrix M_n(blf_M.SpMat());
        mfem::SparseMatrix M_dt;
        M_dt = M_n;
        M_dt *= 1/dt;
        M_n *= -1.;
        M_dt.Finalize();
        M_n.Finalize();
        
        // Matrix N
        mfem::BilinearForm blf_N(&RT);
        blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_N.Assemble();
        blf_N.Finalize();
        mfem::SparseMatrix N_n(blf_N.SpMat());
        mfem::SparseMatrix N_dt;
        N_dt = N_n;
        N_dt *= 1/dt;
        N_n *= -1.;
        N_dt.Finalize();
        N_n.Finalize();

        // Matrix C
        mfem::MixedBilinearForm blf_C(&ND, &RT);
        blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator()); //=(curl u,v)
        blf_C.Assemble();
        blf_C.Finalize();
        mfem::SparseMatrix C(blf_C.SpMat());
        mfem::SparseMatrix *CT;
        mfem::SparseMatrix C_Re;
        mfem::SparseMatrix CT_Re;
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
        blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator()); //=(div u,v)
        blf_D.Assemble();
        blf_D.Finalize();
        mfem::SparseMatrix D(blf_D.SpMat());
        mfem::SparseMatrix *DT_n;
        DT_n = Transpose(D);
        *DT_n *= -1.;
        D.Finalize();
        DT_n->Finalize();

        // Matrix G
        mfem::MixedBilinearForm blf_G(&CG, &ND);
        blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
        blf_G.Assemble();
        blf_G.Finalize();
        mfem::SparseMatrix G(blf_G.SpMat());
        mfem::SparseMatrix *GT;
        GT = Transpose(G);
        G.Finalize();
        GT->Finalize();    








        // TODO : enforce ess dofs hardcore
        // matrix E2_left
        int rows_E2 = ess_dof2.Size();
        mfem::SparseMatrix E2_left (rows_E2, v.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            E2_left.Set(i, RT_ess_tdof[i], 1.);
        }
        E2_left.Finalize();
        std::cout << rows_E2 << " " << v.Size() << " "<<w.Size()<<" "<<ess_dof2.Size()<<"\n";

        // matrix E2_cent
        mfem::SparseMatrix E2_cent (rows_E2, w.Size());
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            E2_cent.Set(i + RT_ess_tdof.Size(), ND_ess_tdof_0[i], 1.);
        }
        E2_cent.Finalize();
        // mfem::DenseMatrix* dense = E2_cent.ToDenseMatrix();
        // dense->PrintMatlab(std::cout);

        // matrix E2_right
        mfem::SparseMatrix E2_right (rows_E2, q.Size());
        E2_right = 0.;
        E2_right.Finalize();

        // vector e2
        mfem::Vector e2(ess_dof2.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            e2[i] = v[RT_ess_tdof[i]];
        }
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            e2[i + RT_ess_tdof.Size()] = w[ND_ess_tdof_0[i]];
            // std::cout << i + RT_ess_tdof.Size()  << " " << ND_ess_tdof_0[i] << "\n";
        }








        // initialize system matrices
        mfem::Array<int> offsets_1 (4);
        offsets_1[0] = 0;
        offsets_1[1] = u.Size();
        offsets_1[2] = z.Size();
        offsets_1[3] = p.Size();
        offsets_1.PartialSum(); // exclusive scan
        mfem::BlockOperator A1(offsets_1);

        mfem::Array<int> offsets_2 (4);
        offsets_2[0] = 0;
        offsets_2[1] = v.Size();
        offsets_2[2] = w.Size();
        offsets_2[3] = q.Size();
        offsets_2.PartialSum();

        mfem::Array<int> offsets_2_rows (5);
        offsets_2_rows[0] = 0;
        offsets_2_rows[1] = v.Size();
        offsets_2_rows[2] = w.Size();
        offsets_2_rows[3] = q.Size();
        offsets_2_rows[4] = ess_dof2.Size();
        offsets_2_rows.PartialSum();
        
        mfem::BlockOperator A2(offsets_2_rows, offsets_2);

        // initialize rhs
        mfem::Vector b1(size_1);
        mfem::Vector b1sub(u.Size());
        mfem::Vector b2(size_2 + ess_dof2.Size()); 
        mfem::Vector b2sub(v.Size());

        ////////////////////////////////////////////////////////////////////////////
        // forcing function constructed by initial value of gridfunctions
        ////////////////////////////////////////////////////////////////////////////
        
        // R2 
        // mfem::MixedBilinearForm blf_R2(&RT,&RT);
        // mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        // blf_R2.AddDomainIntegrator(
        //     new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        // blf_R2.Assemble();
        // blf_R2.Finalize();
        // mfem::SparseMatrix R2(blf_R2.SpMat());
        // R2 *= 1./2.;
        // R2.Finalize();
        // // R1
        // mfem::MixedBilinearForm blf_R1(&ND,&ND);
        // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        // blf_R1.AddDomainIntegrator(
        //     new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        // blf_R1.Assemble();
        // blf_R1.Finalize();
        // mfem::SparseMatrix R1(blf_R1.SpMat());
        // R1 *= 1./2.;
        // R1.Finalize();

        // mfem::Vector bf1 (u.Size()); 
        // bf1=0.;
        // R1.AddMult(u,bf1,2);
        // CT_Re.AddMult(z,bf1,2);
        // mfem::Vector bf2 (v.Size()); 
        // bf2=0.;
        // R2.AddMult(v,bf2,2);
        // C_Re.AddMult(w,bf2,2);
        
        // mfem::GridFunction bf1_gf(&ND);
        // mfem::GridFunction bf2_gf(&RT);
        // bf1_gf=0.;
        // bf2_gf=0.;
        // double tol = 1e-10;
        // int iter = 1000000;  
        // mfem::MINRES(M_n, bf1, bf1_gf, 0, iter, tol*tol, tol*tol);
        // mfem::MINRES(N_n, bf2, bf2_gf, 0, iter, tol*tol, tol*tol);

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        mfem::ConstantCoefficient two_over_dt(2.0/dt);
        blf_MR_eul.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_MR_eul.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_MR_eul.Assemble();
        blf_MR_eul.Finalize();
        mfem::SparseMatrix MR_eul(blf_MR_eul.SpMat());
        MR_eul.Finalize();
        
        // CT for eulerstep
        mfem::SparseMatrix CT_eul = CT_Re;
        CT_eul *= 2;
        CT_eul.Finalize();

        // assemble and solve system
        A1.SetBlock(0,0, &MR_eul);
        A1.SetBlock(0,1, &CT_eul);
        A1.SetBlock(0,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &N_n);
        A1.SetBlock(2,0, GT);

        // update b1, b2 for eulerstep
        b1 = 0.0;
        b1sub = 0.0;
        M_dt.AddMult(u,b1sub,2);
        b1.AddSubVector(f1,0);
        // b1.AddSubVector(bf1,0); 
        b1.AddSubVector(b1sub,0);
        b1.AddSubVector(lform_zxn, 0); // NEU
        b1.AddSubVector(lform_un, u.Size() + z.Size());

        // transpose here:
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // form linear system with BC
        // mfem::Operator *A1_BC;
        // mfem::Operator *A2_BC;
        // mfem::Vector X;
        // mfem::Vector B1;
        // mfem::Vector Y;
        // mfem::Vector B2;
        // ATA1.FormLinearSystem(ess_dof1, x, ATb1, A1_BC, X, B1);

        // solve 
        double tol = 1e-10;
        int iter = 10000000;  
        // mfem::MINRES(*A1_BC, B1, X, 0, iter, tol*tol, tol*tol);
        mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        // ATA1.RecoverFEMSolution(X, b1, x);
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);

        // time loop
        double t;
        for (t = dt ; t < tmax+dt ; t+=dt) {

            // std::cout << "--- t = "<<t<<"\n";
            // std::cout << t << ",";

            ////////////////////////////////////////////////////////////////////
            // DUAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R2 
            mfem::MixedBilinearForm blf_R2(&RT,&RT);
            mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
            blf_R2.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_R2.Assemble();
            blf_R2.Finalize();
            mfem::SparseMatrix R2(blf_R2.SpMat());
            R2 *= 1./2.;
            R2.Finalize();

            // update NR
            mfem::MixedBilinearForm blf_NR(&RT,&RT); 
            blf_NR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_NR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_NR.Assemble();
            blf_NR.Finalize();
            mfem::SparseMatrix NR(blf_NR.SpMat());
            NR *= 1./2.;
            NR.Finalize();

            // update A2
            A2.SetBlock(0,0, &NR);
            A2.SetBlock(0,1, &C_Re);
            A2.SetBlock(0,2, DT_n);
            A2.SetBlock(1,0, CT);
            A2.SetBlock(1,1, &M_n);
            A2.SetBlock(2,0, &D);

            A2.SetBlock(3,0, &E2_left); //TODO
            A2.SetBlock(3,1, &E2_cent);
            A2.SetBlock(3,2, &E2_right);
            
            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N_dt.AddMult(v,b2sub);
            R2.AddMult(v,b2sub,-1);
            C_Re.AddMult(w,b2sub,-1);
            b2.AddSubVector(f2,0); 
            // b2.AddSubVector(bf2,0);
            b2.AddSubVector(b2sub,0);
            b2.AddSubVector(e2, size_2);

            // transpose here:
            mfem::TransposeOperator AT2 (&A2);
            mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
            mfem::Vector ATb2 (size_2);
            A2.MultTranspose(b2,ATb2);

            // form linear system with BC
            // ATA2.FormLinearSystem(ess_dof2, y, ATb2, A2_BC, Y, B2);

            // solve  
            // mfem::MINRES(*A2_BC, B2, Y, 0, iter, tol*tol, tol*tol);
            mfem::MINRES(ATA2, ATb2, y, 0, iter, tol*tol, tol*tol);
            // ATA2.RecoverFEMSolution(Y, b2, y);
            y.GetSubVector(v_dofs, v);
            y.GetSubVector(w_dofs, w);
            y.GetSubVector(q_dofs, q);                

            ////////////////////////////////////////////////////////////////////
            // PRIMAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R1
            mfem::MixedBilinearForm blf_R1(&ND,&ND);
            mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
            blf_R1.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_R1.Assemble();
            blf_R1.Finalize();
            mfem::SparseMatrix R1(blf_R1.SpMat());
            R1 *= 1./2.;
            R1.Finalize();

            // update MR
            mfem::MixedBilinearForm blf_MR(&ND,&ND); 
            blf_MR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_MR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_MR.Assemble();
            blf_MR.Finalize();
            mfem::SparseMatrix MR(blf_MR.SpMat());
            MR *= 1./2.;
            MR.Finalize();

            // update A1
            A1.SetBlock(0,0, &MR);
            A1.SetBlock(0,1, &CT_Re);
            A1.SetBlock(0,2, &G);
            A1.SetBlock(1,0, &C);
            A1.SetBlock(1,1, &N_n);
            A1.SetBlock(2,0, GT);

            // update b1
            b1 = 0.0;
            b1sub = 0.0;
            M_dt.AddMult(u,b1sub);
            R1.AddMult(u,b1sub,-1);
            CT_Re.AddMult(z,b1sub,-1);
            b1.AddSubVector(b1sub,0);
            b1.AddSubVector(f1,0);
            b1.AddSubVector(lform_zxn, 0); // NEU
            b1.AddSubVector(lform_un, u.Size() + z.Size());
            // b1.AddSubVector(bf1,0);

            //Transposition
            mfem::TransposeOperator AT1 (&A1);
            mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
            mfem::Vector ATb1 (size_1);
            A1.MultTranspose(b1,ATb1);

            // form linear system with BC
            // ATA1.FormLinearSystem(ess_dof1, x, ATb1, A1_BC, X, B1);

            // solve 
            // mfem::MINRES(*A1_BC, B1, X, 0, iter, tol*tol, tol*tol); 
            mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);
            // ATA1.RecoverFEMSolution(X, b1, x);
            x.GetSubVector(u_dofs, u);
            x.GetSubVector(z_dofs, z);
            x.GetSubVector(p_dofs, p);

            ////////////////////////////////////////////////////////////////////
            // CONSERVATION
            ////////////////////////////////////////////////////////////////////
            
            // energy conservation 
            // double K1 = 1./2.*blf_M.InnerProduct(u,u);
            // double K2 = 1./2.*blf_N.InnerProduct(v,v);
            // std::cout <<std::abs(K1) << ",\n" << std::abs(K2) << ",\n";
            // file <<std::setprecision(15)<< std::fixed<<K1<< ","
            //               << K2 << ",\n";

        } // time loop

        // convergence error 
        double err_L2_u = u.ComputeL2Error(u_0_coeff);
        double err_L2_v = v.ComputeL2Error(u_0_coeff);
        mfem::GridFunction v_ND (&ND);
        v_ND.ProjectGridFunction(v);
        double err_L2_diff = 0;
        for (int i=0; i<u.Size(); i++) {
            err_L2_diff += ((u(i)-v_ND(i))*(u(i)-v_ND(i)));
        }
        std::cout << "L2err of v = "<< err_L2_v<<"\n";
        std::cout << "L2err of u = "<< err_L2_u<<"\n";
        std::cout << "L2err(u-v) = "<< std::pow(err_L2_diff, 0.5) <<"\n";
        // file <<std::setprecision(15)<< std::fixed<<err_L2_diff<< ","
        //               << err_L2_u << ",\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        std::cout << "runtime = " << duration.count() << "ms" << std::endl;

        // visuals    
        // char vishost[] = "localhost";
        // int  visport   = 19916;
        // mfem::socketstream u_sock(vishost, visport);
        // u_sock.precision(8);
        // u_sock << "solution\n" << mesh << u << "window_title 'u in hcurl'" << std::endl;
        
        // mfem::socketstream v_sock(vishost, visport);
        // v_sock.precision(8);
        // v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;
        
        // mfem::GridFunction u_exact(&ND);
        // u_exact.ProjectCoefficient(u_0_coeff);
        // mfem::socketstream ue_sock(vishost, visport);
        // ue_sock.precision(8);
        // ue_sock << "solution\n" << mesh << u_exact << "window_title 'u_exact'" << std::endl;

        // mfem::socketstream p_sock(vishost, visport);
        // p_sock.precision(8);
        // p_sock << "solution\n" << mesh << p << "window_title 'p in H1'" << std::endl;    

        // free memory
        delete fec_DG;
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;

    } // refinement loop

    // close file
    file.close();
}


void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = std::sin(Y);
    returnvalue(1) = std::sin(Z);
    returnvalue(2) = 0.;
}
void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = -std::cos(Z);
    returnvalue(1) = 0.;
    returnvalue(2) = -std::cos(Y);
}
void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    Parameters param;
    double Re_inv = param.Re_inv; // = 1/Re 
    
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = std::sin(Y)*Re_inv + std::cos(Y)*std::sin(Z);
    returnvalue(1) = -std::cos(Y)*std::sin(Y) + std::sin(Z)*Re_inv;
    returnvalue(2) = - std::cos(Z)*std::sin(Z);
}