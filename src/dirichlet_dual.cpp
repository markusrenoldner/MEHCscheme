#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

#include <chrono>



// MEHC scheme for dirichlet problem
// essential BC at Hdiv and Hcurl of dual system only




void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void     u_0(const mfem::Vector &x, mfem::Vector &v);
void     w_0(const mfem::Vector &x, mfem::Vector &v);
void       f(const mfem::Vector &x, mfem::Vector &v); 
// void u_exact(const mfem::Vector &x, mfem::Vector &returnvalue);

int main(int argc, char *argv[]) {

    // simulation parameters
    // careful: Re also has to be defined in the manufactured sol
    double Re_inv = 0.01; // = 1/Re 
    double dt = 2;
    double tmax = 30*dt;
    int ref_steps = 0;
    // std::cout <<"----------\n"<<"Re:   "<<1/Re_inv <<"\ndt:   "<<dt<< "\ntmax: "<<tmax<<"\n----------\n";

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        auto start = std::chrono::high_resolution_clock::now();

        // mesh
        const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        for (int l = 0; l<ref_step; l++) {mesh.UniformRefinement();} 
        std::cout << "----------ref: " << ref_step << "----------\n";
        mesh.UniformRefinement();
        mesh.UniformRefinement();
        // mesh.UniformRefinement();
        // mesh.UniformRefinement();

        // TODO rename FEM spaces (remove the zero)

        // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
        int order = 1;
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);
        mfem::FiniteElementSpace RT(&mesh, fec_RT);
        mfem::FiniteElementSpace CG(&mesh, fec_CG);

        // boundary arrays: contain indices of essential boundary DOFs
        mfem::Array<int> ND_ess_tdof;
        mfem::Array<int> RT_ess_tdof;
        ND.GetBoundaryTrueDofs(ND_ess_tdof); 
        RT.GetBoundaryTrueDofs(RT_ess_tdof); 

        // concatenation of essdof arrays
        mfem::Array<int> ess_dof1, ess_dof2;
        ess_dof2.Append(RT_ess_tdof);
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
        // mfem::GridFunction f1(&ND);
        // mfem::GridFunction f2(&RT);
        mfem::GridFunction u_exact(&ND);

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        // mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_0); 
        u.ProjectCoefficient(u_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        // f1.ProjectCoefficient(f_coeff);
        // f2.ProjectCoefficient(f_coeff);
        u_exact.ProjectCoefficient(u_exact_coeff);
    
        // helper vectors for old values
        mfem::Vector u_old(u.Size()); u_old = 0.;
        mfem::Vector v_old(v.Size()); v_old = 0.;
        mfem::Vector z_old(z.Size()); z_old = 0.;
        mfem::Vector w_old(w.Size()); w_old = 0.;
        mfem::Vector u_old_old(u.Size()); u_old_old = 0.;
        mfem::Vector v_old_old(v.Size()); v_old_old = 0.;
        mfem::Vector z_old_old(z.Size()); z_old_old = 0.;

        // helper vectors for average values
        mfem::Vector u_avg (u.Size()); 
        mfem::Vector z_avg (z.Size()); 
        mfem::Vector v_avg (v.Size()); 
        mfem::Vector w_avg (w.Size()); 
        mfem::Vector u_avg_old (u.Size());
        mfem::Vector v_avg_old (v.Size());
        mfem::Vector z_avg_old (z.Size());

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
        // offsets_2[4] = 1;
        offsets_2.PartialSum();
        mfem::BlockOperator A1(offsets_1);
        mfem::BlockOperator A2(offsets_2);

        // initialize rhs
        mfem::Vector b1(size_1);
        mfem::Vector b1sub(u.Size());
        mfem::Vector b2(size_2); 
        mfem::Vector b2sub(v.Size());











        ////////////////////////////////////////////////////////////////////////////
        // forcing function
        // NEU
        
        // R2 
        mfem::MixedBilinearForm blf_R2(&RT,&RT);
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
        blf_R2.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_R2.Assemble();
        blf_R2.Finalize();
        mfem::SparseMatrix R2(blf_R2.SpMat());
        R2 *= 1./2.;
        R2.Finalize();
        // R1
        mfem::MixedBilinearForm blf_R1(&ND,&ND);
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        blf_R1.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_R1.Assemble();
        blf_R1.Finalize();
        mfem::SparseMatrix R1(blf_R1.SpMat());
        R1 *= 1./2.;
        R1.Finalize();

        mfem::Vector bf1 (u.Size()); 
        bf1=0.;
        R1.AddMult(u,bf1,2);
        CT_Re.AddMult(z,bf1,2);
        mfem::Vector bf2 (v.Size()); 
        bf2=0.;
        R2.AddMult(v,bf2,2);
        C_Re.AddMult(w,bf2,2);
        
        mfem::GridFunction bf1_gf(&ND);
        mfem::GridFunction bf2_gf(&RT);
        bf1_gf=0.;
        bf2_gf=0.;
        double tol = 1e-10;
        int iter = 1000000;  
        mfem::MINRES(M_n, bf1, bf1_gf, 0, iter, tol*tol, tol*tol);
        mfem::MINRES(N_n, bf2, bf2_gf, 0, iter, tol*tol, tol*tol);









        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
        // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
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
        // b1.AddSubVector(f1,0);
        b1.AddSubVector(bf1,0);
        b1.AddSubVector(b1sub,0);

        // transpose here:
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // form linear system with BC
        mfem::Operator *A1_BC;
        mfem::Operator *A2_BC;
        mfem::Vector X;
        mfem::Vector B1;
        mfem::Vector Y;
        mfem::Vector B2;
        ATA1.FormLinearSystem(ess_dof1, x, ATb1, A1_BC, X, B1);

        // solve 
        // double tol = 1e-10;
        // int iter = 1000000;  
        mfem::MINRES(*A1_BC, B1, X, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        A1.RecoverFEMSolution(X, b1, x);
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);

        // time loop
        for (double t = dt ; t < tmax+dt ; t+=dt) {
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
            
            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N_dt.AddMult(v,b2sub);
            R2.AddMult(v,b2sub,-1);
            C_Re.AddMult(w,b2sub,-1);
            // b2.AddSubVector(f2,0);
            b2.AddSubVector(bf2,0);
            b2.AddSubVector(b2sub,0);

            // transpose here:
            mfem::TransposeOperator AT2 (&A2);
            mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
            mfem::Vector ATb2 (size_2);
            A2.MultTranspose(b2,ATb2);

            // form linear system with BC
            ATA2.FormLinearSystem(ess_dof2, y, ATb2, A2_BC, Y, B2);

            // solve  
            mfem::MINRES(*A2_BC, B2, Y, 0, iter, tol*tol, tol*tol);
            A2.RecoverFEMSolution(Y, b2, y);
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
            // b1.AddSubVector(f1,0);
            b1.AddSubVector(bf1,0);

            //Transposition
            mfem::TransposeOperator AT1 (&A1);
            mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
            mfem::Vector ATb1 (size_1);
            A1.MultTranspose(b1,ATb1);

            // form linear system with BC
            ATA1.FormLinearSystem(ess_dof1, x, ATb1, A1_BC, X, B1);

            // solve 
            mfem::MINRES(*A1_BC, B1, X, 0, iter, tol*tol, tol*tol); 
            A1.RecoverFEMSolution(X, b1, x);
            x.GetSubVector(u_dofs, u);
            x.GetSubVector(z_dofs, z);
            x.GetSubVector(p_dofs, p);

            ////////////////////////////////////////////////////////////////////
            // CONSERVATION
            ////////////////////////////////////////////////////////////////////

            // averaged values
            u_avg = 0.;
            u_avg.Add(0.5,u);
            u_avg.Add(0.5,u_old);
            z_avg = 0.;
            z_avg.Add(0.5,z);
            z_avg.Add(0.5,z_old);
            v_avg = 0.;
            v_avg.Add(0.5,v);
            v_avg.Add(0.5,v_old);
            w_avg = 0.;
            w_avg.Add(0.5,w);
            w_avg.Add(0.5,w_old);

            // averaged old values
            u_avg_old = 0.;
            u_avg_old.Add(0.5,u_old);
            u_avg_old.Add(0.5,u_old_old);
            v_avg_old = 0.;
            v_avg_old.Add(0.5,v_old);
            v_avg_old.Add(0.5,v_old_old);
            z_avg_old = 0.;
            z_avg_old.Add(0.5,z_old);
            z_avg_old.Add(0.5,z_old_old);

            // conservation test, Re=infty
            // TODO check signs of K1K2H1H2
            mfem::Vector mass_vec1 (p.Size());
            mfem::Vector mass_vec2 (q.Size());
            GT->Mult(u,mass_vec1);
            D.Mult(v,mass_vec2);
            // double K1_old = -1./2.*blf_M.InnerProduct(u_old,u_old);
            // double K1 = -1./2.*blf_M.InnerProduct(u,u);
            // double K2_old = -1./2.*blf_N.InnerProduct(v_old,v_old);
            // double K2 = -1./2.*blf_N.InnerProduct(v,v);
            // double H1_old = -1.*blf_M.InnerProduct(u_avg_old,w_old);
            // double H1 = -1.*blf_M.InnerProduct(u_avg,w);
            // double H2_old = -1.*blf_N.InnerProduct(v_old,z_avg_old); 
            // double H2 = -1.*blf_N.InnerProduct(v,z_avg); //definition in paper!!
            // std::cout << mass_vec1.Norml2() << ",";
            // std::cout << mass_vec2.Norml2() << ",";
            // std::cout << (K1-K1_old)/dt << ",";
            // std::cout << (K2-K2_old)/dt << ",\n";
            // std::cout << (H1-H1_old)/dt << ",";
            // std::cout << (H2-H2_old)/dt << ",\n"; 
            
            // conservation test, Re=100
            double E2 = 1/2.*blf_N.InnerProduct(z_avg,z_avg);
            double E1 = 1/2.*blf_M.InnerProduct(w_avg,w_avg);
            double D = -Re_inv*C.InnerProduct(w_avg,z_old)
                       -Re_inv/2*CT->InnerProduct(z_avg,w)
                       -Re_inv/2*CT->InnerProduct(z_avg_old,w_old); 
            // std::cout << (K1-K1_old)/dt - 2*Re_inv*E2 << ",";
            // std::cout << (K2-K2_old)/dt - 2*Re_inv*E1 << ",\n";
            // std::cout << (H1-H1_old)/dt - D << ",";
            // std::cout << (H2-H2_old)/dt - D << ",\n";

            

            double K1 = 1./2.*blf_M.InnerProduct(u,u);
            double K2 = 1./2.*blf_N.InnerProduct(v,v);
            std::cout <<std::abs(K1) << ",\n" << std::abs(K2) << ",\n";






        } // time loop

        // convergence error
        double err_L2_u = u.ComputeL2Error(u_exact_coeff);
        double err_L2_v = v.ComputeL2Error(u_exact_coeff);
        mfem::GridFunction v_ND (&ND);
        v_ND.ProjectGridFunction(v);
        double err_L2_diff = 0;
        for (int i=0; i<u.Size(); i++) {
            err_L2_diff += ((u(i)-v_ND(i))*(u(i)-v_ND(i)));
        }
        // std::cout << "L2err of v = "<< err_L2_v<<"\n";
        // std::cout << "L2err of u = "<< err_L2_u<<"\n";
        // std::cout << "L2err(u-v) = "<< std::pow(err_L2_diff, 0.5) <<"\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        std::cout << "runtime = " << duration.count() << "ms" << std::endl;

        // visuals
        // std::ofstream mesh_ofs("refined.mesh");
        // mesh_ofs.precision(8);
        // mesh.Print(mesh_ofs);
        // std::ofstream sol_ofs("sol.gf");
        // sol_ofs.precision(8);
        // u.Save(sol_ofs);
    
        char vishost[] = "localhost";
        int  visport   = 19916;
        // mfem::socketstream u_sock(vishost, visport);
        // u_sock.precision(8);
        // u_sock << "solution\n" << mesh << u << "window_title 'u in hcurl'" << std::endl;
        
        // mfem::socketstream v_sock(vishost, visport);
        // v_sock.precision(8);
        // v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;
        
        // mfem::socketstream w_sock(vishost, visport);
        // w_sock.precision(8);
        // w_sock << "solution\n" << mesh << w << "window_title 'w in hcurl'" << std::endl;
        
        // mfem::socketstream ue_sock(vishost, visport);
        // ue_sock.precision(8);
        // ue_sock << "solution\n" << mesh << u_exact << "window_title 'u_0'" << std::endl;
        
        // mfem::socketstream p_sock(vishost, visport);
        // p_sock.precision(8);
        // p_sock << "solution\n" << mesh << p << "window_title 'p in H1'" << std::endl;
        
        // mfem::socketstream q_sock(vishost, visport);
        // q_sock.precision(8);
        // q_sock << "solution\n" << mesh << q << "window_title 'q in L2'" << std::endl;
        
        // if (ref_step==0) {v_sock << "solution\n" << mesh << u << "window_title 'u in hdiv,1'" << std::endl;}
        // if (ref_step==1) {v_sock << "solution\n" << mesh << u << "window_title 'u in hdiv,2'" << std::endl;}
        // if (ref_step==2) {v_sock << "solution\n" << mesh << u << "window_title 'u in hdiv,3'" << std::endl;}
        // if (ref_step==3) {v_sock << "solution\n" << mesh << u << "window_title 'u in hdiv,4'" << std::endl;}
        // if (ref_step==4) {v_sock << "solution\n" << mesh << u << "window_title 'u in hdiv,5'" << std::endl;}

        // free memory
        delete fec_DG;
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;

    } // refinement loop
}


// cos squared init cond that satifies
// dirichlet BC, divu=0 and is C1-continuous (=> w is C continuous)
void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;
   
    double cos = std::cos(C*(X*X+Y*Y+Z*Z));
    double cos2 = cos*cos;

    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = Y * cos2;
        returnvalue(1) = -X * cos2;
        returnvalue(2) = 0;
    }
    else {
        returnvalue(0) = 0;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
}

void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;
   
    double cos = std::cos(C*(X*X+Y*Y+Z*Z));
    double sin = std::sin(C*(X*X+Y*Y+Z*Z));
    double cos2 = cos*cos;

    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = - 4*C*X*Z*sin*cos;
        returnvalue(1) = - 4*C*Y*Z*sin*cos;
        returnvalue(2) = - 2*cos2 + 4*C*X*X*sin*cos + 4*C*Y*Y*sin*cos;
    }
    else {
        returnvalue(0) = 0;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
}

// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1;
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
//     double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
//     double cos4 = cos*cos*cos*cos;
//     // double eC2 = 8*C*C;
    
//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = -X*cos4 + 1*Re_inv*( 
//                           2*Y*C*(4*C*(X*X+Y*Y+Z*Z)*cosof2 + 5*sinof2) );

//         returnvalue(1) = -Y*cos4 - 1*Re_inv*(
//                           2*X*C*(4*C*(X*X+Y*Y+Z*Z)*cosof2 + 5*sinof2) );


//         returnvalue(2) = 0.;
//     }
//     else {
//         returnvalue(0) = 0.; 
//         returnvalue(1) = 0.; 
//         returnvalue(2) = 0.;
//     }   
// }

        // returnvalue(0) = -X*cos4 + 1*Re_inv*( 
        //                   2*C*(4*C*(X*X+Y*Y+Z*Z)*cos+5*sin) );
        // returnvalue(1) = -Y*cos4 - 1*Re_inv*(
        //                   2*C*(4*C*(X*X+Y*Y+Z*Z)*cos+5*sin));