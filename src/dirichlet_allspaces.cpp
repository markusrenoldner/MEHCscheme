#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

#include <chrono>



// MEHC scheme on dirichlet domain
// all vector spaces adapted (H10, H0curl, H0div, L2 with lagr mult)




void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void     u_0(const mfem::Vector &x, mfem::Vector &v);
void     w_0(const mfem::Vector &x, mfem::Vector &v);
void       f(const mfem::Vector &x, mfem::Vector &v); 
// void u_exact(const mfem::Vector &x, mfem::Vector &returnvalue);

int main(int argc, char *argv[]) {

    // simulation parameters
    double Re_inv = 0.01; // = 1/Re 
    double dt = 1/20.;
    double tmax = 3*dt; //tmax=0.;
    int ref_steps = 0;
    std::cout <<"----------\n"<<"Re:   "<<1/Re_inv <<"\ndt:   "<<dt<< "\ntmax: "<<tmax<<"\n----------\n";

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
        mesh.UniformRefinement();
        mesh.UniformRefinement();

        // FE spaces; DG \in L2, ND \in Hcurl, RT \in Hdiv, CG \in H1
        int order = 1;
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_ND0 = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT0 = new mfem::RT_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_CG0 = new mfem::H1_FECollection(order,dim);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);
        mfem::FiniteElementSpace ND0(&mesh, fec_ND0);
        mfem::FiniteElementSpace RT0(&mesh, fec_RT0);
        mfem::FiniteElementSpace CG0(&mesh, fec_CG0);

        // boundary arrays: contain indices of essential boundary DOFs
        mfem::Array<int> ND0_ess_tdof;
        mfem::Array<int> RT0_ess_tdof;
        mfem::Array<int> CG0_ess_tdof;
        ND0.GetBoundaryTrueDofs(ND0_ess_tdof); 
        RT0.GetBoundaryTrueDofs(RT0_ess_tdof); 
        CG0.GetBoundaryTrueDofs(CG0_ess_tdof); 

        // concatenation of essdof arrays
        mfem::Array<int> ess_dof1, ess_dof2;
        ess_dof1.Append(ND0_ess_tdof);
        ess_dof2.Append(RT0_ess_tdof);
        for (int i=0; i<RT0_ess_tdof.Size(); i++) {
            RT0_ess_tdof[i] += ND0.GetNDofs();
        }
        ess_dof1.Append(RT0_ess_tdof);
        for (int i=0; i<CG0_ess_tdof.Size(); i++) {
            CG0_ess_tdof[i] += (ND0.GetNDofs()+RT0.GetNDofs() );
        }
        ess_dof1.Append(CG0_ess_tdof);
        for (int i=0; i<ND0_ess_tdof.Size(); i++) {
            ND0_ess_tdof[i] += RT0.GetNDofs() ;
        }
        ess_dof2.Append(ND0_ess_tdof);
        // no ess BC for DG space!

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND0); //u = 4.3;
        mfem::GridFunction z(&RT0); //z = 5.3;
        mfem::GridFunction p(&CG0); p=0.; //p = 6.3;
        mfem::GridFunction v(&RT0); //v = 3.;
        mfem::GridFunction w(&ND0); //w = 3.; 
        mfem::GridFunction q(&DG); q=0.; //q = 9.3;
        mfem::GridFunction f1(&ND0);
        mfem::GridFunction f2(&RT0);
        mfem::Vector lam (1); // lagrange multiplier
        lam[0] = 0.;

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_0); 
        u.ProjectCoefficient(u_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        f1.ProjectCoefficient(f_coeff);
        f2.ProjectCoefficient(f_coeff);

        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        int size_2 = v.Size() + w.Size() + q.Size() + 1;
        std::cout<< "size1/u/z/p: "<<size_1<<"/"<<u.Size()<<"/"<<z.Size()<<"/"<<p.Size()<<"\n";
        std::cout<< "size2/v/w/q/lam: "<<size_2<<"/"<<v.Size()<<"/"<<w.Size()<<"/"<<q.Size()<<"/"<<1<<"\n"<<"---\n";
        
        // initialize solution vectors
        mfem::Vector x(size_1);
        mfem::Vector y(size_2);
        x.SetVector(u,0);
        x.SetVector(z,u.Size());
        x.SetVector(p,u.Size()+z.Size());
        y.SetVector(v,0);
        y.SetVector(w,v.Size());
        y.SetVector(q,v.Size()+w.Size());
        y.SetVector(lam,v.Size()+w.Size()+1);

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
        mfem::Array<int> lam_dofs (1);
        lam_dofs[0] = size_2+1;

        // Matrix M0
        mfem::BilinearForm blf_M0(&ND0);
        blf_M0.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_M0.Assemble();
        blf_M0.Finalize();
        mfem::SparseMatrix M0_n(blf_M0.SpMat());
        mfem::SparseMatrix M0_dt;
        M0_dt = M0_n;
        M0_dt *= 1/dt;
        M0_n *= -1.;
        M0_dt.Finalize();
        M0_n.Finalize();
        
        // Matrix N0
        mfem::BilinearForm blf_N0(&RT0);
        blf_N0.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_N0.Assemble();
        blf_N0.Finalize();
        mfem::SparseMatrix N0_n(blf_N0.SpMat());
        mfem::SparseMatrix N0_dt;
        N0_dt = N0_n;
        N0_dt *= 1/dt;
        N0_n *= -1.;
        N0_dt.Finalize();
        N0_n.Finalize();

        // Matrix C0
        mfem::MixedBilinearForm blf_C0(&ND0, &RT0);
        blf_C0.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator()); //=(curl u,v)
        blf_C0.Assemble();
        blf_C0.Finalize();
        mfem::SparseMatrix C0(blf_C0.SpMat());
        mfem::SparseMatrix *C0T;
        mfem::SparseMatrix C0_Re;
        mfem::SparseMatrix C0T_Re;
        C0T = Transpose(C0);
        C0_Re = C0;
        C0T_Re = *C0T; 
        C0_Re *= Re_inv/2.;
        C0T_Re *= Re_inv/2.;
        C0.Finalize();
        C0T->Finalize();
        C0_Re.Finalize();
        C0T_Re.Finalize();

        // Matrix D0
        mfem::MixedBilinearForm blf_D0(&RT0, &DG);
        blf_D0.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator()); //=(div u,v)
        blf_D0.Assemble();
        blf_D0.Finalize();
        mfem::SparseMatrix D0(blf_D0.SpMat());
        mfem::SparseMatrix *D0T_n;
        D0T_n = Transpose(D0);
        *D0T_n *= -1.;
        D0.Finalize();
        D0T_n->Finalize();

        // Matrix G0
        mfem::MixedBilinearForm blf_G0(&CG0, &ND0);
        blf_G0.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
        blf_G0.Assemble();
        blf_G0.Finalize();
        mfem::SparseMatrix G0(blf_G0.SpMat());
        mfem::SparseMatrix *G0T;
        G0T = Transpose(G0);
        G0.Finalize();
        G0T->Finalize();

        // prepare some matrices for dual pressure constraint
        mfem::BilinearForm blf_Lam(&DG);
        blf_Lam.AddDomainIntegrator(new mfem::MassIntegrator());
        blf_Lam.Assemble();
        blf_Lam.Finalize();
        mfem::SparseMatrix mat_Lam (blf_Lam.SpMat());
        mat_Lam.Finalize();
        mfem::Vector vec_Lam (q.Size());
        mfem::GridFunction vec_one(&DG);
        vec_one=1.;
        mat_Lam.Mult(vec_one,vec_Lam);

        // Lambda matrix for dual pressure constraint
        mfem::DenseMatrix Lambda(q.Size(), 1);
        mfem::DenseMatrix LambdaT(1, q.Size());
        for (int i=0; i<q.Size(); i++) {
            Lambda.Elem(i,0) = vec_Lam[i];
            LambdaT.Elem(0,i) = vec_Lam[i];
        }        

        // initialize system matrices
        mfem::Array<int> offsets_1 (4);
        offsets_1[0] = 0;
        offsets_1[1] = u.Size();
        offsets_1[2] = z.Size();
        offsets_1[3] = p.Size();
        offsets_1.PartialSum(); // exclusive scan
        mfem::Array<int> offsets_2 (5);
        offsets_2[0] = 0;
        offsets_2[1] = v.Size();
        offsets_2[2] = w.Size();
        offsets_2[3] = q.Size();
        offsets_2[4] = 1;
        offsets_2.PartialSum();
        mfem::BlockOperator A1(offsets_1);
        mfem::BlockOperator A2(offsets_2);

        // initialize rhs
        mfem::Vector b1(size_1);
        mfem::Vector b1sub(u.Size());
        mfem::Vector b2(size_2); 
        mfem::Vector b2sub(v.Size());

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR0_eul(&ND0,&ND0); 
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        mfem::ConstantCoefficient two_over_dt(2.0/dt);
        blf_MR0_eul.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_MR0_eul.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_MR0_eul.Assemble();
        blf_MR0_eul.Finalize();
        mfem::SparseMatrix MR0_eul(blf_MR0_eul.SpMat());
        MR0_eul.Finalize();
        
        // CT for eulerstep
        mfem::SparseMatrix C0T_eul = C0T_Re;
        C0T_eul *= 2;
        C0T_eul.Finalize();

        // assemble and solve system
        A1.SetBlock(0,0, &MR0_eul);
        A1.SetBlock(0,1, &C0T_eul);
        A1.SetBlock(0,2, &G0);
        A1.SetBlock(1,0, &C0);
        A1.SetBlock(1,1, &N0_n);
        A1.SetBlock(2,0, G0T);

        // update b1, b2 for eulerstep
        b1 = 0.0;
        b1sub = 0.0;
        M0_dt.AddMult(u,b1sub,2);
        b1.AddSubVector(f1,0);
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
        double tol = 1e-10;
        int iter = 100000;  
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

            // update R20 
            mfem::MixedBilinearForm blf_R20(&RT0,&RT0);
            mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
            blf_R20.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_R20.Assemble();
            blf_R20.Finalize();
            mfem::SparseMatrix R20(blf_R20.SpMat());
            R20 *= 1./2.;
            R20.Finalize();

            // update NR0
            mfem::MixedBilinearForm blf_NR0(&RT0,&RT0); 
            blf_NR0.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_NR0.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_NR0.Assemble();
            blf_NR0.Finalize();
            mfem::SparseMatrix NR0(blf_NR0.SpMat());
            NR0 *= 1./2.;
            NR0.Finalize();

            // update A2
            A2.SetBlock(0,0, &NR0);
            A2.SetBlock(0,1, &C0_Re);
            A2.SetBlock(0,2, D0T_n);
            A2.SetBlock(1,0, C0T);
            A2.SetBlock(1,1, &M0_n);
            A2.SetBlock(2,0, &D0);
            A2.SetBlock(2,3, &Lambda);
            A2.SetBlock(3,2, &LambdaT);
            
            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N0_dt.AddMult(v,b2sub);
            R20.AddMult(v,b2sub,-1);
            C0_Re.AddMult(w,b2sub,-1);
            b2.AddSubVector(f2,0);
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
            y.GetSubVector(lam_dofs, lam);
            // std::cout <<"lam = "<< lam[0] << "\n";

            // integral of q should be 0
            double integral_q = 0.;
            for (int i=0; i<q.Size(); i++) {
                integral_q += (q[i] * 1/q.Size());
            }  
            std::cout << "int(q) = " << integral_q<<"\n";

            


            ////////////////////////////////////////////////////////////////////
            // PRIMAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R10
            mfem::MixedBilinearForm blf_R10(&ND0,&ND0);
            mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
            blf_R10.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_R10.Assemble();
            blf_R10.Finalize();
            mfem::SparseMatrix R10(blf_R10.SpMat());
            R10 *= 1./2.;
            R10.Finalize();

            // update MR0
            mfem::MixedBilinearForm blf_MR0(&ND0,&ND0); 
            blf_MR0.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_MR0.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_MR0.Assemble();
            blf_MR0.Finalize();
            mfem::SparseMatrix MR0(blf_MR0.SpMat());
            MR0 *= 1./2.;
            MR0.Finalize();

            // update A1
            A1.SetBlock(0,0, &MR0);
            A1.SetBlock(0,1, &C0T_Re);
            A1.SetBlock(0,2, &G0);
            A1.SetBlock(1,0, &C0);
            A1.SetBlock(1,1, &N0_n);
            A1.SetBlock(2,0, G0T);

            // update b1
            b1 = 0.0;
            b1sub = 0.0;
            M0_dt.AddMult(u,b1sub);
            R10.AddMult(u,b1sub,-1);
            C0T_Re.AddMult(z,b1sub,-1);
            b1.AddSubVector(b1sub,0);
            b1.AddSubVector(f1,0);

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
            
        } // time loop

        // convergence error
        double err_L2_u = u.ComputeL2Error(u_exact_coeff);
        double err_L2_v = v.ComputeL2Error(u_exact_coeff);
        mfem::GridFunction v_ND (&ND0);
        v_ND.ProjectGridFunction(v);
        double err_L2_diff = 0;
        for (int i=0; i<u.Size(); i++) {
            err_L2_diff += ((u(i)-v_ND(i))*(u(i)-v_ND(i)));
        }
        // std::cout << "L2err of v = "<< err_L2_v<<"\n";
        // std::cout << "L2err of u = "<< err_L2_u<<"\n";
        std::cout << "L2err(u-v) = "<< std::pow(err_L2_diff, 0.5) <<"\n";

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
        mfem::socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh << u << "window_title 'u in hcurl'" << std::endl;
        mfem::socketstream v_sock(vishost, visport);
        v_sock.precision(8);
        v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;
    
    // free memory
    delete fec_DG;
    delete fec_CG0;
    delete fec_ND0;
    delete fec_RT0;

    } // refinement loop
}


// cos squared init cond that satifies
// dirichlet BC, divu=0 and is C1-continuous (=> w is C0 continuous)
void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double C = 10;
    double DX = 0.5;
    double DY = 0.5;
    double DZ = 0.5;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish

    double cos = std::cos(C*(std::pow((x(0)-DX),2) 
                            +std::pow((x(1)-DY),2) 
                            +std::pow((x(2)-DZ),2)));
    double cos2 = cos*cos;

    if (std::pow((x(0)-DX),2) +
        std::pow((x(1)-DY),2) +
        std::pow((x(2)-DZ),2) < std::pow(R,2)) {

        returnvalue(0) = (x(1)-DY) * cos2;
        returnvalue(1) = -1* (x(0)-DX) * cos2;
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
    double DX = 0.5;
    double DY = 0.5;
    double DZ = 0.5;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    
    double cos = std::cos(C*(std::pow((x(0)-DX),2) +
                             std::pow((x(1)-DY),2) +
                             std::pow((x(2)-DZ),2)) );
    double sin = std::sin(C*(std::pow((x(0)-DX),2) +
                             std::pow((x(1)-DY),2) +
                             std::pow((x(2)-DZ),2)) );
    double cos2 = cos*cos;
    double sin2 = sin*sin;

    if (std::pow((x(0)-DX),2) +
        std::pow((x(1)-DY),2) +
        std::pow((x(2)-DZ),2) < std::pow(R,2)) {

        returnvalue(0) = - 4*C*(x(0)-DX)*(x(2)-DZ)*sin*cos;
        returnvalue(1) = - 4*C*(x(1)-DY)*(x(2)-DZ)*sin*cos;
        returnvalue(2) = - 2*cos2 
                         + 4*C*std::pow((x(0)-DX),2)*sin*cos
                         + 4*C*std::pow((x(1)-DY),2)*sin*cos;
    }
    else {
        returnvalue(0) = 0;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
}

void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    
    //TODO:  forcing
    returnvalue(0) = 0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}

// void u_exact_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
//     double pi = 3.14159265358979323846;
//     double Re = 500.; // chose Re here!
//     double nu = 1*1/Re; // = u*L/Re
//     double t = 0.15;
//     double F = std::exp(-2*nu*t);

//     returnvalue(0) =     std::cos(x(0)*pi)*std::sin(x(1)*pi) * F;
//     returnvalue(1) = -1* std::sin(x(0)*pi)*std::cos(x(1)*pi) * F;
//     returnvalue(2) = 0;
// }