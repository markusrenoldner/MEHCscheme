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
void       f_term1(const mfem::Vector &x, mfem::Vector &v); 
void       f_term2(const mfem::Vector &x, mfem::Vector &v); 
// void u_exact(const mfem::Vector &x, mfem::Vector &returnvalue);

int main(int argc, char *argv[]) {

    // simulation parameters
    // careful: Re also has to be defined in the manufactured sol
    double Re_inv = 1; // = 1/Re 
    double dt = 0.2;
    // dt = std::sqrt(3)*1;
    double tmax = 3*dt;
    int ref_steps = 3;
    // std::cout <<"----------\n"<<"Re:   "<<1/Re_inv <<"\ndt:   "<<dt<< "\ntmax: "<<tmax<<"\n----------\n";

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {

        // dt = std::sqrt(3)/std::pow(2,ref_step+1);
        
        auto start = std::chrono::high_resolution_clock::now();

        // mesh
        const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        for (int l = 0; l<ref_step; l++) {mesh.UniformRefinement();} 
        std::cout << "----------ref: " << ref_step << "----------\n";
        // mesh.UniformRefinement();
        // mesh.UniformRefinement();
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
        mfem::GridFunction v(&RT); //v = 3.;
        mfem::GridFunction w(&ND); //w = 3.; 
        mfem::GridFunction q(&DG); q=0.; //q = 9.3;
        mfem::GridFunction f2(&RT);
        mfem::GridFunction f_t1(&RT);
        mfem::GridFunction f_t2(&RT);
        mfem::GridFunction v_exact(&RT);
        mfem::GridFunction z_exact(&RT);
        mfem::GridFunction w_exact(&ND);

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::VectorFunctionCoefficient ft1_coeff(dim, f_term1);
        mfem::VectorFunctionCoefficient ft2_coeff(dim, f_term2);
        v.ProjectCoefficient(u_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        f2.ProjectCoefficient(f_coeff);
        f_t1.ProjectCoefficient(ft1_coeff);
        f_t2.ProjectCoefficient(ft2_coeff);
        v_exact.ProjectCoefficient(u_0_coeff);
        z_exact.ProjectCoefficient(w_0_coeff);

        mfem::GridFunction fpseudo(&RT);
        for (int i = 0 ; i<v.Size(); i++) {
            fpseudo[i] = f_t1[i] + f_t2[i] - f2[i];
        }


        // visual tests:
        char vishost[] = "localhost";
        int  visport   = 19916;
        // mfem::socketstream v0_sock(vishost, visport);
        // v0_sock.precision(8);
        // v0_sock << "solution\n" << mesh << v_exact << "window_title 'v0'" << std::endl;

        // mfem::socketstream v_sock(vishost, visport);
        // v_sock.precision(8);
        // v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;

        // mfem::socketstream f2_sock(vishost, visport);
        // f2_sock.precision(8);
        // f2_sock << "solution\n" << mesh << f2 << "window_title 'f'" << std::endl;
        
        // mfem::socketstream ft1_sock(vishost, visport);
        // ft1_sock.precision(8);
        // ft1_sock << "solution\n" << mesh << f_t1 << "window_title 'fterm1'" << std::endl;
        
        // mfem::socketstream ft2_sock(vishost, visport);
        // ft2_sock.precision(8);
        // ft2_sock << "solution\n" << mesh << f_t2 << "window_title 'fterm2'" << std::endl;
        
        // mfem::socketstream fpseudo_sock(vishost, visport);
        // fpseudo_sock.precision(8);
        // fpseudo_sock << "solution\n" << mesh << fpseudo << "window_title 'fpseudo'" << std::endl;

        








        // helper vectors for old values
        mfem::Vector v_old(v.Size()); v_old = 0.;
        mfem::Vector w_old(w.Size()); w_old = 0.;
    

        // system size
        int size_2 = v.Size() + w.Size() + q.Size();
        std::cout << "size:"<<v.Size()<<"\n";
        
        // initialize solution vectors
        mfem::Vector y(size_2);
        y.SetVector(v,0);
        y.SetVector(w,v.Size());
        y.SetVector(q,v.Size()+w.Size());

        // helper dofs
        mfem::Array<int> v_dofs (v.Size());
        mfem::Array<int> w_dofs (w.Size());
        mfem::Array<int> q_dofs (q.Size());
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
        mfem::Array<int> offsets_2 (4);
        offsets_2[0] = 0;
        offsets_2[1] = v.Size();
        offsets_2[2] = w.Size();
        offsets_2[3] = q.Size();
        // offsets_2[4] = 1;
        offsets_2.PartialSum();
        mfem::BlockOperator A2(offsets_2);

        // initialize rhs
        mfem::Vector b2(size_2); 
        mfem::Vector b2sub(v.Size());

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////



        // form linear system with BC
        mfem::Operator *A1_BC;
        mfem::Operator *A2_BC;
        mfem::Vector X;
        mfem::Vector B1;
        mfem::Vector Y;
        mfem::Vector B2;

        // solve 
        double tol = 1e-15;
        int iter = 1000000;  

        // time loop
        for (double t = dt ; t < tmax+dt ; t+=dt) {
            // std::cout << "--- t = "<<t<<"\n";
            // std::cout << t << ",";

            v_old = v;
            
            mfem::ConstantCoefficient two_over_dt(2.0/dt);

            ////////////////////////////////////////////////////////////////////
            // DUAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R2 
            mfem::MixedBilinearForm blf_R2(&RT,&RT);
            // mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
            mfem::VectorGridFunctionCoefficient z_gfcoeff(&z_exact);

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

            // forcing f
            // b2.AddSubVector(f2,0);
            mfem::Vector bf (v.Size()); 
            bf=0.;
            R2.AddMult(v,bf,2);
            C_Re.AddMult(w,bf,2);

            // bf_gf = -1* N_n ^-1 * bf            
            mfem::GridFunction bf_gf(&RT);
            bf_gf=0.;
            mfem::MINRES(N_n, bf, bf_gf, 0, iter, tol*tol, tol*tol);

            // visual artificial forcing term
            // mfem::socketstream bf_sock(vishost, visport);
            // bf_sock.precision(8);
            // bf_sock << "solution\n" << mesh << bf_gf << "window_title 'bf'" << std::endl;
        



            // add together
            b2.AddSubVector(bf,0);
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
            // CONSERVATION
            ////////////////////////////////////////////////////////////////////

         

            double K2_old = -1./2.*blf_N.InnerProduct(v_old,v_old);
            double K2 = -1./2.*blf_N.InnerProduct(v,v);
            // std::cout <<std::abs(K2) << ",\n";




        } // time loop

        // convergence error
        double err_L2_v = v.ComputeL2Error(u_0_coeff);
        std::cout << "L2err of v = "<< err_L2_v<<"\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        // std::cout << "runtime = " << duration.count() << "ms" << std::endl;

        // visuals
        // char vishost[] = "localhost";
        // int  visport   = 19916;
        // mfem::socketstream v_sock(vishost, visport);
        // v_sock.precision(8);
        // v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;

        // mfem::socketstream ve_sock(vishost, visport);
        // ve_sock.precision(8);
        // ve_sock << "solution\n" << mesh << v_exact << "window_title 'v_0'" << std::endl;

        
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

void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double Re_inv = 1.;
    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
    double cos3 = cos*cos*cos;
    double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
    double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
    double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
    double cos4 = cos*cos*cos*cos;
    
    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) + Re_inv * (2*C*Y* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
        returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) - Re_inv * (2*C*X* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
        returnvalue(2) = 4*C*(X*X+Y*Y)*Z * cos3 *sin;
    }
    else {
        returnvalue(0) = 0.; 
        returnvalue(1) = 0.; 
        returnvalue(2) = 0.;
    }   
}

void f_term1(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double Re_inv = 1.;
    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
    double cos3 = cos*cos*cos;
    double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
    double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
    double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
    double cos4 = cos*cos*cos*cos;
    
    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin);
        returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin);
        returnvalue(2) = 4*C*(X*X+Y*Y)*Z * cos3 *sin;
    }
    else {
        returnvalue(0) = 0.; 
        returnvalue(1) = 0.; 
        returnvalue(2) = 0.;
    }   
}

void f_term2(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double Re_inv = 1.;
    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
    double cos3 = cos*cos*cos;
    double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
    double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
    double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
    double cos4 = cos*cos*cos*cos;
    
    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = + Re_inv * (2*C*Y* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
        returnvalue(1) = - Re_inv * (2*C*X* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
        returnvalue(2) = 0.;
    }
    else {
        returnvalue(0) = 0.; 
        returnvalue(1) = 0.; 
        returnvalue(2) = 0.;
    }   
}