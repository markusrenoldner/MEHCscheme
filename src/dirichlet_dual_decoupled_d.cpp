#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"




// decouple the systems by replacing the coupling vorticities by its exact
// valus (static manufactured solution)

// this file contains the dual system with dirichlet BC



struct Parameters {
    double Re_inv = 1; // = 1/Re 
    double dt     = 0.01;
    double tmax   = 1*dt;
    int ref_steps = 4;
    int init_ref  = 0;
    int order  = 0;
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
    int order = para.order;

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        auto start = std::chrono::high_resolution_clock::now();

        // mesh
        const char *mesh_file = param.mesh_file;
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        int l;
        // dt *= 0.5; // TODO
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

        // TODO
        mfem::Array<int> ND_ess_tdof2;
        ND.GetBoundaryTrueDofs(ND_ess_tdof2); 

        // unkowns and gridfunctions
        mfem::GridFunction v(&RT); //v = 3.;
        mfem::GridFunction w(&ND); //w = 3.; 
        mfem::GridFunction q(&DG); q=0.; //q = 9.3;
        mfem::GridFunction v_exact(&RT);
        mfem::GridFunction z_exact(&RT);
        mfem::GridFunction w_exact(&ND);

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_0); 
        v.ProjectCoefficient(u_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        v_exact.ProjectCoefficient(u_0_coeff);
        z_exact.ProjectCoefficient(w_0_coeff);

        // linearform for forcing term
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::LinearForm f2(&RT);
        f2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f2.Assemble();

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

        // C_Wouter 
        // TODO
        mfem::MixedBilinearForm blf_C_Wouter(&RT,&ND);
        blf_C_Wouter.AddDomainIntegrator(new mfem::MixedVectorWeakCurlIntegrator());
        blf_C_Wouter.Assemble();
        blf_C_Wouter.Finalize();
        mfem::Vector blf_C_Wouter_v_0(ND.GetNDofs());
        blf_C_Wouter.Mult(v_exact,blf_C_Wouter_v_0);
        std::cout << blf_C_Wouter_v_0.Normlinf() << std::endl;

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


            ////////////////////////////////////////////////////////////////////
            // DUAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R2 
            mfem::MixedBilinearForm blf_R2(&RT,&RT);
            // mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
            mfem::VectorGridFunctionCoefficient z_gfcoeff(&z_exact);
            mfem::ConstantCoefficient two_over_dt(2.0/dt);
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
            // A2.SetBlock(1,0, CT); //TODO
            // A2.SetBlock(1,1, &M_n);
            A2.SetBlock(2,0, &D);
            
            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N_dt.AddMult(v,b2sub);
            R2.AddMult(v,b2sub,-1);
            C_Re.AddMult(w,b2sub,-1);
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
            ATA2.RecoverFEMSolution(Y, b2, y);
            y.GetSubVector(v_dofs, v);
            y.GetSubVector(w_dofs, w);
            y.GetSubVector(q_dofs, q);    














            ///////////////////////////////////////////////////////////////////
            //TODO
            mfem::Vector testminres(size_2);
            A2.Mult(y,testminres);
            std::cout <<"minres-residual:"<< testminres.Normlinf() << "\n";

            // check first term of 13e
            mfem::Vector test13e1(w.Size());
            CT->Mult(v,test13e1);
            std::cout <<"eq13e1: "<< test13e1.Norml2() << "\n";

            // check first term of 13e
            mfem::Vector test13e1b(w.Size());
            CT->Mult(v_exact,test13e1b);
            std::cout <<"eq13e1b: "<< test13e1b.Norml2() << "\n";

            // M_n
            mfem::Vector righthandside(w.Size());
            CT->Mult(v,righthandside);
            mfem::Vector W_vec;
            mfem::Vector RHS;
            mfem::Operator* Oper;
            std::cout<< "righthandsideA:" << righthandside.Norml2() << "\n";

            // blf_M.FormSystemMatrix(ND_ess_tdof2, Oper);
            mfem::SparseMatrix mat_Wouter (3,3);
            mat_Wouter.Set(0,0,1.);
            mat_Wouter.Set(0,2,1.);
            mat_Wouter.Set(1,0,1.);
            mat_Wouter.Set(1,1,1.);
            mat_Wouter.Set(2,2,1.);
            mfem::Vector rhs_Wouter (5); rhs_Wouter = 0.;
            mfem::Vector x_Wouter (5);
            mfem::Array <int> esstdof_Wouter;
            //.Append(0);
            esstdof_Wouter.Append(3);
            mfem::Vector X_wouter;
            mfem::Vector RHS_wouter;
            x_Wouter.Elem(0)=2.;
            x_Wouter.Elem(1)=3.;
            x_Wouter.Elem(2)=4.;
            x_Wouter.Elem(3)=5.;
            x_Wouter.Elem(4)=6.;
            
            mfem::SparseMatrix mat_Wouter2 (2,2);
            mat_Wouter2.Set(1,0,1.);
            mat_Wouter2.Set(0,1,1.);

            mfem::SparseMatrix mat_Wouter3 (2,3);
            mat_Wouter3.Set(1,0,1.);

            mfem::Array<int> offsets_wouter (3);
            offsets_wouter[0] = 0;
            offsets_wouter[1] = 3;
            offsets_wouter[2] = 2;
            offsets_wouter.PartialSum();
            mfem::BlockMatrix blm_Wouter(offsets_wouter);

            blm_Wouter.SetBlock(0,0,&mat_Wouter);
            blm_Wouter.SetBlock(1,1,&mat_Wouter2);
            blm_Wouter.SetBlock(1,0,&mat_Wouter3);

            mfem::DenseMatrix densmatwouter(5,5);
            blm_Wouter.FormLinearSystem(esstdof_Wouter, x_Wouter, rhs_Wouter, Oper, X_wouter, RHS_wouter);
            for(int i=0; i<5;i++){
                mfem::Vector temp_Wouter_in(5), temp_Wouter_out(5);
                temp_Wouter_in =0.;
                temp_Wouter_out = 0.;
                temp_Wouter_in.Elem(i) = 1.;
                Oper->Mult(temp_Wouter_in,temp_Wouter_out);
                // temp_Wouter_out.Print(std::cout);
                densmatwouter.SetCol(i,temp_Wouter_out);
                // std::cout << std::endl;

            }
            densmatwouter.PrintMatlab(std::cout);
            std::cout << "RHS_Wouter: \n";
            RHS_wouter.Print(std::cout);
            rhs_Wouter.Print(std::cout);

            // blf_M.FormLinearSystem(ND_ess_tdof2, w, righthandside, Oper, W_vec, RHS);

            mfem::MINRES(*Oper,RHS,W_vec, 2, iter, tol*tol, tol*tol);

            // M_n.RecoverFEMSolution(W_vec, righthandside, w);



            mfem::Vector testminres2(w.Size());
            M_n.Mult(w,testminres2);
            std::cout <<"minres-residual_w:"<< testminres2.Normlinf() << "\n";
            std::cout<< "righthandsideB:" << righthandside.Norml2() << "\n";
            // div-free cond
            mfem::Vector test(v.Size());
            D.Mult(v,test);
            // std::cout <<"div(v):"<< test.Norml2() << "\n";
            ///////////////////////////////////////////////////////////////////












           
            ////////////////////////////////////////////////////////////////////
            // CONSERVATION
            ////////////////////////////////////////////////////////////////////

            // double K2_old = -1./2.*blf_N.InnerProduct(v_old,v_old);
            double K2 = -1./2.*blf_N.InnerProduct(v,v);
            // std::cout <<std::abs(K2) << ",\n";


        } // time loop

        // convergence error
        double err_L2_v = v.ComputeL2Error(u_0_coeff);
        std::cout << "L2err of v = "<< err_L2_v<<"\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        std::cout << "runtime = " << duration.count() << "ms" << std::endl;

        // visuals
        // char vishost[] = "localhost";
        // int  visport   = 19916;
        // mfem::socketstream v_sock(vishost, visport);
        // v_sock.precision(8);
        // v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;

        // mfem::socketstream ve_sock(vishost, visport);
        // ve_sock.precision(8);
        // ve_sock << "solution\n" << mesh << v_exact << "window_title 'v_0'" << std::endl;


        // mfem::socketstream p_sock(vishost, visport);
        // p_sock.precision(8);
        // p_sock << "solution\n" << mesh << q << "window_title 'pressure'" << std::endl;

        // free memory
        delete fec_DG;
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;

    } // refinement loop
}




// another manuf solution, with non-zero boundary values of u
void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = 1.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}
void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    returnvalue(0) =0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}
void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    returnvalue(0) =0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;

}





// cos squared init cond that satifies
// dirichlet BC, divu=0 and is C1-continuous (=> w is C continuous)
// void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
   
//     double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     double cos2 = cos*cos;

//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = Y * cos2;
//         returnvalue(1) = -X * cos2;
//         returnvalue(2) = 0;
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
// }

// void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
   
//     double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z));
//     double cos2 = cos*cos;

//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = - 4*C*X*Z*sin*cos;
//         returnvalue(1) = - 4*C*Y*Z*sin*cos;
//         returnvalue(2) = - 2*cos2 + 4*C*X*X*sin*cos + 4*C*Y*Y*sin*cos;
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
// }

// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1.;
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
//     double cos3 = cos*cos*cos;
//     double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
//     double cos4 = cos*cos*cos*cos;
    
//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) + Re_inv * (2*C*Y* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
//         returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) - Re_inv * (2*C*X* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
//         returnvalue(2) = 4*C*(X*X+Y*Y)*Z * cos3 *sin;
//     }
//     else {
//         returnvalue(0) = 0.; 
//         returnvalue(1) = 0.; 
//         returnvalue(2) = 0.;
//     }   
// }

// void f_term1(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1.;
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
//     double cos3 = cos*cos*cos;
//     double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
//     double cos4 = cos*cos*cos*cos;
    
//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin);
//         returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin);
//         returnvalue(2) = 4*C*(X*X+Y*Y)*Z * cos3 *sin;
//     }
//     else {
//         returnvalue(0) = 0.; 
//         returnvalue(1) = 0.; 
//         returnvalue(2) = 0.;
//     }   
// }

// void f_term2(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1.;
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
//     double cos3 = cos*cos*cos;
//     double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
//     double cos4 = cos*cos*cos*cos;
    
//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = + Re_inv * (2*C*Y* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
//         returnvalue(1) = - Re_inv * (2*C*X* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
//         returnvalue(2) = 0.;
//     }
//     else {
//         returnvalue(0) = 0.; 
//         returnvalue(1) = 0.; 
//         returnvalue(2) = 0.;
//     }   
// }