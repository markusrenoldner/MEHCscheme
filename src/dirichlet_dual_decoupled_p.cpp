#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"




// decouple the systems by replacing the coupling vorticities by its exact
// valus (static manufactured solution)

// this file contains the primal system



struct Parameters {
    double Re_inv = 1; // = 1/Re 
    double dt     = 0.01;
    double tmax   = 3*dt;
    int ref_steps = 4;
    int init_ref  = 0;
    int order     = 1;
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
        mfem::GridFunction u_exact(&ND);
        mfem::GridFunction z_exact(&RT);
        mfem::GridFunction w_exact(&ND);

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_0); 
        u.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        u_exact.ProjectCoefficient(u_exact_coeff);
        w_exact.ProjectCoefficient(w_0_coeff);

        // linearform for forcing term
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::LinearForm f1(&ND);
        f1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f1.Assemble();

        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        std::cout << "size:"<<u.Size() << "\n";

        // initialize solution vectors
        mfem::Vector x(size_1);
        x.SetVector(u,0);
        x.SetVector(z,u.Size());
        x.SetVector(p,u.Size()+z.Size());

        // helper dofs
        mfem::Array<int> u_dofs (u.Size());
        mfem::Array<int> z_dofs (z.Size());
        mfem::Array<int> p_dofs (p.Size());
        std::iota(&u_dofs[0], &u_dofs[u.Size()], 0);
        std::iota(&z_dofs[0], &z_dofs[z.Size()], u.Size());
        std::iota(&p_dofs[0], &p_dofs[p.Size()], u.Size()+z.Size());

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
        mfem::BlockOperator A1(offsets_1);

        // initialize rhs
        mfem::Vector b1(size_1);
        mfem::Vector b1sub(u.Size());

        ////////////////////////////////////////////////////////////////////////////
        // forcing function constructed by initial value of gridfunctions
        ////////////////////////////////////////////////////////////////////////////
        
        // R1
        // mfem::MixedBilinearForm blf_R1(&ND,&ND);
        // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w_exact); // decoupling
        // // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
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
        
        // mfem::GridFunction bf1_gf(&ND);
        // bf1_gf=0.;
        // double tol = 1e-10;
        // int iter = 1000000;  
        // mfem::MINRES(M_n, bf1, bf1_gf, 0, iter, tol*tol, tol*tol);

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
        // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w); 
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w_exact); // decoupling
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
        int iter = 1000000;  
        mfem::MINRES(*A1_BC, B1, X, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        ATA1.RecoverFEMSolution(X, b1, x);
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);

        // time loop
        for (double t = dt ; t < tmax+dt ; t+=dt) {
            
            // std::cout << "--- t = "<<t<<"\n";
            // std::cout << t << ",";
                       
            ////////////////////////////////////////////////////////////////////
            // PRIMAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R1
            mfem::MixedBilinearForm blf_R1(&ND,&ND);
            // mfem::VectorGridFunctionCoefficient w_gfcoeff(&w); 
            mfem::VectorGridFunctionCoefficient w_gfcoeff(&w_exact); // decoupling
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
            // b1.AddSubVector(bf1,0);

            //Transposition
            mfem::TransposeOperator AT1 (&A1);
            mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
            mfem::Vector ATb1 (size_1);
            A1.MultTranspose(b1,ATb1);

            // form linear system with BC
            ATA1.FormLinearSystem(ess_dof1, x, ATb1, A1_BC, X, B1);

            // solve 
            mfem::MINRES(*A1_BC, B1, X, 0, iter, tol*tol, tol*tol); 
            ATA1.RecoverFEMSolution(X, b1, x);
            x.GetSubVector(u_dofs, u);
            x.GetSubVector(z_dofs, z);
            x.GetSubVector(p_dofs, p);


            ////////////////////////////////////////////////////////////////////
            // EQUATION
            ////////////////////////////////////////////////////////////////////

            mfem::Vector vec_47a (u.Size());vec_47a=0.;
            MR.AddMult(u,vec_47a);
            CT_Re.AddMult(z,vec_47a);
            G.AddMult(p,vec_47a);
            std::cout << "A--\n"<<vec_47a.Norml2() << "\n" << b1.Norml2() << "\n";

            mfem::Vector vec_47b (u.Size());vec_47b=0.;
            C.AddMult(u,vec_47b);
            N_n.AddMult(z,vec_47b);
            std::cout << "B--\n"<<vec_47b.Norml2()  << "\n";

            mfem::Vector vec_47c (u.Size());vec_47c=0.;
            GT->AddMult(u,vec_47c);
            std::cout << "C--\n"<<vec_47c.Norml2()  << "\n";

            
            ////////////////////////////////////////////////////////////////////
            // CONSERVATION
            ////////////////////////////////////////////////////////////////////

            double K1 = 1./2.*blf_M.InnerProduct(u,u);
            // std::cout <<std::abs(K1) << ",\n";

        } // time loop

        // convergence error
        double err_L2_u = u.ComputeL2Error(u_exact_coeff);
        std::cout << "L2err of u = "<< err_L2_u<<"\n";

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

        // mfem::socketstream ue_sock(vishost, visport);
        // ue_sock.precision(8);
        // ue_sock << "solution\n" << mesh << u_exact << "window_title 'u_0'" << std::endl;
        
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

    Parameters param;
    double Re_inv = param.Re_inv; // = 1/Re 

    double pi = 3.14159265358979323846;
    double C = 10.;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
    double cos3 = cos*cos*cos;
    double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
    double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
    double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
    
    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) + Re_inv * 2*C*Y* (4*C * (X*X+Y*Y+Z*Z) * cosof2 + 5*sinof2);
        returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) - Re_inv * 2*C*X* (4*C * (X*X+Y*Y+Z*Z) * cosof2 + 5*sinof2);
        returnvalue(2) = 4*C*(X*X+Y*Y) * Z*cos3*sin;
    }
    else {
        returnvalue(0) = 0.; 
        returnvalue(1) = 0.; 
        returnvalue(2) = 0.;
    }   
}





/////////////////////////////////////
// i think this is the simpler f resulting from including ther bernouulli term

// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1.;
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







///////////////////////////////////
// ?
        
// void f_alt(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1;
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
//     // double eC2 = 8*C*C;
    
//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin)        
//             + 1*Re_inv*(2*Y*C*(4*C*(X*X+Y*Y+Z*Z)*cosof2 + 5*sinof2) );

//         returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin)
//             - 1*Re_inv*(2*X*C*(4*C*(X*X+Y*Y+Z*Z)*cosof2 + 5*sinof2) );

//         returnvalue(2) = 4*C*(X*X+Y*Y)*Z*cos3*sin;
//     }
//     else {
//         returnvalue(0) = 0.; 
//         returnvalue(1) = 0.; 
//         returnvalue(2) = 0.;
//     }   
// }