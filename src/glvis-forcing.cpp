#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

#include <chrono>



// visualizing some gridfunctions




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
    int ref_steps = 0;
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
        mesh.UniformRefinement();
        mesh.UniformRefinement();
        mesh.UniformRefinement();
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
        z.ProjectCoefficient(w_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        f2.ProjectCoefficient(f_coeff);
        f_t1.ProjectCoefficient(ft1_coeff);
        f_t2.ProjectCoefficient(ft2_coeff);
        v_exact.ProjectCoefficient(u_0_coeff);
        z_exact.ProjectCoefficient(w_0_coeff);

        // mfem::GridFunction fpseudo(&RT);
        // fpseudo = f_t1 + f_t2;

        // visual tests:
        char vishost[] = "localhost";
        int  visport   = 19916;
        // mfem::socketstream v0_sock(vishost, visport);
        // v0_sock.precision(8);
        // v0_sock << "solution\n" << mesh << v_exact << "window_title 'v0'" << std::endl;

        // mfem::socketstream v_sock(vishost, visport);
        // v_sock.precision(8);
        // v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;

        mfem::socketstream f2_sock(vishost, visport);
        f2_sock.precision(8);
        f2_sock << "solution\n" << mesh << f2 << "window_title 'f'" << std::endl;
        
        // mfem::socketstream ft1_sock(vishost, visport);
        // ft1_sock.precision(8);
        // ft1_sock << "solution\n" << mesh << f_t1 << "window_title 'fterm1'" << std::endl;
        
        // mfem::socketstream ft2_sock(vishost, visport);
        // ft2_sock.precision(8);
        // ft2_sock << "solution\n" << mesh << f_t2 << "window_title 'fterm2'" << std::endl;

        //// new f
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


        mfem::Vector bf2 (v.Size()); 
        bf2=0.;
        R2.AddMult(v,bf2,2);
        C_Re.AddMult(w,bf2,2);
        
        
        mfem::GridFunction bf2_gf(&RT);
        bf2_gf=0.;
        double tol = 1e-10;
        int iter = 1000000;  
        mfem::MINRES(N_n, bf2, bf2_gf, 0, iter, tol*tol, tol*tol);













        ///////

        mfem::socketstream fnew_sock(vishost, visport);
        fnew_sock.precision(8);
        fnew_sock << "solution\n" << mesh << bf2_gf << "window_title 'fnew'" << std::endl;
        
        






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