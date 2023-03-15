#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

#include <chrono>



// curl-curl problem to investigate mfem projection behaviour



struct Parameters {
    int ref_steps = 6;
    int init_ref  = 0;
    int order = 1;
    const char* mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
};


void     u_exact(const mfem::Vector &x, mfem::Vector &v);
void     f_exact(const mfem::Vector &x, mfem::Vector &v); 

int main(int argc, char *argv[]) {

    std::cout << "---------------\n";
    
    // simulation parameters
    Parameters param;
    int ref_steps = param.ref_steps;
    int init_ref  = param.init_ref;
    const char *mesh_file = param.mesh_file;
    int order = param.order;

    for (int ref_step=0; ref_step<ref_steps; ref_step++) {

        // mesh
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        for (int l = 0; l<init_ref+ref_step; l++) {
            mesh.UniformRefinement();
        }

        // FE space
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);

        // boundary arrays: contain indices of essential boundary DOFs
        mfem::Array<int> ND_ess_tdof;
        ND.GetBoundaryTrueDofs(ND_ess_tdof);

        // Linform
        mfem::VectorFunctionCoefficient f_coeff(dim, f_exact);
        mfem::LinearForm b(&ND);
        b.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        b.Assemble();

        // gridfunction + exact sol as func coeff
        mfem::GridFunction u(&ND);
        u = 98765.;
        mfem::VectorFunctionCoefficient u_ex_coeff(dim, u_exact);
        std::cout <<"size: "<< u.Size() << "\n";

        // set boundary values at u (can also just project u_ex_coeff onto u)
        mfem::GridFunction u_boundary(&ND);
        u_boundary.ProjectCoefficient(u_ex_coeff);
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            u[ND_ess_tdof[i]] = u_boundary[ND_ess_tdof[i]];
        }
    
        // blf
        mfem::BilinearForm blf_M(&ND);
        blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_M.AddDomainIntegrator(new mfem::MixedCurlCurlIntegrator()); //=(curl u,curl v)
        blf_M.Assemble();
        // blf_M.Finalize();

        // form linear system
        mfem::OperatorPtr A;
        mfem::Vector B, X;
        blf_M.FormLinearSystem(ND_ess_tdof, u, b, A, X, B);  

        // solve 
        double tol = 1e-15;
        int iter = 10000000;  
        mfem::MINRES(*A, B, X, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        blf_M.RecoverFEMSolution(X, b, u);

        // convergence error
        double err_L2_u = u.ComputeL2Error(u_ex_coeff);
        std::cout << "L2err of u = "<< err_L2_u<<"\n";

        // visuals
        char vishost[] = "localhost";
        int  visport   = 19916;
        mfem::socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh << u << "window_title 'u in hcurl'" << std::endl;

        // delete FE space
        delete fec_ND;
    }
}

void u_exact(const mfem::Vector &x, mfem::Vector &E) {
    // E(0) = sin( x(1));
    // E(1) = sin( x(2));
    // E(2) = sin( x(0));

    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;
   
    double cos = std::cos(C*(X*X+Y*Y+Z*Z));
    double cos2 = cos*cos;
    double cosx = std::cos(C*(X*X));
    double cos2x = cosx*cosx;
    
    E(0) = std::sin(Y);
    E(1) = std::sin(Z);
    E(2) = 0.;
}


void f_exact(const mfem::Vector &x, mfem::Vector &f) {
    double kappa = 1.;
    // f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
    // f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
    // f(2) = (1. + kappa * kappa) * sin(kappa * x(0));

    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    double cos = std::cos(C*(X*X+Y*Y+Z*Z));
    double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z));
    double cos2 = cos*cos;
    double sin = std::sin(C*(X*X+Y*Y+Z*Z));
    double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z));
    double cosx = std::cos(C*(X*X));
    double cos2x = cosx*cosx;

    f(0) = 2*std::sin(Y);
    f(1) = 2*std::sin(Z);
    f(2) = 0.;
}

// void u_exact(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double pi = 3.14159265358979323846;
//     double C = 1;
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


// void f_exact(const mfem::Vector &x, mfem::Vector &returnvalue) { 


//     double pi = 3.14159265358979323846;
//     double C = 1;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
   
//     double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z));
//     double cos2 = cos*cos;
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z));
//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z));

//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) =  Y*cos2 + 2*C*Y*(5*sinof2 + 4*C*(X*X+Y*Y+Z*Z)*cosof2);
//         returnvalue(1) = -X*cos2 - 2*C*X*(5*sinof2 + 4*C*(X*X+Y*Y+Z*Z)*cosof2);
//         returnvalue(2) = 0;
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }

// }