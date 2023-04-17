//                                MFEM Example 3

//               equation curl curl E + E = f with boundary condition
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f.


// size: 12
// || E_h - E ||_{L^2} = 0.0744286
// size: 54
// || E_h - E ||_{L^2} = 0.0198627
// size: 300
// || E_h - E ||_{L^2} = 0.00501965
// size: 1944
// || E_h - E ||_{L^2} = 0.00125785




#include "mfem.hpp"
#include <fstream>
#include <iostream>

// using namespace std;
// using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const mfem::Vector &, mfem::Vector &);
void f_exact(const mfem::Vector &, mfem::Vector &);

int dim;

int main(int argc, char *argv[]) {

    std::cout << "---------------\n";

    // simulation parameters
    const char *mesh_file =  "extern/mfem-4.5/data/ref-cube.mesh";
    int order = 1;
    int ref_steps = 4;
    int init_ref = 0;

    for (int ref_step=0; ref_step<ref_steps; ref_step++) {

        // 3. Read the mesh from the given mesh file
        mfem::Mesh mesh(mesh_file, 1, 1); 
        dim = mesh.Dimension();
        int sdim = mesh.SpaceDimension();
        for (int l = 0; l<init_ref+ref_step; l++) {
            mesh.UniformRefinement();
        } 

        // FE space
        mfem::FiniteElementCollection *fec = new mfem::ND_FECollection(order, dim);
        mfem::FiniteElementSpace *fespace = new mfem::FiniteElementSpace(&mesh, fec);

        // boundary arrays: contain indices of essential boundary DOFs
        mfem::Array<int> ess_tdof_list;
        fespace->GetBoundaryTrueDofs(ess_tdof_list);

        // linform
        mfem::VectorFunctionCoefficient f(sdim, f_exact);
        mfem::LinearForm *b = new mfem::LinearForm(fespace);
        b->AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f));
        b->Assemble();

        // grid func x (Unknown); exact sol E as func coeff
        mfem::GridFunction x(fespace);
        mfem::VectorFunctionCoefficient E(sdim, E_exact);
        x.ProjectCoefficient(E);
        std::cout <<"size: "<< x.Size() << "\n";

        // 9. bilinear form corresponding to curl muinv curl + sigma I
        mfem::BilinearForm *a = new mfem::BilinearForm(fespace);
        a->AddDomainIntegrator(new mfem::CurlCurlIntegrator());
        a->AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
        a->Assemble();

        // formlinsys (eliminating BC, applying conforming constraint)
        mfem::OperatorPtr A;
        mfem::Vector B, X;
        a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

        // 11. Solve the linear system A X = B.
        // mfem::OperatorJacobiSmoother M(*a, ess_tdof_list);
        // PCG(*A, M, B, X, 0, 1000, 1e-15, 0.0);
        
        // 11. Solve with different preconditioner
        // mfem::GSSmoother M((mfem::SparseMatrix&)(*A));
        // mfem::PCG(*A, M, B, X, 0, 1000, 1e-15, 0.0);

        // solve with minres
        double tol = 1e-15;
        int iter = 10000000;  
        mfem::MINRES(*A, B, X, 0, iter, tol*tol, tol*tol);

        // 12. Recover the solution as a finite element grid function.
        a->RecoverFEMSolution(X, *b, x);

        // 13. Compute and print the L^2 norm of the error.
        std::cout << "|| E_h - E ||_{L^2} = " << x.ComputeL2Error(E) << '\n';

        // 15. Send the solution by socket to a GLVis server.
        char vishost[] = "localhost";
        int  visport   = 19916;
        mfem::socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << mesh << x << std::flush;
    
        // 16. Free the used memory.
        delete a;
        delete b;
        delete fespace;
        delete fec;

    } // refinement loop
    return 0;
}


void E_exact(const mfem::Vector &x, mfem::Vector &E) {
    E(0) = sin( x(1));
    E(1) = sin( x(2));
    E(2) = sin( x(0));

    // E(0) = x(1);
    // E(1) = -x(0);
    // E(2) = 0.;
}


void f_exact(const mfem::Vector &x, mfem::Vector &f) {
    double kappa = 1.;
    f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
    f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
    f(2) = (1. + kappa * kappa) * sin(kappa * x(0));

    // f(0) = x(1);
    // f(1) = -x(0);
    // f(2) = 0.;
}

// void E_exact(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
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


// void f_exact(const mfem::Vector &x, mfem::Vector &returnvalue) { 


//     double pi = 3.14159265358979323846;
//     double C = 10;
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