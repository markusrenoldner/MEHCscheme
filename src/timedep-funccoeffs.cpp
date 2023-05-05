#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"


// goal: project a time dependent function onto a gridfunction


// declare this time dependent function
// important: the 2nd argument is the time variable!
void u_t(const mfem::Vector &x, double t, mfem::Vector &v);


int main(int argc, char *argv[]) {

    // space and gridfunction
    mfem::Mesh mesh("extern/mfem-4.5/data/ref-cube.mesh", 1, 1); 
    int dim = mesh.Dimension(); 
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(1,dim);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);

    // gridfunction, initialized to 0
    mfem::GridFunction u(&ND);
    u = 0.;

    // define FunctionCoefficient; contains analytical expression of u_t
    mfem::VectorFunctionCoefficient u_t_coeff(dim, u_t);

    // time loop
    for (double t=0 ; t<1 ; t+=0.1) {

        // update the protected time variable of the FunctionCoefficient
        u_t_coeff.SetTime(t);

        // project the updated FunctionCoefficient
        u.ProjectCoefficient(u_t_coeff);

        // print value from gridfunction to check if its updated
        std::cout << t << "\t" << u[0] << "\t" << u[1] << "\t" << u[2] << "\n";
    }

    // clean up
    delete fec_ND;
}


// define the time dependent function
void u_t(const mfem::Vector &x, double t, mfem::Vector &returnvalue) {
   
    // here one can use "t" to define time dependent functions
    returnvalue(0) = t;
    returnvalue(1) = 2*t;
    returnvalue(2) = std::sin(t);
}