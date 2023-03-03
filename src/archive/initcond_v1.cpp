
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>


// project a vector-valued function onto a gridfunction
// see ex9






void velocity_function(const mfem::Vector &x, mfem::Vector &v);


int main(int argc, char *argv[])
{

    // mesh etc
    const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh";
    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    int order = 1;
    std::cout << dim << "\n" << order << "\n";

    // spaces
    mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
    mfem::FiniteElementSpace ND(&mesh, fec_ND);

    // project function onto u
    mfem::VectorFunctionCoefficient velocity(dim, velocity_function);
    mfem::GridFunction u(&ND); 
    u.ProjectCoefficient(velocity);

    // check u
    // for (int i=0; i<u.Size(); i++){
    //    std::cout << u[i] << "\n";
    // }

    // visuals
    std::ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh.Print(mesh_ofs);
    std::ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    u.Save(sol_ofs);
    char vishost[] = "localhost";
    int  visport   = 19916;
    mfem::socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << u << std::flush;

    delete fec_ND;
    return 0;
}


void velocity_function(const mfem::Vector &x, mfem::Vector &v) {
   
    // TODO: initial values u0=(cos(2piz),sin(2piz),sin(2pix))T 
    double pi = 3.14159265358979323846;
    int dim = x.Size();

    // projects everything to unit cube, see ex9
    // mfem::Vector x(dim);
    // for (int i = 0; i < dim; i++){
    //    double center = (bb_min[i] + bb_max[i]) * 0.5;
    //    X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
    // }

    v(0) = std::cos(2*pi*x(3));
    v(1) = std::sin(2*pi*x(3)); 
    v(2) = std::sin(2*pi*x(1));
}

