#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



int main(int argc, char *argv[]) {

    // mesh
    // const char *mesh_file = "extern/mfem-4.5/data/star.mesh";
    const char *mesh_file = "extern/mfem-4.5/data/ref-square.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    mesh.UniformRefinement();
    // for (int l = 0; l < 5; l++) {mesh.UniformRefinement();}
    
    // FE space
    int order = 1;
    FiniteElementCollection *fec;
    fec = new H1_FECollection(order, dim);
    FiniteElementSpace fespace(&mesh, fec);

    // boundary conditions
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh.bdr_attributes.Max()); 
    // mesh.bdr_attributes.Max() = 4
    ess_bdr = 1;
    // ess_bdr = {1,1,1,1}
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // fespace.GetBoundaryTrueDofs(ess_tdof_list);
    // trueDOFs exclude e.g double-counted DOFs that are part of 2 elem
    // ess_tdof_list = {0,1,2,3} or {0,1,2,3,4,5,6,7}

    //prints
    // std::cout << "bdr_attributes.Max(): "<<mesh.bdr_attributes.Max()<<"\n";
    // std::cout << "ess_bdr: \n";
    // for (int i=0; i<ess_bdr.Size(); i++){
    //     std::cout << ess_bdr[i]<<"\n";
    // }
    // std::cout << "ess_tdof_list: \n";
    // for (int i=0; i<ess_tdof_list.Size(); i++){
    //     std::cout << ess_tdof_list[i]<<"\n";
    // }
    // comment: ess_bdr contains 


    // Linform
    LinearForm b(&fespace);
    ConstantCoefficient one(1.0);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();

    // gridfunction
    GridFunction x(&fespace);
    x = 0.;
    std::cout << "size: "<<x.Size()<<"\n";

    // blf
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

    // solve
    GSSmoother M((SparseMatrix&)(*A));
    PCG(*A, M, B, X, 0, 200, 1e-12, 0.0);
    
    
    // instead of operatorptr using sparsemat:
    // mfem::SparseMatrix A;
    // a.FormSystemMatrix(ess_tdof_list,A);
    // GSSmoother M((SparseMatrix&)(A));
    // PCG(A, M, B, X, 0, 20000, 1e-12, 0.0);


    // Recover the solution as a finite element grid function.
    // linalg   system: a*x = b
    // gridfunc system: A*X = B
    a.RecoverFEMSolution(X, b, x);
    
    
    
    // what?! this changes also X !
    x[1] = 11;
    std::cout << x[1] << " " << X[1] << "\n";
    
    




    // visuals
    // ofstream mesh_ofs("refined.mesh");
    // mesh_ofs.precision(8);
    // mesh.Print(mesh_ofs);
    // ofstream sol_ofs("sol.gf");
    // sol_ofs.precision(8);
    // x.Save(sol_ofs);


    socketstream sol_sock("localhost", 19916);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << x << flush;

    delete fec;
}