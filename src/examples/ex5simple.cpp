//                                MFEM Example 5
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//
//                                 k*u + grad p = f
//                                 - div u      = g

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

int main(int argc, char *argv[])
{

    // 1. Parse command-line options.
    // const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    const char *mesh_file = "extern/mfem-4.5/data/ref-square.mesh";
    int order = 1;
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Refine the mesh to increase the resolution.
    for (int l = 0; l < 1; l++) {
        mesh->UniformRefinement();
    }

    // 5. Define a finite element space on the mesh. Here we use the
    //    Raviart-Thomas finite elements of the specified order.
    FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
    FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

    FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
    FiniteElementSpace *W_space = new FiniteElementSpace(mesh, l2_coll);

    // 6. Define the BlockStructure of the problem, i.e. define the array of
    //    offsets for each variable. The last component of the Array is the sum
    //    of the dimensions of each block.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = R_space->GetVSize();
    block_offsets[2] = W_space->GetVSize();
    block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";

    // 7. Define the coefficients, analytical solution, and rhs of the PDE.
    ConstantCoefficient k(1.0);

    VectorFunctionCoefficient fcoeff(dim, fFun);
    FunctionCoefficient fnatcoeff(f_natural);
    FunctionCoefficient gcoeff(gFun);

    VectorFunctionCoefficient ucoeff(dim, uFun_ex);
    FunctionCoefficient pcoeff(pFun_ex);

    // 8. Allocate memory (x, rhs) for the analytical solution and the right hand
    //    side.  Define the GridFunction u,p for the finite element solution and
    //    linear forms fform and gform for the right hand side.  The data
    //    allocated by x and rhs are passed as a reference to the grid functions
    //    (u,p) and the linear forms (fform, gform).
    BlockVector x(block_offsets);
    BlockVector rhs(block_offsets);

    LinearForm *fform(new LinearForm);
    fform->Update(R_space, rhs.GetBlock(0), 0);
    fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
    fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
    fform->Assemble();
    fform->SyncAliasMemory(rhs);

    LinearForm *gform(new LinearForm);
    gform->Update(W_space, rhs.GetBlock(1), 0);
    gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
    gform->Assemble();
    gform->SyncAliasMemory(rhs);

    rhs = 1; // TODO

    // 9. Assemble the finite element matrices for the Darcy operator
    //
    //                            D = [ M  B^T ]
    //                                [ B   0  ]
    //     where:
    //
    //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
    //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
    BilinearForm *mVarf(new BilinearForm(R_space));
    MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));

    mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
    mVarf->Assemble();
    mVarf->Finalize();

    bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->Finalize();

    BlockOperator darcyOp(block_offsets);

    TransposeOperator *Bt = NULL;

    SparseMatrix &M(mVarf->SpMat());
    SparseMatrix &B(bVarf->SpMat());
    B *= -1.;
    Bt = new TransposeOperator(&B);

    darcyOp.SetBlock(0,0, &M);
    darcyOp.SetBlock(0,1, Bt);
    darcyOp.SetBlock(1,0, &B);

    // 11. Solve the linear system with MINRES.
    //     Check the norm of the unpreconditioned residual.
    int maxIter(1000);
    double rtol(1.e-6);
    double atol(1.e-10);

    MINRESSolver solver;
    solver.SetAbsTol(atol);
    solver.SetRelTol(rtol);
    solver.SetMaxIter(maxIter);
    solver.SetOperator(darcyOp);
    solver.SetPrintLevel(1);
    x = 0.0;
    solver.Mult(rhs, x);
    // for (int i=0; i<block_offsets[2]; i++){
    //    std::cout << x.Elem(i) << "\n";
    // }

    // 12. Create the grid functions u and p. Compute the L2 error norms.
    GridFunction u, p;
    u.MakeRef(R_space, x.GetBlock(0), 0);
    p.MakeRef(W_space, x.GetBlock(1), 0);

    int order_quad = max(2, 2*order+1);
    const IntegrationRule *irs[Geometry::NumGeom];
    for (int i=0; i < Geometry::NumGeom; ++i)
    {
        irs[i] = &(IntRules.Get(i, order_quad));
    }

    double err_u  = u.ComputeL2Error(ucoeff, irs);
    double norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
    double err_p  = p.ComputeL2Error(pcoeff, irs);
    double norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

    std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
    std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

    // 13. Save the mesh and the solution. This output can be viewed later using
    //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
    //     sol_p.gf".
    {
        ofstream mesh_ofs("ex5.mesh");
        mesh_ofs.precision(8);
        mesh->Print(mesh_ofs);

        ofstream u_ofs("sol_u.gf");
        u_ofs.precision(8);
        u.Save(u_ofs);

        ofstream p_ofs("sol_p.gf");
        p_ofs.precision(8);
        p.Save(p_ofs);
    }

    // 16. Send the solution by socket to a GLVis server.
    char vishost[] = "localhost";
    int  visport   = 19916;
    socketstream u_sock(vishost, visport);
    u_sock.precision(8);
    u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
    socketstream p_sock(vishost, visport);
    p_sock.precision(8);
    p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;

    // 17. Free the used memory.
    delete fform;
    delete gform;
    delete Bt;
    delete mVarf;
    delete bVarf;
    delete W_space;
    delete R_space;
    delete l2_coll;
    delete hdiv_coll;
    delete mesh;
    return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);
    if (x.Size() == 3)
    {
        zi = x(2);
    }

    u(0) = - exp(xi)*sin(yi)*cos(zi);
    u(1) = - exp(xi)*cos(yi)*cos(zi);

    if (x.Size() == 3)
    {
        u(2) = exp(xi)*sin(yi)*sin(zi);
    }
}

// Change if needed
double pFun_ex(const Vector & x)
{
    double xi(x(0));
    double yi(x(1));
    double zi(0.0);

    if (x.Size() == 3)
    {
        zi = x(2);
    }

    return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
    f = 0.0;
}

double gFun(const Vector & x)
{
    if (x.Size() == 3)
    {
        return -pFun_ex(x);
    }
    else
    {
        return 0;
    }
}

double f_natural(const Vector & x)
{
    return (-pFun_ex(x));
}
