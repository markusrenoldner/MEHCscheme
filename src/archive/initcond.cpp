#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

#include <chrono>



// init cond on dirichlet domain
// all vector spaces adapted (H10, H0curl, H0div, L2 with lagr mult)




void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void     u_0(const mfem::Vector &x, mfem::Vector &v);
void     u_0_adapted(const mfem::Vector &x, mfem::Vector &v);
void     w_0(const mfem::Vector &x, mfem::Vector &v);
void       f(const mfem::Vector &x, mfem::Vector &v); 
// void u_exact(const mfem::Vector &x, mfem::Vector &returnvalue);

int main(int argc, char *argv[]) {

    // simulation parameters
    double Re_inv = 0.01; // = 1/Re 
    double dt = 1/20.;
    double tmax = 3*dt; //tmax=0.;
    int ref_steps = 0;
    std::cout <<"----------\n"<<"Re:   "<<1/Re_inv <<"\ndt:   "<<dt<< "\ntmax: "<<tmax<<"\n----------\n";


    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    mfem::Mesh mesh(mesh_file, 1, 1); 
    int dim = mesh.Dimension(); 
    

    mesh.UniformRefinement();
    mesh.UniformRefinement();
    mesh.UniformRefinement();
    mesh.UniformRefinement();

    // FE spaces; DG \in L2, ND \in Hcurl, RT \in Hdiv, CG \in H1
    int order = 1;
    mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order,dim);
    mfem::FiniteElementCollection *fec_ND0 = new mfem::ND_FECollection(order,dim);
    mfem::FiniteElementCollection *fec_RT0 = new mfem::RT_FECollection(order-1,dim);
    mfem::FiniteElementCollection *fec_CG0 = new mfem::H1_FECollection(order,dim);
    mfem::FiniteElementSpace DG(&mesh, fec_DG);
    mfem::FiniteElementSpace ND0(&mesh, fec_ND0);
    mfem::FiniteElementSpace RT0(&mesh, fec_RT0);
    mfem::FiniteElementSpace CG0(&mesh, fec_CG0);

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND0); //u = 4.3;
    mfem::GridFunction z(&RT0); //z = 5.3;
    mfem::GridFunction p(&CG0); p=0.; //p = 6.3;
    mfem::GridFunction v(&RT0); //v = 3.;
    mfem::GridFunction w(&ND0); //w = 3.; 
    mfem::GridFunction q(&DG); q=0.; //q = 9.3;
    mfem::GridFunction f1(&ND0);
    mfem::GridFunction f2(&RT0);
    mfem::Vector lam (1); // lagrange multiplier
    lam[0] = 0.;

    // initial condition
    mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
    mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
    mfem::VectorFunctionCoefficient f_coeff(dim, f);
    mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_0); 
    u.ProjectCoefficient(u_0_coeff);
    v.ProjectCoefficient(u_0_coeff);
    z.ProjectCoefficient(w_0_coeff);
    w.ProjectCoefficient(w_0_coeff);
    f1.ProjectCoefficient(f_coeff);
    f2.ProjectCoefficient(f_coeff);


    // visuals
    // std::ofstream mesh_ofs("refined.mesh");
    // mesh_ofs.precision(8);
    // mesh.Print(mesh_ofs);
    // std::ofstream sol_ofs("sol.gf");
    // sol_ofs.precision(8);
    // w.Save(sol_ofs);
    char vishost[] = "localhost";
    int  visport   = 19916;
    mfem::socketstream sol_sock(vishost, visport);
    sol_sock.precision(8);
    sol_sock << "solution\n" << mesh << w << std::flush;



    // system size
    int size_1 = u.Size() + z.Size() + p.Size();
    int size_2 = v.Size() + w.Size() + q.Size() + 1;
    std::cout<< "size1/u/z/p: "<<size_1<<"/"<<u.Size()<<"/"<<z.Size()<<"/"<<p.Size()<<"\n";
    std::cout<< "size2/v/w/q/lam: "<<size_2<<"/"<<v.Size()<<"/"<<w.Size()<<"/"<<q.Size()<<"/"<<1<<"\n"<<"---\n";
    





    /////////////////////////////////////

    // initialize solution vectors
    mfem::Vector x(size_1);
    mfem::Vector y(size_2);
    x.SetVector(u,0);
    x.SetVector(z,u.Size());
    x.SetVector(p,u.Size()+z.Size());
    y.SetVector(v,0);
    y.SetVector(w,v.Size());
    y.SetVector(q,v.Size()+w.Size());
    y.SetVector(lam,v.Size()+w.Size()+1);

    // helper dofs
    mfem::Array<int> u_dofs (u.Size());
    mfem::Array<int> z_dofs (z.Size());
    mfem::Array<int> p_dofs (p.Size());
    mfem::Array<int> v_dofs (v.Size());
    mfem::Array<int> w_dofs (w.Size());
    mfem::Array<int> q_dofs (q.Size());
    std::iota(&u_dofs[0], &u_dofs[u.Size()], 0);
    std::iota(&z_dofs[0], &z_dofs[z.Size()], u.Size());
    std::iota(&p_dofs[0], &p_dofs[p.Size()], u.Size()+z.Size());
    std::iota(&v_dofs[0], &v_dofs[v.Size()], 0);
    std::iota(&w_dofs[0], &w_dofs[w.Size()], v.Size());
    std::iota(&q_dofs[0], &q_dofs[q.Size()], v.Size()+w.Size());
    mfem::Array<int> lam_dofs (1);
    lam_dofs[0] = size_2+1;

    // extract solution values u,z,p from eulerstep
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(z_dofs, z);
    x.GetSubVector(p_dofs, p);
    y.GetSubVector(v_dofs, v);
    y.GetSubVector(w_dofs, w);
    y.GetSubVector(q_dofs, q);     
    y.GetSubVector(lam_dofs, lam);
    // std::cout <<"lam = "<< lam[0] << "\n";



    // visuals
    // std::ofstream mesh_ofs("refined.mesh");
    // mesh_ofs.precision(8);
    // mesh.Print(mesh_ofs);
    // std::ofstream sol_ofs("sol.gf");
    // sol_ofs.precision(8);
    // u.Save(sol_ofs);
    // char vishost[] = "localhost";
    // int  visport   = 19916;
    // mfem::socketstream sol_sock(vishost, visport);
    // sol_sock.precision(8);
    // sol_sock << "solution\n" << mesh << u << std::flush;
        




    // convergence error
    double err_L2_u = u.ComputeL2Error(u_exact_coeff);
    double err_L2_v = v.ComputeL2Error(u_exact_coeff);
    mfem::GridFunction v_ND (&ND0);
    v_ND.ProjectGridFunction(v);
    double err_L2_diff = 0;
    for (int i=0; i<u.Size(); i++) {
        err_L2_diff += ((u(i)-v_ND(i))*(u(i)-v_ND(i)));
    }
    // std::cout << "L2err of v = "<< err_L2_v<<"\n";
    // std::cout << "L2err of u = "<< err_L2_u<<"\n";
    std::cout << "L2err(u-v) = "<< std::pow(err_L2_diff, 0.5) <<"\n";


    // free memory
    delete fec_DG;
    delete fec_CG0;
    delete fec_ND0;
    delete fec_RT0;

}


void u_0_adapted(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    
    double pi = 3.14159265358979323846;
    double C = 10;
    double DX = 0.5;
    double DY = 0.5;
    double DZ = 0.5;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish

    if (std::pow((x(0)-DX),2) +
        std::pow((x(1)-DY),2) +
        std::pow((x(2)-DZ),2) < std::pow(R,2)) {

        returnvalue(0) = 1;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
    else {
        returnvalue(0) = 0;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
}

// cos squared init cond that satifies
// dirichlet BC, divu=0 and is C1-continuous (=> w is C0 continuous)
void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double C = 10;
    double DX = 0.5;
    double DY = 0.5;
    double DZ = 0.5;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish

    double cos = std::cos(C*(std::pow((x(0)-DX),2) 
                            +std::pow((x(1)-DY),2) 
                            +std::pow((x(2)-DZ),2)));
    double cos2 = cos*cos;

    if (std::pow((x(0)-DX),2) +
        std::pow((x(1)-DY),2) +
        std::pow((x(2)-DZ),2) < std::pow(R,2)) {

        returnvalue(0) = (x(1)-DY) * cos2;
        returnvalue(1) = -1* (x(0)-DX) * cos2;
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
    double DX = 0.5;
    double DY = 0.5;
    double DZ = 0.5;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    
    double cos = std::cos(C*(std::pow((x(0)-DX),2) +
                             std::pow((x(1)-DY),2) +
                             std::pow((x(2)-DZ),2)) );
    double sin = std::sin(C*(std::pow((x(0)-DX),2) +
                             std::pow((x(1)-DY),2) +
                             std::pow((x(2)-DZ),2)) );
    double cos2 = cos*cos;
    double sin2 = sin*sin;

    if (std::pow((x(0)-DX),2) +
        std::pow((x(1)-DY),2) +
        std::pow((x(2)-DZ),2) < std::pow(R,2)) {

        returnvalue(0) = - 4*C*(x(0)-DX)*(x(2)-DZ)*sin*cos;
        returnvalue(1) = - 4*C*(x(1)-DY)*(x(2)-DZ)*sin*cos;
        returnvalue(2) = - 2*cos2 
                         + 4*C*std::pow((x(0)-DX),2)*sin*cos
                         + 4*C*std::pow((x(1)-DY),2)*sin*cos;
    }
    else {
        returnvalue(0) = 0;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
}

void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    
    //TODO:  forcing
    returnvalue(0) = 0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}

// void u_exact_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
//     double pi = 3.14159265358979323846;
//     double Re = 500.; // chose Re here!
//     double nu = 1*1/Re; // = u*L/Re
//     double t = 0.15;
//     double F = std::exp(-2*nu*t);

//     returnvalue(0) =     std::cos(x(0)*pi)*std::sin(x(1)*pi) * F;
//     returnvalue(1) = -1* std::sin(x(0)*pi)*std::cos(x(1)*pi) * F;
//     returnvalue(2) = 0;
// }