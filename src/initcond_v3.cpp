#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"


// check div of init cond of periodic conservation test


void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);

void u_0(const mfem::Vector &x, mfem::Vector &v);
void u_0b(const mfem::Vector &x, mfem::Vector &v);




int main(int argc, char *argv[]) {

    std::cout << "---------------launch MEHC---------------\n";

    for (int max=0; max<5; max++) {

        // mesh
        // const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh"; 
        const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh"; 
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        for (int l = 0; l < max; l++) {mesh.UniformRefinement();} 
        mesh.UniformRefinement();
        // mesh.UniformRefinement();
        // mesh.UniformRefinement();
        // mesh.UniformRefinement();
        // mesh.UniformRefinement();
        // mesh.UniformRefinement();

        // FE spaces (CG \in H1, DG \in L2)
        int order = 1;
        mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order,dim);
        mfem::FiniteElementSpace CG(&mesh, fec_CG);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);
        mfem::FiniteElementSpace RT(&mesh, fec_RT);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND); // u = 4.3;
        mfem::GridFunction p(&CG); // u = 4.3;

        std::cout << "size "<< u.Size() << "\n";

        // initial condition
        // mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0b);
        u.ProjectCoefficient(u_0_coeff);
        
        char vishost[] = "localhost";
        int  visport   = 19916;
        mfem::socketstream sol_sock_2(vishost, visport);
        sol_sock_2.precision(8);
        sol_sock_2 << "solution\n" << mesh << u << std::flush;

        
        // Matrix G0
        mfem::MixedBilinearForm blf_G0(&CG, &ND);
        blf_G0.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
        blf_G0.Assemble();
        blf_G0.Finalize();
        mfem::SparseMatrix G0(blf_G0.SpMat());
        mfem::SparseMatrix *G0T;
        G0T = Transpose(G0);
        G0.Finalize();
        G0T->Finalize();
        
        mfem::Vector mass_vec1 (p.Size());
        G0T->Mult(u,mass_vec1);
        std::cout <<"mass="<< mass_vec1.Norml2() << "\n";
    

        // free memory
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;
        delete fec_DG;    

    } // refinement steps

    std::cout << "---------------finish MEHC---------------\n";

} // main


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

void u_0b(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
    double pi = 3.14159265358979323846;
    // int dim = x.Size();
    // std::cout << "size ----" <<returnvalue.Size() << "\n";

    returnvalue.SetSize(3);
    // returnvalue(0) = std::cos(pi*x.Elem(2)); 
    // returnvalue(1) = std::sin(pi*x.Elem(2));
    // returnvalue(2) = std::sin(pi*x.Elem(0));
    
    // returnvalue(0) = x(1);
    // returnvalue(1) = -x(0);
    // returnvalue(2) = 0;
    returnvalue(0) = 1;
    returnvalue(1) = 0;
    returnvalue(2) = 0;
}
