#include <fstream>
#include <iostream>
#include <algorithm>
#include "mfem.hpp"

#include <chrono>



// this implements the problem: find u,p st
// d/dt u + grad p = 0
// div u           = 0

// which forces p=0 and therefore u=const=u_0

// this system is the result from cancelling f with uxw + 1/Re(curl w)
// and therefore equivalent to the full system!
// this implementation converges => the forcing function might contain a bug 
 

void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void     u_0(const mfem::Vector &x, mfem::Vector &v);
void     w_0(const mfem::Vector &x, mfem::Vector &v);
void       f(const mfem::Vector &x, mfem::Vector &v); 
// void u_exact(const mfem::Vector &x, mfem::Vector &returnvalue);

int main(int argc, char *argv[]) {

    // simulation parameters
    // careful: Re also has to be defined in the manufactured sol
    double Re_inv = 1; // = 1/Re 
    double dt = 0.2;
    // dt = std::sqrt(3)*1;
    double tmax = 50*dt;
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
        mfem::Array<int> ess_dof2;//, ess_dof2;
        ess_dof2.Append(RT_ess_tdof);
        // for (int i=0; i<ND_ess_tdof.Size(); i++) {
        //     ND_ess_tdof[i] += RT.GetNDofs() ;
        // }
        // ess_dof2.Append(ND_ess_tdof);

        // unkowns and gridfunctions
        mfem::GridFunction v(&RT); //v = 3.;
        mfem::GridFunction q(&DG); q=0.; //q = 9.3;
        mfem::GridFunction v_exact(&RT);

 


        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_0); 
        v.ProjectCoefficient(u_0_coeff);
        v_exact.ProjectCoefficient(u_0_coeff);


        // initialize solution vectors
        int size_2=v.Size() + q.Size();
        mfem::Vector y(size_2);
        y.SetVector(v,0);
        y.SetVector(q,v.Size());

        // helper dofs
        mfem::Array<int> v_dofs (v.Size());
        mfem::Array<int> q_dofs (q.Size());
        std::iota(&v_dofs[0], &v_dofs[v.Size()], 0);
        std::iota(&q_dofs[0], &q_dofs[q.Size()], v.Size());
            


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
            
  

        // initialize system matrices
        mfem::Array<int> offsets_2 (3);
        offsets_2[0] = 0;
        offsets_2[1] = v.Size();
        offsets_2[2] = q.Size();
        offsets_2.PartialSum();
        mfem::BlockOperator A2(offsets_2);
            

        // initialize rhs
        mfem::Vector b2(size_2); 
        mfem::Vector b2sub(v.Size());

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////



        // form linear system with BC
        mfem::Operator *A1_BC;
        mfem::Operator *A2_BC;
        mfem::Vector X;
        mfem::Vector B1;
        mfem::Vector Y;
        mfem::Vector B2;

        // solve 
        double tol = 1e-15;
        int iter = 1000000;  

        // time loop
        for (double t = dt ; t < tmax+dt ; t+=dt) {
            // std::cout << "--- t = "<<t<<"\n";



            // update A2
            A2.SetBlock(0,0, &N_dt);
            A2.SetBlock(0,1, DT_n);
            A2.SetBlock(1,0, &D);
            
            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N_dt.AddMult(v,b2sub);
            b2.AddSubVector(b2sub,0);

            // transpose here:
            mfem::TransposeOperator AT2 (&A2);
            mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
            mfem::Vector ATb2 (size_2);
            A2.MultTranspose(b2,ATb2);

            // form linear system with BC
            ATA2.FormLinearSystem(ess_dof2, y, ATb2, A2_BC, Y, B2);

            // solve  
            mfem::MINRES(*A2_BC, B2, Y, 0, iter, tol*tol, tol*tol);
            A2.RecoverFEMSolution(Y, b2, y);
            y.GetSubVector(v_dofs, v);
            y.GetSubVector(q_dofs, q);                



        } // time loop

        // convergence error
        double err_L2_v = v.ComputeL2Error(u_exact_coeff);
        std::cout << "L2err of v = "<< err_L2_v<<"\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        // std::cout << "runtime = " << duration.count() << "ms" << std::endl;

        visuals
        char vishost[] = "localhost";
        int  visport   = 19916;
        mfem::socketstream v_sock(vishost, visport);
        v_sock.precision(8);
        v_sock << "solution\n" << mesh << v << "window_title 'v in hdiv'" << std::endl;

        mfem::socketstream ve_sock(vishost, visport);
        ve_sock.precision(8);
        ve_sock << "solution\n" << mesh << v_exact << "window_title 'v_0'" << std::endl;

        // mfem::socketstream q_sock(vishost, visport);
        // q_sock.precision(8);
        // q_sock << "solution\n" << mesh << q << "window_title 'q'" << std::endl;

        
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
    double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
    double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
    double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
    double cos4 = cos*cos*cos*cos;
    // double eC2 = 8*C*C;
    
    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = -X*cos4 + 1*Re_inv*( 
                          2*Y*C*(4*C*(X*X+Y*Y+Z*Z)*cosof2 + 5*sinof2) );

        returnvalue(1) = -Y*cos4 - 1*Re_inv*(
                          2*X*C*(4*C*(X*X+Y*Y+Z*Z)*cosof2 + 5*sinof2) );


        returnvalue(2) = 0.;
    }
    else {
        returnvalue(0) = 0.; 
        returnvalue(1) = 0.; 
        returnvalue(2) = 0.;
    }   
}

        // returnvalue(0) = -X*cos4 + 1*Re_inv*( 
        //                   2*C*(4*C*(X*X+Y*Y+Z*Z)*cos+5*sin) );
        // returnvalue(1) = -Y*cos4 - 1*Re_inv*(
        //                   2*C*(4*C*(X*X+Y*Y+Z*Z)*cos+5*sin));