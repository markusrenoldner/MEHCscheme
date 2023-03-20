#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"





// cleaned up version of a SINGLE timestep of 
// the decoupled dual field of the dirichlet_dual code







struct Parameters {
    double Re_inv = 1; // = 1/Re 
    double dt     = 0.01;
    double tmax   = 1*dt;
    int ref_steps = 3;
    int init_ref  = 0;
    int order     = 1;
    const char* mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    double t;
};


void u_0(const mfem::Vector &x, mfem::Vector &v);
void w_0(const mfem::Vector &x, mfem::Vector &v);
void   f(const mfem::Vector &x, mfem::Vector &v); 


int main() {

    // param
    Parameters param;
    double Re_inv = param.Re_inv;
    double dt     = param.dt;
    double tmax   = param.tmax;
    int ref_steps = param.ref_steps;
    int init_ref  = param.init_ref;
    int order     = param.order;

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        auto start = std::chrono::high_resolution_clock::now();

        // mesh
        const char *mesh_file = param.mesh_file;
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        int l;
        // dt *= 0.5; // TODO
        for (l = 0; l<init_ref+ref_step; l++) {
            mesh.UniformRefinement();
        } 
        std::cout << "----------ref: " << ref_step << "----------\n";

        // spaces
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);
        mfem::FiniteElementSpace RT(&mesh, fec_RT);
        mfem::FiniteElementSpace CG(&mesh, fec_CG);

        // essdofs
        mfem::Array<int> ND_ess_tdof;
        mfem::Array<int> ND_ess_tdof_0;
        mfem::Array<int> RT_ess_tdof;
        mfem::Array<int> ess_dof2;
        ND.GetBoundaryTrueDofs(ND_ess_tdof); 
        ND.GetBoundaryTrueDofs(ND_ess_tdof_0); 
        RT.GetBoundaryTrueDofs(ess_dof2);
        RT.GetBoundaryTrueDofs(RT_ess_tdof);
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            ND_ess_tdof[i] += RT.GetNDofs() ;
        }
        ess_dof2.Append(ND_ess_tdof);

        // gridfuncs
        mfem::GridFunction v(&RT);
        mfem::GridFunction w(&ND); 
        mfem::GridFunction q(&DG); q=0.;
        mfem::GridFunction z_exact(&RT);
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0); 
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        v.ProjectCoefficient(u_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        z_exact.ProjectCoefficient(w_0_coeff);

        // linear form
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::LinearForm f2(&RT);
        f2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f2.Assemble();
        
        // system size
        int size_2 = v.Size() + w.Size() + q.Size();
        std::cout << "size:"<<size_2<<"\n";
            
        // solution vector
        mfem::Vector y(size_2);
        y.SetVector(v,0);
        y.SetVector(w,v.Size());
        y.SetVector(q,v.Size()+w.Size());
        
        // Matrix M
        mfem::BilinearForm blf_M(&ND);
        blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_M.Assemble();
        blf_M.Finalize();
        mfem::SparseMatrix M_n(blf_M.SpMat());
        mfem::SparseMatrix M_nRe; // TODO
        mfem::SparseMatrix M_dt;
        M_dt = M_n;
        M_dt *= 1/dt;
        M_n *= -1.;
        M_nRe = M_n;
        M_nRe *= Re_inv/2.;
        M_dt.Finalize();
        M_nRe.Finalize();
        M_n.Finalize();
        
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

        // Matrix D
        mfem::MixedBilinearForm blf_D(&RT, &DG);
        blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator()); //=(div u,v)
        blf_D.Assemble();
        blf_D.Finalize();
        mfem::SparseMatrix D(blf_D.SpMat());
        mfem::SparseMatrix *DT_n;
        mfem::SparseMatrix D_n;
        DT_n = Transpose(D);
        D_n = D;
        D_n *= -1;
        *DT_n *= -1.;
        D.Finalize();
        D_n.Finalize(); // TODO
        DT_n->Finalize();

        // Matrix G
        mfem::MixedBilinearForm blf_G(&CG, &ND);
        blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
        blf_G.Assemble();
        blf_G.Finalize();
        mfem::SparseMatrix G(blf_G.SpMat());
        mfem::SparseMatrix *GT;
        GT = Transpose(G);
        G.Finalize();
        GT->Finalize();  

        // update R2 
        mfem::MixedBilinearForm blf_R2(&RT,&RT);
        mfem::VectorGridFunctionCoefficient z_gfcoeff(&z_exact);
        blf_R2.AddDomainIntegrator(
            new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_R2.Assemble();
        blf_R2.Finalize();
        mfem::SparseMatrix R2(blf_R2.SpMat());
        R2 *= 1./2.;
        R2.Finalize();

        // update NR
        mfem::MixedBilinearForm blf_NR(&RT,&RT); 
        mfem::ConstantCoefficient two_over_dt(2.0/dt);
        blf_NR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_NR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
        blf_NR.Assemble();
        blf_NR.Finalize();
        mfem::SparseMatrix NR(blf_NR.SpMat());
        NR *= 1./2.;
        NR.Finalize();





        // TODO : enforce ess dofs hardcore

        // matrix E1_left
        int rows_E1 = ess_dof2.Size();
        mfem::SparseMatrix E1_left (rows_E1, v.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            E1_left.Set(i, RT_ess_tdof[i], 1.);
            // std::cout << i << " " << RT_ess_tdof[i] << "\n";
        }
        E1_left.Finalize();
        // E1_left.PrintMatlab(std::cout);

        std::cout << rows_E1 << " " << v.Size() << " "<<w.Size()<<" "<<ess_dof2.Size()<<"\n";


        // matrix E1_cent
        mfem::SparseMatrix E1_cent (rows_E1, w.Size());

        // for (int i=RT_ess_tdof.Size(); i<rows_E1; i++) {
        for (int i=0; i<ND_ess_tdof.Size(); i++) {

            E1_cent.Set(i + RT_ess_tdof.Size(), ND_ess_tdof_0[i], 1.);
            // std::cout << i << " " << ND_ess_tdof_0[i] << "\n";
        }
        E1_cent.Finalize();

        mfem::DenseMatrix* dense = E1_cent.ToDenseMatrix();
        // dense->PrintMatlab(std::cout);

        std::cout << "------hi\n";
        // matrix E1_right
        mfem::SparseMatrix E1_right (rows_E1, q.Size());
        E1_right = 0.;




        // vector e
        mfem::Vector e(ess_dof2.Size());
        
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            e[i] = v[RT_ess_tdof[i]];
        }
        std::cout << "------hi\n";
        
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            e[i + RT_ess_tdof.Size()] = w[ND_ess_tdof_0[i]];
            // std::cout << i + RT_ess_tdof.Size()  << " " << ND_ess_tdof_0[i] << "\n";
        }









        // initialize system matrices
        mfem::Array<int> offsets_2 (4);
        offsets_2[0] = 0;
        offsets_2[1] = v.Size();
        offsets_2[2] = w.Size();
        offsets_2[3] = q.Size();
        offsets_2.PartialSum();

        mfem::Array<int> offsets_2_rows (5);
        offsets_2_rows[0] = 0;
        offsets_2_rows[1] = v.Size();
        offsets_2_rows[2] = w.Size();
        offsets_2_rows[3] = q.Size();
        offsets_2_rows[4] = ess_dof2.Size();
        offsets_2_rows.PartialSum();
        
        mfem::BlockOperator A2(offsets_2_rows, offsets_2);
        
        // update A2
        A2.SetBlock(0,0, &NR);
        A2.SetBlock(0,1, &C_Re);
        A2.SetBlock(0,2, DT_n);
        A2.SetBlock(1,0, &CT_Re);
        A2.SetBlock(1,1, &M_nRe);
        A2.SetBlock(2,0, &D_n);

        A2.SetBlock(3,0, &E1_left); //TODO
        A2.SetBlock(3,1, &E1_cent);
        A2.SetBlock(3,2, &E1_right);

        // initialize rhs
        mfem::Vector b2(size_2 + ess_dof2.Size()); 
        mfem::Vector b2sub(v.Size());
        b2 = 0.0;
        b2sub = 0.0;
        N_dt.AddMult(v,b2sub);
        R2.AddMult(v,b2sub,-1);
        C_Re.AddMult(w,b2sub,-1);
        b2.AddSubVector(f2,0);
        b2.AddSubVector(b2sub,0);
        b2.AddSubVector(e, size_2);

        // form linear system with BC
        // mfem::Operator *A2_BC;
        // mfem::Vector Y = y;
        // mfem::Vector B2;


        // transpose here: // TODO
        mfem::TransposeOperator AT2 (&A2);
        mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
        mfem::Vector ATb2 (size_2);
        A2.MultTranspose(b2,ATb2);

        // A2.FormLinearSystem(ess_dof2, y, b2, A2_BC, Y, B2);

        double tol = 1e-10;
        int iter = 10000000;  
        mfem::MINRES(ATA2, ATb2, y, 0, iter, tol*tol, tol*tol);
     // mfem::MINRES(*A2_BC, B2, Y, 0, iter, tol*tol, tol*tol);

        // get gridfunction back
        mfem::Array<int> v_dofs (v.Size());
        std::iota(&v_dofs[0], &v_dofs[v.Size()], 0);
        // A2.RecoverFEMSolution(Y, b2, y);
        y.GetSubVector(v_dofs, v);

        // error
        double err_L2_v = v.ComputeL2Error(u_0_coeff);
        std::cout << "L2err of v = "<< err_L2_v<<"\n";

        // free memory
        delete fec_DG;
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;

    } // refinement loop
}

// cos squared init cond that satifies
// dirichlet BC, divu=0 and is C1-continuous (=> w is C continuous)
// void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
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

// void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;
   
//     double cos = std::cos(C*(X*X+Y*Y+Z*Z));
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z));
//     double cos2 = cos*cos;

//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = - 4*C*X*Z*sin*cos;
//         returnvalue(1) = - 4*C*Y*Z*sin*cos;
//         returnvalue(2) = - 2*cos2 + 4*C*X*X*sin*cos + 4*C*Y*Y*sin*cos;
//     }
//     else {
//         returnvalue(0) = 0;
//         returnvalue(1) = 0;
//         returnvalue(2) = 0;
//     }
// }

// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double Re_inv = 1.;
//     double pi = 3.14159265358979323846;
//     double C = 10;
//     double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     double cos = std::cos(C*(X*X+Y*Y+Z*Z) );
//     double cos3 = cos*cos*cos;
//     double cosof2 = std::cos(2*C*(X*X+Y*Y+Z*Z) );
//     double sin = std::sin(C*(X*X+Y*Y+Z*Z) );
//     double sinof2 = std::sin(2*C*(X*X+Y*Y+Z*Z) );
//     double cos4 = cos*cos*cos*cos;
    
//     if (X*X + Y*Y + Z*Z < R*R) {
//         returnvalue(0) = 2*X*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) + Re_inv * (2*C*Y* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
//         returnvalue(1) = 2*Y*cos3 * (-cos + 2*C*(X*X+Y*Y)*sin) - Re_inv * (2*C*X* (4*C*(X*X+Y*Y+Z*Z)) * cosof2 + 5*sinof2);
//         returnvalue(2) = 4*C*(X*X+Y*Y)*Z * cos3 *sin;
//     }
//     else {
//         returnvalue(0) = 0.; 
//         returnvalue(1) = 0.; 
//         returnvalue(2) = 0.;
//     }   
// }

void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = std::sin(Y);
    returnvalue(1) = std::sin(Z);
    returnvalue(2) = 0.;
}
void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = -std::cos(Z);
    returnvalue(1) = 0.;
    returnvalue(2) = -std::cos(Y);
}
void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    Parameters param;
    double Re_inv = param.Re_inv; // = 1/Re 
    
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    returnvalue(0) = std::sin(Y)*Re_inv + std::cos(Y)*std::sin(Z);
    returnvalue(1) = -std::cos(Y)*std::sin(Y) + std::sin(Z)*Re_inv;
    returnvalue(2) = - std::cos(Z)*std::sin(Z);
}