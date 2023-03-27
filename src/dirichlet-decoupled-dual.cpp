#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"





// cleaned up version of a SINGLE timestep of 
// the decoupled dual field of the dirichlet_dual code


// use wouters trick for the BC







struct Parameters {
    double Re_inv = 1.; // = 1/Re 
    // double Re_inv = 0.; // = 1/Re 
    // double Re_inv = 1.; // = 1/Re 
    double dt     = 0.01;
    double tmax   = 1*dt;
    int ref_steps = 4;
    int init_ref  = 0;
    int order     = 1;
    double tol    = 1e-7;
    std::string outputfile = "out/rawdata/dirichlet-dual-conv-Re100.txt";
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
    double tol    = param.tol;

    std::cout << "Re_inv=" << Re_inv << "\n";

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        auto start = std::chrono::high_resolution_clock::now();

        // output file 
        std::string outputfile = param.outputfile;
        std::ofstream file;
        file.precision(6);
        // file.open(outputfile);
        file.open(outputfile, std::ios::app);

        // mesh
        const char *mesh_file = param.mesh_file;
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        int l;
        dt *= 0.5; // TODO
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
        mfem::Array<int> RT_ess_tdof;
        mfem::Array<int> ND_ess_tdof;
        mfem::Array<int> ND_ess_tdof_0;
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

        // matrix E2_left
        int rows_E2 = ess_dof2.Size();
        mfem::SparseMatrix E2_left (rows_E2, v.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            E2_left.Set(i, RT_ess_tdof[i], 1.);
            // std::cout << i << " " << RT_ess_tdof[i] << "\n";
        }
        E2_left.Finalize();
        // E2_left.PrintMatlab(std::cout);

        std::cout << rows_E2 << " " << v.Size() << " "<<w.Size()<<" "<<ess_dof2.Size()<<"\n";


        // matrix E2_cent
        mfem::SparseMatrix E2_cent (rows_E2, w.Size());
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            E2_cent.Set(i + RT_ess_tdof.Size(), ND_ess_tdof_0[i], 1.);
            // std::cout << i << " " << ND_ess_tdof_0[i] << "\n";
        }
        E2_cent.Finalize();

        mfem::DenseMatrix* dense = E2_cent.ToDenseMatrix();
        // dense->PrintMatlab(std::cout);

        
        // matrix E2_right
        mfem::SparseMatrix E2_right (rows_E2, q.Size());
        E2_right = 0.;
        E2_right.Finalize();


        // vector e2
        mfem::Vector e2(ess_dof2.Size());
        
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            e2[i] = v[RT_ess_tdof[i]];
        }
        
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            e2[i + RT_ess_tdof.Size()] = w[ND_ess_tdof_0[i]];
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
        A2.SetBlock(1,0, CT);
        A2.SetBlock(1,1, &M_n);
        A2.SetBlock(2,0, &D_n);

        A2.SetBlock(3,0, &E2_left); //TODO
        A2.SetBlock(3,1, &E2_cent);
        A2.SetBlock(3,2, &E2_right);

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
        b2.AddSubVector(e2, size_2);



        

        // remove unnecessary equations from system matrix:
        // corresponding to essdofs
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            NR.EliminateRow(RT_ess_tdof[i]);
            C_Re.EliminateRow(RT_ess_tdof[i]);
            DT_n->EliminateRow(RT_ess_tdof[i]);

            b2[RT_ess_tdof[i]] = 0.;
        }
        for (int i=0; i<ND_ess_tdof_0.Size(); i++) {
            CT->EliminateRow(ND_ess_tdof_0[i]);
            M_n.EliminateRow(ND_ess_tdof_0[i]);
        }

        // NR, C_Re, DT_n, CT_Re, MnRe
        // A2.SetBlock(0,0, &NR);
        // A2.SetBlock(0,1, &C_Re);
        // A2.SetBlock(0,2, DT_n);
        // A2.SetBlock(1,0, &CT_Re);
        // A2.SetBlock(1,1, &M_nRe);
        // A2.SetBlock(2,0, &D_n);








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

        // double tol = 1e-7;
        int iter = 10000000;  
        mfem::MINRES(ATA2, ATb2, y, 0, iter, tol*tol, tol*tol);
     // mfem::MINRES(*A2_BC, B2, Y, 0, iter, tol*tol, tol*tol);

        // get gridfunction back
        mfem::Array<int> v_dofs (v.Size());
        std::iota(&v_dofs[0], &v_dofs[v.Size()], 0);
        // A2.RecoverFEMSolution(Y, b2, y);
        y.GetSubVector(v_dofs, v);

        // } // time

        // error
        double err_L2_v = v.ComputeL2Error(u_0_coeff);
        std::cout << "L2err of v = "<< err_L2_v<<"\n";

        // write to file
        file << std::setprecision(15) << std::fixed << std::pow(1/2.,ref_step) << ","   
        << err_L2_v <<  "\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        std::cout << "runtime = " << duration.count() << "ms" << std::endl;


        // free memory
        delete fec_DG;
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;
        
        // close file
        file.close();

    } // refinement loop
}



void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 

    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    // returnvalue(0) = std::sin(Y);
    // returnvalue(1) = std::sin(Z);
    // returnvalue(2) = 0.;
    // returnvalue(0) = 1;
    // returnvalue(1) = 0.;
    // returnvalue(2) = 0.;
    returnvalue(0) = -Y;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}
void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    // returnvalue(0) = -std::cos(Z);
    // returnvalue(1) = 0.;
    // returnvalue(2) = -std::cos(Y);
//     returnvalue(0) = 0.;
//     returnvalue(1) = 0.;
//     returnvalue(2) = 0.;
    returnvalue(0) = 0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 1.;
}
void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    Parameters param;
    double Re_inv = param.Re_inv; // = 1/Re 
    
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;

    // returnvalue(0) = std::sin(Y)*Re_inv + std::cos(Y)*std::sin(Z);
    // returnvalue(1) = -std::cos(Y)*std::sin(Y) + std::sin(Z)*Re_inv;
    // returnvalue(2) = - std::cos(Z)*std::sin(Z);

    // returnvalue(0) = 0.;
    // returnvalue(1) = 0;
    // returnvalue(2) = 0.;

    returnvalue(0) = 0.;
    returnvalue(1) = -Y;
    returnvalue(2) = 0.;


}