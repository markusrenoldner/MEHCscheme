#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"




// decouple the systems by replacing the coupling vorticities by its exact
// valus (static manufactured solution)

// this file contains the primal system


// use wouters trick for the BC



struct Parameters {
    double Re_inv = 0; // = 1/Re 
    double dt     = 0.1;
    double tmax   = 3*dt;
    int ref_steps = 4;
    int init_ref  = 0;
    int order     = 1;
    std::string outputfile = "out/rawdata/dirichlet-primal-conv-invisc.txt";
    const char* mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    double t;
};

void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void     u_0(const mfem::Vector &x, mfem::Vector &v);
void     w_0(const mfem::Vector &x, mfem::Vector &v);
void       f(const mfem::Vector &x, mfem::Vector &v); 


int main(int argc, char *argv[]) {

    // simulation parameters
    Parameters param;
    double Re_inv = param.Re_inv; 
    double dt     = param.dt;
    double tmax   = param.tmax;
    int ref_steps = param.ref_steps;
    int init_ref  = param.init_ref;
    int order     = param.order;

    std::cout << "Re=" << Re_inv << "\n";
    
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
        // std::cout << "----------ref: " << ref_step << "----------\n";

        // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
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
        mfem::Array<int> RT_ess_tdof_0;
        mfem::Array<int> ess_dof1;
        // RT.GetBoundaryTrueDofs(RT_ess_tdof); 
        // RT.GetBoundaryTrueDofs(RT_ess_tdof_0); 
        // ND.GetBoundaryTrueDofs(ess_dof1);
        // ND.GetBoundaryTrueDofs(ND_ess_tdof); 
        // for (int i=0; i<RT_ess_tdof.Size(); i++) {
        //     RT_ess_tdof[i] += ND.GetNDofs() ;
        // }
        // ess_dof1.Append(RT_ess_tdof);

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND); //u = 4.3;
        mfem::GridFunction z(&RT); //z = 5.3;
        mfem::GridFunction p(&CG); p=0.; //p = 6.3;
        mfem::GridFunction w_exact(&ND);
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        u.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w_exact.ProjectCoefficient(w_0_coeff);

        // linearform for forcing term
        mfem::VectorFunctionCoefficient f_coeff(dim, f);
        mfem::LinearForm f1(&ND);
        f1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        f1.Assemble();

        // linearform for boundary integral
        mfem::LinearForm lform_zxn(&ND);
        lform_zxn.AddBoundaryIntegrator(new mfem::VectorFEBoundaryTangentLFIntegrator(w_0_coeff)); // !!!
        lform_zxn.Assemble();
        // lform_zxn *= Re_inv;
        lform_zxn *= -1.*Re_inv;

        // boundary integral für div-free cond
        mfem::LinearForm lform_un(&CG);
        lform_un.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(u_0_coeff));
        lform_un.Assemble();
        // lform_un *= Re_inv;
        // lform_un *= -1.;
        
        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        // std::cout << "size:"<<size_1 << "\n";

        // initialize solution vectors
        mfem::Vector x(size_1);
        x.SetVector(u,0);
        x.SetVector(z,u.Size());
        x.SetVector(p,u.Size()+z.Size());

        // helper dofs
        mfem::Array<int> u_dofs (u.Size());
        mfem::Array<int> z_dofs (z.Size());
        mfem::Array<int> p_dofs (p.Size());
        std::iota(&u_dofs[0], &u_dofs[u.Size()], 0);
        std::iota(&z_dofs[0], &z_dofs[z.Size()], u.Size());
        std::iota(&p_dofs[0], &p_dofs[p.Size()], u.Size()+z.Size());

        // Matrix M
        mfem::BilinearForm blf_M(&ND);
        blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_M.Assemble();
        blf_M.Finalize();
        mfem::SparseMatrix M_n(blf_M.SpMat());
        mfem::SparseMatrix M_dt;
        M_dt = M_n;
        M_dt *= 1/dt;
        M_n *= -1.;
        M_dt.Finalize();
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
        DT_n = Transpose(D);
        *DT_n *= -1.;
        D.Finalize();
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



        




        // matrix E1_left
        int rows_E1 = ess_dof1.Size();
        mfem::SparseMatrix E1_left (rows_E1, u.Size());
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            E1_left.Set(i, ND_ess_tdof[i], 1.);
        }
        E1_left.Finalize();

        // matrix E1_cent
        mfem::SparseMatrix E1_cent (rows_E1, z.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            E1_cent.Set(i + ND_ess_tdof.Size(), RT_ess_tdof_0[i], 1.);
        }
        E1_cent.Finalize();
        // mfem::DenseMatrix* dense = E1_cent.ToDenseMatrix();
        // dense->PrintMatlab(std::cout);

        // matrix E1_right
        mfem::SparseMatrix E1_right (rows_E1, p.Size());
        E1_right = 0.;
        E1_right.Finalize();
        
        // vector e1
        mfem::Vector e1(ess_dof1.Size());
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            e1[i] = u[ND_ess_tdof[i]];
        }
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            e1[i + ND_ess_tdof.Size()] = z[RT_ess_tdof_0[i]];
        }


        
        





        // initialize system matrices
        mfem::Array<int> offsets_1 (4);
        offsets_1[0] = 0;
        offsets_1[1] = u.Size();
        offsets_1[2] = z.Size();
        offsets_1[3] = p.Size();
        offsets_1.PartialSum(); // exclusive scan

        mfem::Array<int> offsets_1_rows (5); // NEU
        offsets_1_rows[0] = 0;
        offsets_1_rows[1] = u.Size();
        offsets_1_rows[2] = z.Size();
        offsets_1_rows[3] = p.Size();
        offsets_1_rows[4] = ess_dof1.Size(); //NEU
        offsets_1_rows.PartialSum();

        mfem::BlockOperator A1(offsets_1_rows, offsets_1);

        // initialize rhs
        mfem::Vector b1(size_1 + ess_dof1.Size());
        mfem::Vector b1sub(u.Size());

        
        

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w_exact); // decoupling
        mfem::ConstantCoefficient two_over_dt(2.0/dt);
        blf_MR_eul.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_MR_eul.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_MR_eul.Assemble();
        blf_MR_eul.Finalize();
        mfem::SparseMatrix MR_eul(blf_MR_eul.SpMat());
        MR_eul.Finalize();
        
        
        // CT for eulerstep
        mfem::SparseMatrix CT_eul = CT_Re;
        CT_eul *= 2;
        CT_eul.Finalize();
        
        // assemble and solve system
        A1.SetBlock(0,0, &MR_eul);
        A1.SetBlock(0,1, &CT_eul);
        A1.SetBlock(0,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &N_n);
        A1.SetBlock(2,0, GT);
        A1.SetBlock(3,0, &E1_left); //TODO
        A1.SetBlock(3,1, &E1_cent);
        A1.SetBlock(3,2, &E1_right);

        
        // update b1, b2 for eulerstep
        b1 = 0.0;
        b1sub = 0.0;
        M_dt.AddMult(u,b1sub,2);
        b1.AddSubVector(f1,0);
        b1.AddSubVector(b1sub,0);
        b1.AddSubVector(e1, size_1); // NEU
        b1.AddSubVector(lform_zxn, 0); // NEU
        b1.AddSubVector(lform_un, u.Size() + z.Size());

        // transpose here: // TODO
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // solve 
        double tol = 1e-12;
        int iter = 1000000;  
        mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);

        



        ////////////////////////////////////////////////////////////////////
        // PRIMAL FIELD
        ////////////////////////////////////////////////////////////////////

        for (int i=0; i<1; i++) { // time


            // update R1
            mfem::MixedBilinearForm blf_R1(&ND,&ND);
            mfem::VectorGridFunctionCoefficient w_gfcoeff(&w_exact); // decoupling
            blf_R1.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_R1.Assemble();
            blf_R1.Finalize();
            mfem::SparseMatrix R1(blf_R1.SpMat());
            R1 *= 1./2.;
            R1.Finalize();


            // update MR
            mfem::MixedBilinearForm blf_MR(&ND,&ND); 
            blf_MR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_MR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_MR.Assemble();
            blf_MR.Finalize();
            mfem::SparseMatrix MR(blf_MR.SpMat());
            MR *= 1./2.;
            MR.Finalize();


            // update A1
            A1.SetBlock(0,0, &MR);
            A1.SetBlock(0,1, &CT_Re);
            A1.SetBlock(0,2, &G);
            A1.SetBlock(1,0, &C);
            A1.SetBlock(1,1, &N_n);
            A1.SetBlock(2,0, GT);

            A1.SetBlock(3,0, &E1_left); //NEU
            A1.SetBlock(3,1, &E1_cent);
            A1.SetBlock(3,2, &E1_right);


            // update b1
            b1 = 0.0;
            b1sub = 0.0;
            M_dt.AddMult(u,b1sub);
            R1.AddMult(u,b1sub,-1);
            CT_Re.AddMult(z,b1sub,-1);
            b1.AddSubVector(b1sub,0);
            b1.AddSubVector(f1,0);
            b1.AddSubVector(e1, size_1);
            b1.AddSubVector(lform_zxn, 0); // NEU
            b1.AddSubVector(lform_un, u.Size() + z.Size());

        
            //Transposition
            mfem::TransposeOperator AT1 (&A1);
            mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
            mfem::Vector ATb1 (size_1);
            A1.MultTranspose(b1,ATb1);

            // solve 
            mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);
            x.GetSubVector(u_dofs, u);
            x.GetSubVector(z_dofs, z);
            x.GetSubVector(p_dofs, p);

        } // time






        // convergence error
        double err_L2_u = u.ComputeL2Error(u_0_coeff);
        // std::cout << "L2err of u = "<< err_L2_u<<"\n";
        std::cout <<  err_L2_u<<"\n";

        // write to file
        file << std::setprecision(15) << std::fixed << std::pow(1/2.,ref_step) << ","   
        << err_L2_u <<  "\n";

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        std::cout << "runtime = " << duration.count() << "ms" << std::endl;

        // visuals
        // mfem::GridFunction u_ex(&ND);
        // u_ex.ProjectCoefficient(u_0_coeff);

        // char vishost[] = "localhost";
        // int  visport   = 19916;
        // mfem::socketstream u_sock(vishost, visport);
        // u_sock.precision(8);
        // u_sock << "solution\n" << mesh << u << "window_title 'u in hcurl'" << std::endl;

        // mfem::socketstream u_ex_sock(vishost, visport);
        // u_ex_sock.precision(8);
        // u_ex_sock << "solution\n" << mesh << u_ex << "window_title 'u_ex'" << std::endl;

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