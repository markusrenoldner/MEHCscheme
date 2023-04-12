#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"


// MEHC scheme on periodic domain
// like described in the paper


// primal: A1*x=b1
// [M_dt+R1  CT_Re    G] [u]   [(M_dt-R1)*u - CT_Re*z  + f]
// [C        N_n      0] [z] = [             0            ]
// [GT       0        0] [p]   [             0            ]
//
// dual: A2*y=b2
// [N_dt+R2  C_Re     DT_n] [v]   [(N_dt-R2)*u - C_Re*w + f]
// [CT       M_n      0   ] [w] = [            0           ]
// [D        0        0   ] [q]   [            0           ]


struct Parameters {
    // double Re_inv = 1/100.; // = 1/Re 
    double Re_inv = 0.; // = 1/Re 
    // double Re_inv = 1.; // = 1/Re 
    double dt     = 1/20.;
    double tmax   = 3/20.;
    int ref_steps = 1;
    int init_ref  = 0;
    int order     = 1;
    double tol    = 1e-15;
    std::string outputfile = "out/rawdata/periodic-conv-invisc.txt";
    const char* mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh";
    double t;
};


void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void u_0_TGV(const mfem::Vector &x, mfem::Vector &v);
void w_0_TGV(const mfem::Vector &x, mfem::Vector &v);
void   f_TGV(const mfem::Vector &x, mfem::Vector &v); 
void u_exact_TGV(const mfem::Vector &x, mfem::Vector &returnvalue);


int main(int argc, char *argv[]) {

    // simulation parameters
    Parameters param;
    double Re_inv = param.Re_inv; 
    double dt     = param.dt;
    double tmax   = param.tmax;
    int ref_steps = param.ref_steps;
    int init_ref  = param.init_ref;
    int order     = param.order;
    double tol    = param.tol;
    std::cout <<"----------\n"<<"Re:   "<<1/Re_inv
    <<"\ndt:   "<<dt<< "\ntmax: "<<tmax<<"\n----------\n";

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        // start timer
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
        for (int l = 0; l<ref_step; l++) {mesh.UniformRefinement();} 
        // std::cout << "refinement: " << ref_step << "\n";

        // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
        // int order = 1;
        mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1,dim);
        mfem::FiniteElementSpace CG(&mesh, fec_CG);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);
        mfem::FiniteElementSpace RT(&mesh, fec_RT);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND); // u = 4.3;
        mfem::GridFunction z(&RT); // z = 5.3;
        mfem::GridFunction p(&CG); // p = 6.3;
        mfem::GridFunction v(&RT); // v = 7.3;
        mfem::GridFunction w(&ND); // w = 8.3;
        mfem::GridFunction q(&DG); // q = 9.3;
        mfem::GridFunction f1(&ND);
        mfem::GridFunction f2(&RT);

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0_TGV);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0_TGV); 
        mfem::VectorFunctionCoefficient f_coeff(dim, f_TGV);
        mfem::VectorFunctionCoefficient u_exact_coeff(dim, u_exact_TGV); 
        u.ProjectCoefficient(u_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        f1.ProjectCoefficient(f_coeff);
        f2.ProjectCoefficient(f_coeff);

        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        int size_2 = v.Size() + w.Size() + q.Size();
        // std::cout << "size1: " << size_1 << "\n"<<"size2: "<<size_2<< "\n";
        // std::cout<< "size u/z/p: "<<u.Size()<<"/"<<z.Size()<<"/"<<p.Size()<<"\n";
        // std::cout<< "size v/w/q: "<<v.Size()<<"/"<<w.Size()<<"/"<<q.Size()<<"\n"
        // <<"----------------------\n";
        
        // initialize solution vectors
        mfem::Vector x(size_1);
        mfem::Vector y(size_2);
        x.SetVector(u,0);
        x.SetVector(z,u.Size());
        x.SetVector(p,u.Size()+z.Size());
        y.SetVector(v,0);
        y.SetVector(w,v.Size());
        y.SetVector(q,v.Size()+w.Size());

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

        // boundary conditions
        mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof;

        // Matrix M
        mfem::BilinearForm blf_M(&ND);
        mfem::SparseMatrix M_dt;
        mfem::SparseMatrix M_n;
        blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_M.Assemble();
        blf_M.FormSystemMatrix(ND_etdof,M_n);
        M_dt = M_n;
        M_dt *= 1/dt;
        M_n *= -1.;
        M_dt.Finalize();
        M_n.Finalize();

        // Matrix N
        mfem::BilinearForm blf_N(&RT);
        mfem::SparseMatrix N_dt;
        mfem::SparseMatrix N_n;
        blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        blf_N.Assemble();
        blf_N.FormSystemMatrix(RT_etdof,N_n);
        N_dt = N_n;
        N_dt *= 1/dt;
        N_n *= -1.;
        N_dt.Finalize();
        N_n.Finalize();

        // Matrix C
        mfem::MixedBilinearForm blf_C(&ND, &RT);
        mfem::SparseMatrix C;
        mfem::SparseMatrix *CT;
        mfem::SparseMatrix C_Re;
        mfem::SparseMatrix CT_Re;
        blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator()); //=(curl u,v)
        blf_C.Assemble();
        blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
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
        mfem::SparseMatrix D;
        mfem::SparseMatrix *DT_n;
        blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator()); //=(div u,v)
        blf_D.Assemble();
        blf_D.FormRectangularSystemMatrix(RT_etdof,DG_etdof,D);
        DT_n = Transpose(D);
        *DT_n *= -1.;
        D.Finalize();
        DT_n->Finalize();

        // Matrix G
        mfem::MixedBilinearForm blf_G(&CG, &ND);
        mfem::SparseMatrix G;
        mfem::SparseMatrix *GT;
        blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator()); //=(grad u,v)
        blf_G.Assemble();
        blf_G.FormRectangularSystemMatrix(CG_etdof,ND_etdof,G);
        GT = Transpose(G);
        G.Finalize();
        GT->Finalize();

        // initialize system matrices
        mfem::Array<int> offsets_1 (4);
        offsets_1[0] = 0;
        offsets_1[1] = u.Size();
        offsets_1[2] = z.Size();
        offsets_1[3] = p.Size();
        offsets_1.PartialSum(); // exclusive scan
        mfem::Array<int> offsets_2 (4);
        offsets_2[0] = 0;
        offsets_2[1] = v.Size();
        offsets_2[2] = w.Size();
        offsets_2[3] = q.Size();
        offsets_2.PartialSum();
        mfem::BlockMatrix A1(offsets_1);
        mfem::BlockMatrix A2(offsets_2);

        // initialize rhs
        mfem::Vector b1(size_1);
        mfem::Vector b1sub(u.Size());
        mfem::Vector b2(size_2); 
        mfem::Vector b2sub(v.Size());

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
        mfem::SparseMatrix MR_eul;
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
        mfem::ConstantCoefficient two_over_dt(2.0/dt);
        blf_MR_eul.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
        blf_MR_eul.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
        blf_MR_eul.Assemble();
        blf_MR_eul.FormRectangularSystemMatrix(ND_etdof,ND_etdof,MR_eul);
        MR_eul.Finalize();
        
        // CT for eulerstep
        mfem::SparseMatrix CT_eul = CT_Re;
        CT_eul *= 2;
        CT_eul.Finalize();

        // update A1 for eulerstep
        A1.SetBlock(0,0, &MR_eul);
        A1.SetBlock(0,1, &CT_eul);
        A1.SetBlock(0,2, &G);
        A1.SetBlock(1,0, &C);
        A1.SetBlock(1,1, &N_n);
        A1.SetBlock(2,0, GT);

        // update b1, b2 for eulerstep
        b1 = 0.0;
        b1sub = 0.0;
        M_dt.AddMult(u,b1sub,2);
        b1.AddSubVector(f1,0);
        b1.AddSubVector(b1sub,0); 

        // create symmetric system AT*A*x=AT*b for eulerstep
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // solve eulerstep
        // double tol = 1e-15;
        int iter = 1000000;
        mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);
            
        // time loop
        for (double t = dt ; t < tmax+dt ; t+=dt) {
            // std::cout << "--- t = "<<t<<"\n";
            // std::cout << t << ",";

            ////////////////////////////////////////////////////////////////////
            // DUAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R2
            mfem::MixedBilinearForm blf_R2(&RT,&RT);
            mfem::SparseMatrix R2;
            mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
            blf_R2.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_R2.Assemble();
            blf_R2.FormRectangularSystemMatrix(RT_etdof,RT_etdof,R2);
            R2 *= 1./2.;
            R2.Finalize();

            // update NR
            mfem::MixedBilinearForm blf_NR(&RT,&RT); 
            mfem::SparseMatrix NR;
            blf_NR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_NR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_NR.Assemble();
            blf_NR.FormRectangularSystemMatrix(RT_etdof,RT_etdof,NR);
            NR *= 1./2.;
            NR.Finalize();

            // update A2
            A2.SetBlock(0,0, &NR);
            A2.SetBlock(0,1, &C_Re);
            A2.SetBlock(0,2, DT_n);
            A2.SetBlock(1,0, CT);
            A2.SetBlock(1,1, &M_n);
            A2.SetBlock(2,0, &D);

            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N_dt.AddMult(v,b2sub);
            R2.AddMult(v,b2sub,-1);
            C_Re.AddMult(w,b2sub,-1);
            b2.AddSubVector(f2,0);
            b2.AddSubVector(b2sub,0);

            // create symmetric system AT*A*x=AT*b
            mfem::TransposeOperator AT2 (&A2);
            mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
            mfem::Vector ATb2 (size_2);
            A2.MultTranspose(b2,ATb2);

            // solve 
            mfem::MINRES(ATA2, ATb2, y, 0, iter, tol*tol, tol*tol); 
            y.GetSubVector(v_dofs, v);
            y.GetSubVector(w_dofs, w);
            y.GetSubVector(q_dofs, q);

            ////////////////////////////////////////////////////////////////////
            // PRIMAL FIELD
            ////////////////////////////////////////////////////////////////////

            // update R1
            mfem::MixedBilinearForm blf_R1(&ND,&ND);
            mfem::SparseMatrix R1;
            mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
            blf_R1.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_R1.Assemble();
            blf_R1.FormRectangularSystemMatrix(ND_etdof,ND_etdof,R1);
            R1 *= 1./2.;
            R1.Finalize();

            // update MR
            mfem::MixedBilinearForm blf_MR(&ND,&ND); 
            mfem::SparseMatrix MR;
            blf_MR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_MR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff)); //=(wxu,v)
            blf_MR.Assemble();
            blf_MR.FormRectangularSystemMatrix(ND_etdof,ND_etdof,MR);
            MR *= 1./2.;
            MR.Finalize();

            // update A1
            A1.SetBlock(0,0, &MR);
            A1.SetBlock(0,1, &CT_Re);
            A1.SetBlock(0,2, &G);
            A1.SetBlock(1,0, &C);
            A1.SetBlock(1,1, &N_n);
            A1.SetBlock(2,0, GT);

            // update b1
            b1 = 0.0;
            b1sub = 0.0;
            M_dt.AddMult(u,b1sub);
            R1.AddMult(u,b1sub,-1);
            CT_Re.AddMult(z,b1sub,-1);
            b1.AddSubVector(b1sub,0);
            b1.AddSubVector(f1,0);

            // create symmetric system AT*A*x=AT*b
            mfem::TransposeOperator AT1 (&A1);
            mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
            mfem::Vector ATb1 (size_1);
            A1.MultTranspose(b1,ATb1);

            // solve 
            mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);
            x.GetSubVector(u_dofs, u);
            x.GetSubVector(z_dofs, z);
            x.GetSubVector(p_dofs, p);
            
        } // time loop

        // convergence error
        double err_L2_u = u.ComputeL2Error(u_exact_coeff);
        double err_L2_v = v.ComputeL2Error(u_exact_coeff);
        
        mfem::GridFunction v_ND (&ND);
        v_ND.ProjectGridFunction(v);
        double err_L2_diff = 0;
        for (int i=0; i<u.Size(); i++) {
            err_L2_diff += ((u(i)-v_ND(i))*(u(i)-v_ND(i)));
        }
        err_L2_diff = std::pow(err_L2_diff, 0.5);

        // write to file
        file << std::setprecision(15) << std::fixed << ref_step << ","   
        << err_L2_u << "," << err_L2_v << "," << err_L2_diff << ",\n";
        

        // runtime
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = 1000*(end - start);
        // std::cout << duration.count() << "ms" << std::endl;
        std::cout << err_L2_u << " " << err_L2_v << " "<<duration.count() << "ms"<<"\n";
        // std::cout << "---------------finish MEHC---------------\n";
    
        // free memory
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;
        delete fec_DG;

        // close file
        file.close();

    } // refinement loop
}

// easy conv test von dirichlet


// void u_0_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = std::sin(Y);
//     returnvalue(1) = std::sin(Z);
//     returnvalue(2) = 0.;
// }
// void w_0_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = -std::cos(Z);
//     returnvalue(1) = 0.;
//     returnvalue(2) = -std::cos(Y);
// }
// void f_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) { 
//     Parameters param;
//     double Re_inv = param.Re_inv; // = 1/Re 
    
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = std::sin(Y)*Re_inv + std::cos(Y)*std::sin(Z);
//     returnvalue(1) = -std::cos(Y)*std::sin(Y) + std::sin(Z)*Re_inv;
//     returnvalue(2) = - std::cos(Z)*std::sin(Z);
// }
// void u_exact_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) {

//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = std::sin(Y);
//     returnvalue(1) = std::sin(Z);
//     returnvalue(2) = 0.;
// }


// TGV - möglicherweise falsch
void u_0_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
    double pi = 3.14159265358979323846;
    returnvalue(0) =     std::cos(x(0)*pi)*std::sin(x(1)*pi);
    returnvalue(1) = -1* std::sin(x(0)*pi)*std::cos(x(1)*pi);
    returnvalue(2) = 0;

    // manuf. sol from paper sec5.1.2
    // returnvalue(0) = std::sin(x(0))*std::cos(x(1))*std::cos(x(2)); 
    // returnvalue(1) = -1*std::cos(x(0))*std::sin(x(1))*std::cos(x(2)); 
    // returnvalue(2) = 0.;
}

void w_0_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    returnvalue(0) = 0;
    returnvalue(1) = 0;
    returnvalue(2) = -2*pi* std::cos(x(0)*pi) * std::cos(x(1)*pi);

    // manuf. sol from paper sec5.1.2
    // returnvalue(0) = -1*std::cos(x(0))*std::sin(x(1))*std::sin(x(2));
    // returnvalue(1) = -1*std::cos(x(1))*std::sin(x(0))*std::sin(x(2));
    // returnvalue(2) = 2*std::cos(x(2))*std::sin(x(0))*std::sin(x(1));
}

void f_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    returnvalue(0) = 0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}

void u_exact_TGV(const mfem::Vector &x, mfem::Vector &returnvalue) {
    
    Parameters param;
    double Re_inv = param.Re_inv; 
    double tmax = param.tmax;
   
    double pi = 3.14159265358979323846;
    // double Re_inv = 1/500.; // chose Re here!
    double nu = 1*1*Re_inv; // = u*L/Re
    double F = std::exp(-2*nu*tmax);

    returnvalue(0) =     std::cos(x(0)*pi)*std::sin(x(1)*pi) * F;
    returnvalue(1) = -1* std::sin(x(0)*pi)*std::cos(x(1)*pi) * F;
    returnvalue(2) = 0;

    // manuf. sol from paper sec5.1.2
    // returnvalue(0) = std::cos(x(0)) * sin(x(1)) * std::exp(-2*nu*t);
    // returnvalue(1) = -1* std::sin(x(0)) * cos(x(1)) * std::exp(-2*nu*t);
    // returnvalue(2) = 0.; 
}