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
    // double Re_inv = 0.; // = 1/Re 
    double Re_inv = 1/1600.; // = 1/Re 
    double dt     = 1/20.;
    double tmax   = 20.;
    int ref_steps = 0;
    int init_ref  = 1;
    int order     = 1;
    double tol    = 1e-14;
    // std::string outputfile = "out/rawdata/periodic-cons-Re100.txt";
    // std::string outputfile = "out/rawdata/periodic-cons-invisc.txt";
    std::string outputfile = "out/rawdata/periodic-cons-testmooslechner.txt";
    const char* mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    double t;
};

void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void u_0(const mfem::Vector &x, mfem::Vector &v);
void w_0(const mfem::Vector &x, mfem::Vector &v);


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

    // loop over refinement steps to check convergence
    for (int ref_step=0; ref_step<=ref_steps; ref_step++) {
        
        auto start = std::chrono::high_resolution_clock::now();

        // output file 
        std::string outputfile = param.outputfile;
        std::ofstream file;
        file.precision(6);
        file.open(outputfile);
        // file.open(outputfile, std::ios::app);

        // mesh
        const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh"; 
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        int l;
        for (l = 0; l<init_ref+ref_step; l++) {mesh.UniformRefinement();} 
        std::cout << "ref levels:" << l <<"\n";

        // FE spaces (CG \in H1, DG \in L2)
        mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order-1,dim);
        mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        // mfem::FiniteElementCollection *fec_CG = new mfem::H1_FECollection(order,dim);
        // mfem::FiniteElementCollection *fec_ND = new mfem::ND_FECollection(order,dim);
        // mfem::FiniteElementCollection *fec_RT = new mfem::RT_FECollection(order,dim);
        // mfem::FiniteElementCollection *fec_DG = new mfem::L2_FECollection(order,dim);
        mfem::FiniteElementSpace CG(&mesh, fec_CG);
        mfem::FiniteElementSpace ND(&mesh, fec_ND);
        mfem::FiniteElementSpace RT(&mesh, fec_RT);
        mfem::FiniteElementSpace DG(&mesh, fec_DG);
        std::cout << "test\n";

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND); // u = 4.3;
        mfem::GridFunction z(&RT); // z = 5.3;
        mfem::GridFunction p(&CG); // p = 6.3;
        mfem::GridFunction v(&RT); // v = 7.3;
        mfem::GridFunction w(&ND); // w = 8.3;
        mfem::GridFunction q(&DG); // q = 9.3;
        std::cout << "test\n";

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        u.ProjectCoefficient(u_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w.ProjectCoefficient(w_0_coeff);

        


        // the problem is, that the kinetic energy does not make sense
        // the initial value should be different (according to the analytical value)
        // the following code tries to find the error:

        // Matrix M
        // mfem::BilinearForm blf_M3(&ND);
        // blf_M3.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        // blf_M3.Assemble();
        // blf_M3.Finalize();

        // std::cout << "test\n";
        // double K_init = -1./2.*blf_M3.InnerProduct(u,u);
        // std::cout << "K1:"<<K_init << "\n";
        
        // mfem::BilinearForm blf_M1(&ND);
        // std::cout << "test1\n";
        // blf_M1.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        // std::cout << "test2\n";
        // // blf_M1.Assemble();
        // std::cout << "test3\n";
        // double K_init = -1./2.*blf_M1.InnerProduct(u,u);
        // std::cout << "K1:"<<K_init << "\n";

        
        // mfem::BilinearForm blf_N1(&RT);
        // blf_N1.AddDomainIntegrator(new mfem::VectorFEMassIntegrator()); //=(u,v)
        // blf_N1.Assemble();
        // double K_init2 = -1./2.*blf_N1.InnerProduct(v,v);
        // std::cout << "K2:"<<K_init2 << "\n";








        
        // helper vectors for old values
        mfem::Vector u_old(u.Size()); u_old = 0.;
        mfem::Vector v_old(v.Size()); v_old = 0.;
        mfem::Vector z_old(z.Size()); z_old = 0.;
        mfem::Vector w_old(w.Size()); w_old = 0.;
        mfem::Vector u_old_old(u.Size()); u_old_old = 0.;
        mfem::Vector v_old_old(v.Size()); v_old_old = 0.;
        mfem::Vector z_old_old(z.Size()); z_old_old = 0.;

        // helper vectors for average values
        mfem::Vector u_avg (u.Size()); 
        mfem::Vector z_avg (z.Size()); 
        mfem::Vector v_avg (v.Size()); 
        mfem::Vector w_avg (w.Size()); 
        mfem::Vector u_avg_old (u.Size());
        mfem::Vector v_avg_old (v.Size());
        mfem::Vector z_avg_old (z.Size());

        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        int size_2 = v.Size() + w.Size() + q.Size();
        std::cout <<"size ND/RT: " <<u.Size()<<"/"<<v.Size() << "\n";
        
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


        // another try to find the energy value:
        std::cout << blf_M.InnerProduct(u,u) * dt /8 << "\n";

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
        b1.AddSubVector(b1sub,0);

        // create symmetric system AT*A*x=AT*b for eulerstep
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // solve eulerstep
        int iter = 10000;
        mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);
            
        // time loop
        for (double t = dt ; t < tmax+dt ; t+=dt) {
            
            // update old values before computing new ones
            u_old_old = u_old;
            v_old_old = v_old;
            z_old_old = z_old;
            u_old = u;
            v_old = v;
            z_old = z;
            w_old = w;

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
            b2.AddSubVector(b2sub,0);

            // create symmetric system AT*A*x=AT*b
            mfem::TransposeOperator AT2 (&A2);
            mfem::ProductOperator ATA2 (&AT2,&A2,false,false);
            mfem::Vector ATb2 (size_2);
            A2.MultTranspose(b2,ATb2);

            // solve 
            int iter = 10000;
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
            
            ////////////////////////////////////////////////////////////////////
            // CONSERVATION
            ////////////////////////////////////////////////////////////////////

            // averaged values
            u_avg = 0.;
            u_avg.Add(0.5,u);
            u_avg.Add(0.5,u_old);
            z_avg = 0.;
            z_avg.Add(0.5,z);
            z_avg.Add(0.5,z_old);
            v_avg = 0.;
            v_avg.Add(0.5,v);
            v_avg.Add(0.5,v_old);
            w_avg = 0.;
            w_avg.Add(0.5,w);
            w_avg.Add(0.5,w_old);

            // averaged old values
            u_avg_old = 0.;
            u_avg_old.Add(0.5,u_old);
            u_avg_old.Add(0.5,u_old_old);
            v_avg_old = 0.;
            v_avg_old.Add(0.5,v_old);
            v_avg_old.Add(0.5,v_old_old);
            z_avg_old = 0.;
            z_avg_old.Add(0.5,z_old);
            z_avg_old.Add(0.5,z_old_old);

            // conservation test without dissipation rate (inviscid limit)
            mfem::Vector mass_vec1 (p.Size());
            mfem::Vector mass_vec2 (q.Size());
            GT->Mult(u,mass_vec1);
            D.Mult(v,mass_vec2);
            double K1_old = -1./2.*blf_M.InnerProduct(u_old,u_old);
            double K1 = -1./2.*blf_M.InnerProduct(u,u);
            double K2_old = -1./2.*blf_N.InnerProduct(v_old,v_old);
            double K2 = -1./2.*blf_N.InnerProduct(v,v);
            double H1_old = -1.*blf_M.InnerProduct(u_avg_old,w_old);
            double H1 = -1.*blf_M.InnerProduct(u_avg,w);
            double H2_old = -1.*blf_N.InnerProduct(v_old,z_avg_old); 
            double H2 = -1.*blf_N.InnerProduct(v,z_avg); 
        
            // dissipation rates
            double E2 = 1/2.*blf_N.InnerProduct(z_avg,z_avg);
            double E1 = 1/2.*blf_M.InnerProduct(w_avg,w_avg);
            double D = -Re_inv*C.InnerProduct(w_avg,z_old)
                    -Re_inv/2*CT->InnerProduct(z_avg,w)
                    -Re_inv/2*CT->InnerProduct(z_avg_old,w_old); 

            // write to file
            // file << std::setprecision(15) << std::fixed << t << ","   
            // << mass_vec1.Norml2() << ","
            // << mass_vec2.Norml2() << ","
            // << (K1-K1_old)/dt - 2*Re_inv*E2 << ","
            // << (K2-K2_old)/dt - 2*Re_inv*E1 << ","
            // << (H1-H1_old)/dt - D << ","
            // << (H2-H2_old)/dt - D << "\n";

            // print values
            // std::cout 
            // << mass_vec1.Norml2() << ","
            // << mass_vec2.Norml2() << ","
            // << (K1-K1_old)/dt - 2*Re_inv*E2 << ","
            // << (K1-K1_old)/dt - 2*Re_inv*E2 
            // << "\n";

            // compare dissipation rate to Xaver Mooslechners scheme
            std::cout 
            << (K1-K1_old)/dt << ","
            << 2*Re_inv*E2 
            << (K2-K2_old)/dt << ","
            << 2*Re_inv*E1 << ","
            << "\n";
            file << std::setprecision(15) << std::fixed << t << ","  
            << K1 << ","
            << K2 << ","
            << (K1-K1_old)/dt << ","
            << 2*Re_inv*E2 
            << (K2-K2_old)/dt << ","
            << 2*Re_inv*E1
            << "\n";


        } // time loop

        // free memory
        delete fec_CG;
        delete fec_ND;
        delete fec_RT;
        delete fec_DG;

        // close file
        file.close();

    } // refinement loop
}




/////////////////////////////////////////////////////////////////////
// initial condition from paper sec 5.1.1
// void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
//     double pi = 3.14159265358979323846;

//     returnvalue(0) = std::cos(pi*x.Elem(2)); 
//     returnvalue(1) = std::sin(pi*x.Elem(2));
//     returnvalue(2) = std::sin(pi*x.Elem(0));
// }

// void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double pi = 3.14159265358979323846;

//     returnvalue(0) = -pi*std::cos(pi*x(2));
//     returnvalue(1) = -pi*std::cos(pi*x(0)) -  pi*std::sin(pi*x(2)); 
//     returnvalue(2) = 0;
// }




/////////////////////////////////////////////////////////////////////
// TGV initial condition von Xaver Mooslechner
// boundary cond nicht erfüllt nach einigen time steps?
void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double X = x(0); //-0.5;
    double Y = x(1); //-0.5;
    double Z = x(2); //-0.5;

    returnvalue(0) = +1 * std::cos(X*pi) * std::sin(Y*pi) * std::sin(Z*pi);
    returnvalue(1) = -1 * std::sin(X*pi) * std::cos(Y*pi) * std::sin(Z*pi);
    returnvalue(2) = 0;
}

void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double X = x(0); //-0.5;
    double Y = x(1); //-0.5;
    double Z = x(2); //-0.5;

    returnvalue(0) = pi*std::cos(pi*Y)*std::cos(pi*Z)*std::sin(pi*X);
    returnvalue(1) = pi*std::cos(pi*X)*std::cos(pi*Z)*std::sin(pi*Y);
    returnvalue(2) = -(pi*std::cos(pi*X)*std::cos(pi*Y)*std::sin(pi*Z));
}