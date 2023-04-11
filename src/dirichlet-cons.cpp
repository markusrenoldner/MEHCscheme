#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "mfem.hpp"




// MEHC scheme for dirichlet problem
// essential BC at Hdiv and Hcurl of dual system only


// use wouters trick for the BC


// conservation tests



struct Parameters {
    // double Re_inv = 1/100.; // = 1/Re 
    double Re_inv = 0; // = 1/Re 
    // double Re_inv = 1.; // = 1/Re 
    double dt     = 0.05;
    double tmax   = 10*dt;
    int ref_steps = 0;
    int init_ref  = 2;
    int order     = 1;
    double tol    = 1e-16;
    std::string outputfile = "out/rawdata/dirichlet-cons-invisc.txt";
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
        const char *mesh_file = param.mesh_file;
        mfem::Mesh mesh(mesh_file, 1, 1); 
        int dim = mesh.Dimension(); 
        int l;
        // dt *= 0.5; // TODO
        for (l = 0; l<init_ref+ref_step; l++) {
            mesh.UniformRefinement();
        } 
        std::cout << "----------ref: " << ref_step << "----------\n";

        // FE spaces: DG subset L2, ND subset Hcurl, RT subset Hdiv, CG subset H1
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

        // unkowns and gridfunctions
        mfem::GridFunction u(&ND); //u = 4.3;
        mfem::GridFunction z(&RT); //z = 5.3;
        mfem::GridFunction p(&CG); p=0.; //p = 6.3;
        mfem::GridFunction v(&RT); //v = 3.;
        mfem::GridFunction w(&ND); //w = 3.; 
        mfem::GridFunction q(&DG); q=0.; //q = 9.3;
        mfem::GridFunction u_ex(&ND); //u = 4.3;

        // initial condition
        mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
        mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0); 
        u.ProjectCoefficient(u_0_coeff);
        v.ProjectCoefficient(u_0_coeff);
        z.ProjectCoefficient(w_0_coeff);
        w.ProjectCoefficient(w_0_coeff);
        u_ex.ProjectCoefficient(u_0_coeff);
    
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

        // linearform for forcing term
        // mfem::VectorFunctionCoefficient f_coeff(dim, f);
        // mfem::LinearForm f1(&ND);
        // mfem::LinearForm f2(&RT);
        // f1.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        // f2.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(f_coeff)); //=(f,v)
        // f1.Assemble();
        // f2.Assemble();

        // boundary integral for primal reynolds term
        mfem::LinearForm lform_zxn(&ND);
        lform_zxn.AddBoundaryIntegrator(new mfem::VectorFEBoundaryTangentLFIntegrator(w_0_coeff)); // !!!
        lform_zxn.Assemble();
        lform_zxn *= -1.*Re_inv; // minus!

        // boundary integral für div-free cond
        mfem::LinearForm lform_un(&CG);
        lform_un.AddBoundaryIntegrator(new mfem::BoundaryNormalLFIntegrator(u_0_coeff));
        lform_un.Assemble();

        // system size
        int size_1 = u.Size() + z.Size() + p.Size();
        int size_2 = v.Size() + w.Size() + q.Size();
        
        // initialize solution vectors
        mfem::Vector x(size_1);
        mfem::Vector y(size_2);
        x.SetVector(u,0);
        x.SetVector(z,u.Size());
        x.SetVector(p,u.Size()+z.Size());
        y.SetVector(v,0);
        y.SetVector(w,v.Size());
        y.SetVector(q,v.Size()+w.Size());
        std::cout <<"size ND/RT: " <<u.Size()<<"/"<<v.Size() << "\n";

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

        // TODO : enforce ess dofs hardcore
        // matrix E2_left
        int rows_E2 = ess_dof2.Size();
        mfem::SparseMatrix E2_left (rows_E2, v.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            E2_left.Set(i, RT_ess_tdof[i], 1.);
        }
        E2_left.Finalize();

        // matrix E2_cent
        mfem::SparseMatrix E2_cent (rows_E2, w.Size());
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            E2_cent.Set(i + RT_ess_tdof.Size(), ND_ess_tdof_0[i], 1.);
        }
        E2_cent.Finalize();

        // vector e2
        mfem::Vector e2(ess_dof2.Size());
        for (int i=0; i<RT_ess_tdof.Size(); i++) {
            e2[i] = v[RT_ess_tdof[i]];
        }
        for (int i=0; i<ND_ess_tdof.Size(); i++) {
            e2[i + RT_ess_tdof.Size()] = w[ND_ess_tdof_0[i]];
        }

        // initialize system matrices
        mfem::Array<int> offsets_1 (4);
        offsets_1[0] = 0;
        offsets_1[1] = u.Size();
        offsets_1[2] = z.Size();
        offsets_1[3] = p.Size();
        offsets_1.PartialSum(); // exclusive scan
        mfem::BlockOperator A1(offsets_1);
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

        // initialize rhs
        mfem::Vector b1(size_1);
        mfem::Vector b1sub(u.Size());
        mfem::Vector b2(size_2 + ess_dof2.Size()); 
        mfem::Vector b2sub(v.Size());

        ////////////////////////////////////////////////////////////////////
        // EULERSTEP: code up to the loop computes euler step for primal sys
        ////////////////////////////////////////////////////////////////////

        // Matrix MR_eul for eulerstep
        mfem::MixedBilinearForm blf_MR_eul(&ND,&ND); 
        mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
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

        // update b1, b2 for eulerstep
        b1 = 0.0;
        b1sub = 0.0;
        M_dt.AddMult(u,b1sub,2);
        // b1.AddSubVector(f1,0);
        b1.AddSubVector(b1sub,0);
        b1.AddSubVector(lform_zxn, 0);
        b1.AddSubVector(lform_un, u.Size() + z.Size());

        // transpose here:
        mfem::TransposeOperator AT1 (&A1);
        mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
        mfem::Vector ATb1 (size_1);
        A1.MultTranspose(b1,ATb1);

        // solve 
        int iter = 100000000;
        mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol);

        // extract solution values u,z,p from eulerstep
        x.GetSubVector(u_dofs, u);
        x.GetSubVector(z_dofs, z);
        x.GetSubVector(p_dofs, p);

        // time loop
        double t;
        for (t = dt ; t < tmax+dt ; t+=dt) {
        // for (t = dt ; t < 2*dt ; t+=dt) {
        
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
            mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
            blf_R2.AddDomainIntegrator(
                new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_R2.Assemble();
            blf_R2.Finalize();
            mfem::SparseMatrix R2(blf_R2.SpMat());
            R2 *= 1./2.;
            R2.Finalize();

            // update NR
            mfem::MixedBilinearForm blf_NR(&RT,&RT); 
            blf_NR.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(two_over_dt)); //=(u,v)
            blf_NR.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff)); //=(wxu,v)
            blf_NR.Assemble();
            blf_NR.Finalize();
            mfem::SparseMatrix NR(blf_NR.SpMat());
            NR *= 1./2.;
            NR.Finalize();

            // update A2
            A2.SetBlock(0,0, &NR);
            A2.SetBlock(0,1, &C_Re);
            A2.SetBlock(0,2, DT_n);
            A2.SetBlock(1,0, CT);
            A2.SetBlock(1,1, &M_n);
            A2.SetBlock(2,0, &D);
            A2.SetBlock(3,0, &E2_left);
            A2.SetBlock(3,1, &E2_cent);
            
            // update b2
            b2 = 0.0;
            b2sub = 0.0;
            N_dt.AddMult(v,b2sub);
            R2.AddMult(v,b2sub,-1);
            C_Re.AddMult(w,b2sub,-1);
            // b2.AddSubVector(f2,0); 
            b2.AddSubVector(b2sub,0);
            b2.AddSubVector(e2, size_2);

            // TODO
            // remove unnecessary equations from matrix corresponding to essdofs
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

            // transpose here:
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
            mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
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

            // update b1
            b1 = 0.0;
            b1sub = 0.0;
            M_dt.AddMult(u,b1sub);
            R1.AddMult(u,b1sub,-1);
            CT_Re.AddMult(z,b1sub,-1);
            b1.AddSubVector(b1sub,0);
            // b1.AddSubVector(f1,0);
            b1.AddSubVector(lform_zxn, 0);
            b1.AddSubVector(lform_un, u.Size() + z.Size());
            //Transposition
            mfem::TransposeOperator AT1 (&A1);
            mfem::ProductOperator ATA1 (&AT1,&A1,false,false);
            mfem::Vector ATb1 (size_1);
            A1.MultTranspose(b1,ATb1);
            // ATA1.FormLinearSystem(ess_dof1, x, ATb1, A1_BC, X, B1);

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
            mass_vec1 -= lform_un;
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

            // diff
            mfem::Vector v_diff (v.Size()); 
            v_diff= 0.;
            v_diff.Add(1,v);
            v_diff.Add(-1.,v_old);
            mfem::Vector u_diff (u.Size()); 
            u_diff= 0.;
            u_diff.Add(1,u);
            u_diff.Add(-1.,u_old);

            // primal momentum equation 
            // mfem::Vector result (u.Size());
            // result = 0.;
            // MR.AddMult(u,result);
            // CT_Re.AddMult(z,result);
            // G.AddMult(p, result);
            // result.Add(-1, b1sub);
            // result.Add(-1, lform_zxn);
            // std::cout << result.Normlinf()<<"\n";
            // double result = 0.;
            // result = MR.InnerProduct(u, u_avg);
            // result += CT_Re.InnerProduct(z, u_avg);
            // result += G.InnerProduct(p, u_avg);
            // for (int i=0; i<u.Size(); i++) {
            //     result += b1sub[i] *     u_avg[i];
            //     result += lform_zxn[i] * u_avg[i];
            // }
            // std::cout << result <<"\n";

            // check prim mom equ
            // std::cout 
            // << MR.InnerProduct(u, u_avg) << ", "
            // << CT_Re.InnerProduct(z, u_avg) << ", "
            // << G.InnerProduct(p, u_avg) << ", "
            // << lform_zxn.Norml2() << ", "
            // << G.InnerProduct(p, u_avg)  +  lform_zxn.Norml2() << ", "
            // << M_n.InnerProduct(u_diff, u_avg)  << ", "
            // << G.InnerProduct(u,p) <<", "
            // << GT->InnerProduct(p,u) <<", "
            // << CT_Re.InnerProduct(z,u) <<", "
            // << lform_un.Norml2() << ", "
            // << (K1-K1_old)/dt  << ", "
            // << 2*Re_inv*E2  << "\n";
            // << mass_vec1.Norml2() << ","
            // << (K2-K2_old)/dt  << "\n";

            // << mass_vec2.Norml2() << "\n";
            // <<K1-K1_old<<K2-K2_old<< K1 <<","<<K1_old<< K2 <<","<<K1_old<< ","<<"\n";
            // << (K1-K1_old)/dt  << "," 
            // << (K2-K2_old)/dt  << "," //<< K2 <<","<< K2_old <<","
            // << 2*Re_inv*E1 << "," << (K2-K2_old)/dt - 2*Re_inv*E1 << "\n";

            // mfem::Vector v_diff (v.Size()); v_diff= 0.;
            // v_diff.Add(1,v);
            // v_diff.Add(-1.,v_old);

            std::cout 
            // << N_n.InnerProduct(v_diff, v_avg) << ","
            // << 2.*C_Re.InnerProduct(w_avg, v_avg) <<","
            // << Re_inv*E1*2. << ","
            // << R2.InnerProduct(z, v_avg) << ","
            // << mass_vec2.Norml2() <<"\n";
            // << u.Normlinf() << ","
            // << K1 << ","
            // << K1_old << ","
            // << 2*Re_inv*E2 << ","
            // << (K1-K1_old)/dt<< ","
            // << 2*Re_inv*E1 << ","
            // << (K2-K2_old)/dt<< "\n";
            << (H1-H1_old)/dt - D << ","
            << (H2-H2_old)/dt - D << "\n";
            // << 1/2*M_dt.InnerProduct(u_diff, u_avg) << ","
            // << (K1-K1_old)/dt - 2*Re_inv*E2 << ","
            // << (K2-K2_old)/dt - 2*Re_inv*E1 << "\n";
            // std::cout << u.Normlinf() << "\n";
            
            




                   
            // write to file
            file << std::setprecision(15) << std::fixed << t << ","   
            << mass_vec1.Norml2() << ","
            << mass_vec2.Norml2() << ","
            << (K1-K1_old)/dt - 2*Re_inv*E2 << ","
            << (K2-K2_old)/dt - 2*Re_inv*E1 << ","
            << (H1-H1_old)/dt - D << ","
            << (H2-H2_old)/dt - D << "\n";
            // << (K1-K1_old)/dt  << ","
            // << (K2-K2_old)/dt  << ","
            // << (H1-H1_old)/dt  << ","
            // << (H2-H2_old)/dt  << "\n";

        } // time 
        std::cout << "visual \n";
        
        // TODO
        // double err_L2_u = u.ComputeL2Error(u_0_coeff);
        // std::cout << err_L2_u << "\n";
        char vishost[] = "localhost";
        int  visport   = 19916;

        mfem::socketstream u_sock(vishost, visport);
        u_sock.precision(8);
        u_sock << "solution\n" << mesh << u << "window_title 'u in hcurl'" << std::endl;
        
        mfem::socketstream u_ex_sock(vishost, visport);
        u_ex_sock.precision(8);
        u_ex_sock << "solution\n" << mesh << u_ex << "window_title 'u_ex'" << std::endl;

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

// cos^4 solution with zero boundary
void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
    double pi = 3.14159265358979323846;
    double C = 10;
    double R = 1/2.*std::sqrt(2*pi/C); // radius where u,w vanish
    double X = x(0)-0.5;
    double Y = x(1)-0.5;
    double Z = x(2)-0.5;
    
    double cos = std::cos(C*(X*X+Y*Y+Z*Z));
    double cos4 = cos*cos*cos*cos;

    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = Y * cos4;
        returnvalue(1) = -X * cos4;
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
    double cos3 = cos*cos*cos;
    double cos4 = cos*cos*cos*cos;
    double sin = std::sin(C*(X*X+Y*Y+Z*Z));
    double sin2 = sin*sin;

    if (X*X + Y*Y + Z*Z < R*R) {
        returnvalue(0) = -8*C*X*Z *sin2 * cos3;
        returnvalue(1) = -8*C*Y*Z *sin2 * cos3;
        returnvalue(2) = 8*C*Y*Y * sin2 * cos3 - cos4;
    }
    else {
        returnvalue(0) = 0;
        returnvalue(1) = 0;
        returnvalue(2) = 0;
    }
}






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


// void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 

//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     // returnvalue(0) = std::sin(Y);
//     // returnvalue(1) = std::sin(Y);
//     returnvalue(0) = 1.;
//     returnvalue(1) = 0.;
//     returnvalue(2) = 0.;
// }
// void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     // returnvalue(0) = -std::cos(Z);
//     // returnvalue(1) = 0.;
//     // returnvalue(2) = -std::cos(Y);
//     returnvalue(0) = 0.;
//     returnvalue(1) = 0.;
//     returnvalue(2) = 0.;
// }

void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
    returnvalue(0) = 0.;
    returnvalue(1) = 0.;
    returnvalue(2) = 0.;
}



/////////////////////////
// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
//     Parameters param;
//     double Re_inv = param.Re_inv; // = 1/Re 
    
//     double X = x(0)-0.5;
//     double Y = x(1)-0.5;
//     double Z = x(2)-0.5;

//     returnvalue(0) = std::sin(Y)*Re_inv + std::cos(Y)*std::sin(Z);
//     returnvalue(1) = -std::cos(Y)*std::sin(Y) + std::sin(Z)*Re_inv;
//     returnvalue(2) = - std::cos(Z)*std::sin(Z);
// }



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




////////////////////


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

// void u_0(const mfem::Vector &x, mfem::Vector &returnvalue) {
   
//     double pi = 3.14159265358979323846;
//     returnvalue(0) =     std::cos(x(0)*pi)*std::sin(x(1)*pi);
//     returnvalue(1) = -1* std::sin(x(0)*pi)*std::cos(x(1)*pi);
//     returnvalue(2) = 0;
// }

// void w_0(const mfem::Vector &x, mfem::Vector &returnvalue) { 
   
//     double pi = 3.14159265358979323846;
//     returnvalue(0) = 0;
//     returnvalue(1) = 0;
//     returnvalue(2) = -2*pi* std::cos(x(0)*pi) * std::cos(x(1)*pi);
// }

// void f(const mfem::Vector &x, mfem::Vector &returnvalue) { 
//     returnvalue(0) = 0.;
//     returnvalue(1) = 0.;
//     returnvalue(2) = 0.;
// }

