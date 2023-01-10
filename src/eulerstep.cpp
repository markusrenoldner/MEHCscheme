#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>



void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix,
                  int rowoffset, int coloffset);
void PrintVector(mfem::Vector vec, int stride=1);
void PrintVector2(mfem::Vector vec, int stride=1);
void PrintVector3(mfem::Vector vec, int stride=1, 
                  int start=0, int stop=0, int prec=3);
void PrintMatrix(mfem::Matrix &mat, int prec=2);
void u_0(const mfem::Vector &x, mfem::Vector &v);
void w_0(const mfem::Vector &x, mfem::Vector &w);




int main(int argc, char *argv[]) {

    std::cout << "---------------launch MEHC---------------\n";

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/periodic-cube.mesh"; 
    mfem::Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();
    // for (int l = 0; l < 2; l++) {mesh.UniformRefinement();}

    // simulation parameters
    double Re_inv = 0; // = 1/Re
    double dt = 1;
    int timesteps = 1;

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
    mfem::GridFunction z(&RT); // z = 5.3; 
    mfem::GridFunction p(&CG); // p = 6.3;
    mfem::GridFunction v(&RT); // v = 7.3; 
    mfem::GridFunction w(&ND); // w = 8.3;
    mfem::GridFunction q(&DG); // q = 9.3;

    // initial condition
    mfem::VectorFunctionCoefficient u_0_coeff(dim, u_0);
    mfem::VectorFunctionCoefficient w_0_coeff(dim, w_0);
    u.ProjectCoefficient(u_0_coeff);
    v.ProjectCoefficient(u_0_coeff);
    w.ProjectCoefficient(w_0_coeff);
    z.ProjectCoefficient(w_0_coeff);

    // system size
    int size_1 = u.Size() + z.Size() + p.Size();
    int size_2 = v.Size() + w.Size() + q.Size();
    std::cout << "size1: " << size_1 << "\n"<<"size2: "<<size_2<< "\n";
    std::cout<< "size u/z/p: "<<u.Size()<<"/"<<z.Size()<<"/"<<p.Size()<<"\n";
    std::cout<< "size v/w/q: "<<v.Size()<<"/"<<w.Size()<<"/"<<q.Size()<<"\n";
    
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

    // initialize system matrices
    mfem::Array<int> offsets_1 (4);
    offsets_1[0] = 0;
    offsets_1[1] = u.Size();
    offsets_1[2] = z.Size();
    offsets_1[3] = p.Size();
    offsets_1.PartialSum(); // exclusive scan
    mfem::BlockMatrix A1(offsets_1);

    // initialize rhs
    mfem::Vector b1(size_1);
    mfem::Vector b1sub(u.Size());
    mfem::Vector b2(size_2); 
    mfem::Vector b2sub(v.Size());

    ///////////////////////////////////////////////////////////////

    // M,R,CT for eulerstep
    mfem::SparseMatrix MR_0 = M_dt;
    mfem::SparseMatrix CT_0 = CT_Re;
    MR_0.Add(1,R1);
    MR_0 *= 2;
    CT_0 *= 2;
    MR_0.Finalize();
    CT_0.Finalize();

    // update A1 for eulerstep
    A1.SetBlock(0,0, &MR_0);
    A1.SetBlock(0,1, &CT_0);
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
    double tol = 1e-12;
    int iter = 10000;
    mfem::MINRES(ATA1, ATb1, x, 0, iter, tol*tol, tol*tol); 

    // extract solution values u,z,p from eulerstep
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(z_dofs, z);
    x.GetSubVector(p_dofs, p);

    ///////////////////////////////////////////////////////////////
    

    // PrintVector3(x,1,0,20);
    // PrintVector3(x,1,u.Size()+1,u.Size()+20);
    // PrintVector3(x,1,u.Size()+z.Size(),u.Size()+z.Size()+20);


    // free memory
    delete fec_CG;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;

    std::cout << "---------------finish MEHC---------------\n";

}

void u_0(const mfem::Vector &x, mfem::Vector &v) {
   
    double pi = 3.14159265358979323846;
    int dim = x.Size();

    // u0=(cos(2piz), sin(2piz), sin(2pix))
    // TODO: make periodic on [-1,1]^3
    v(0) = std::cos(2*pi*x(3));
    v(1) = std::sin(2*pi*x(3)); 
    v(2) = std::sin(2*pi*x(1));
}

void w_0(const mfem::Vector &x, mfem::Vector &w) {
   
    double pi = 3.14159265358979323846;
    int dim = x.Size();

    // w0=(-2pi cos(2piz), -2pi cos(2pix) -2pi sin(2piz), 0) 
    // TODO: make periodic on [-1,1]^3
    w(0) = -2*pi*std::cos(2*pi*x(3));
    w(1) = -2*pi*std::cos(2*pi*x(1)) -2*pi*std::sin(2*pi*x(3)); 
    w(2) = 0;
}