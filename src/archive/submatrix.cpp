#include "mfem.hpp"
#include <fstream>
#include <iostream>
// using namespace std;
using namespace mfem;


// alle submatritzen M,N,C,D,G, sofort (und NICHT transponiert) in a1 und a2 eingebauen


void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix, int rowoffset, int coloffset) {
    for (int r = 0; r < submatrix.NumRows(); r++) {
        mfem::Array<int> cols;
        mfem::Vector srow;
        submatrix.GetRow(r, cols, srow);
        for (int c = 0; c < submatrix.NumCols(); c++) {
            matrix.Add(rowoffset + r, coloffset + cols[c], srow[c]);
        }
        cols.DeleteAll();
    }
}


int main(int argc, char *argv[]) {
    
    std::cout << "---------------launch MEHC---------------\n";
    
    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // FE spaces
    int order = 1;
    FiniteElementCollection *fec_CG, *fec_ND, *fec_RT, *fec_DG, *fec_ND0, *fec_RT0;
    fec_CG = new H1_FECollection(order, dim); // CG: finite subspace of H1
    fec_ND = new ND_FECollection(order, dim);
    fec_RT = new RT_FECollection(order, dim);
    fec_DG = new L2_FECollection(order, dim); // DG: finite subspace of L2
    fec_ND0 = new ND_FECollection(order, dim);
    fec_RT0 = new RT_FECollection(order, dim);
    FiniteElementSpace CG(&mesh, fec_CG);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace DG(&mesh, fec_DG);
    FiniteElementSpace ND0(&mesh, fec_ND0);
    FiniteElementSpace RT0(&mesh, fec_RT0);
    
    // boundary conditions
    mfem::Array<int> CG_etdof, ND_etdof, RT_etdof, DG_etdof, ND0_etdof, RT0_etdof;
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    CG.GetEssentialTrueDofs(ess_bdr, CG_etdof);
    ND.GetEssentialTrueDofs(ess_bdr, ND_etdof);
    RT.GetEssentialTrueDofs(ess_bdr, RT_etdof);
    DG.GetEssentialTrueDofs(ess_bdr, DG_etdof);
    ND0.GetEssentialTrueDofs(ess_bdr, ND0_etdof);
    RT0.GetEssentialTrueDofs(ess_bdr, RT0_etdof);
    std::cout << "progress3" << "\n";

    // Matrix M and -M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ND_etdof,M);
    M.Finalize();
    mfem::SparseMatrix Mn = M;
    Mn *= -1;

    // Matrix N and -N
    mfem::BilinearForm blf_N(&RT);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(RT_etdof,N);
    N.Finalize();
    mfem::SparseMatrix Nn = N;
    Nn *= -1;
    
    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    C.Finalize();
    mfem::SparseMatrix *CT = Transpose(C);

    // Matrix D and DT
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    mfem::SparseMatrix D;
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_D.Assemble();
    blf_D.FormRectangularSystemMatrix(RT_etdof,DG_etdof,D);
    D.Finalize();
    mfem::SparseMatrix *DT = Transpose(D);
    mfem::SparseMatrix Dn = D;
    Dn *= -1;
    mfem::SparseMatrix *DTn = Transpose(Dn);

    // Matrix G and GT
    mfem::MixedBilinearForm blf_G(&CG, &ND);
    mfem::SparseMatrix G;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(ND_etdof,CG_etdof,G);
    G.Finalize();
    mfem::SparseMatrix *GT = Transpose(G);

    // nr of unkonwns
    int size_p = M.NumCols() + CT->NumCols() + G.NumCols();
    int size_d = N.NumCols() + C.NumCols() + DT->NumCols();

    // unkowns and gridfunctions
    mfem::GridFunction u(&ND);
    mfem::GridFunction w(&RT);
    mfem::GridFunction p(&DG);
    mfem::GridFunction v(&ND0); // dual velocity
    mfem::GridFunction z(&RT0); // dual vorticity (z...zeta)
    mfem::GridFunction q(&CG);  // dual pressure

    // indices of unkowns in solution vectors x, y
    mfem::Array<int> u_dofs;
    mfem::Array<int> w_dofs;
    mfem::Array<int> p_dofs;
    mfem::Array<int> v_dofs;
    mfem::Array<int> z_dofs;
    mfem::Array<int> q_dofs;
    for (int k = 0; k < M.NumCols(); ++k)                         { u_dofs.Append(k); }
    for (int k = M.NumCols(); k < M.NumCols()+CT->NumCols(); ++k) { w_dofs.Append(k); }
    for (int k = M.NumCols()+CT->NumCols(); k < size_p; ++k)      { p_dofs.Append(k); }
    for (int k = 0; k < N.NumCols(); ++k)                         { v_dofs.Append(k); }
    for (int k = N.NumCols(); k < N.NumCols()+C.NumCols(); ++k)   { z_dofs.Append(k); }
    for (int k = N.NumCols()+C.NumCols(); k < size_d; ++k)        { q_dofs.Append(k); }

    // solution vectors x and y
    mfem::Vector x(size_p);
    mfem::Vector y(size_d);
    x.GetSubVector(u_dofs, u);
    x.GetSubVector(w_dofs, w);
    x.GetSubVector(p_dofs, p);
    y.GetSubVector(v_dofs, v);
    y.GetSubVector(z_dofs, z);
    y.GetSubVector(q_dofs, q);

    // Matrix Rp
    // TODO: Rp and Rd in 2D
    mfem::BilinearForm blf_Rp(&ND);
    mfem::SparseMatrix Rp;
    mfem::VectorGridFunctionCoefficient w_gfcoeff(&w);
    blf_Rp.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(w_gfcoeff));
    blf_Rp.Assemble();
    blf_Rp.FormSystemMatrix(ND_etdof,Rp);

    // Matrix Rd
    mfem::BilinearForm blf_Rd(&RT0);
    mfem::SparseMatrix Rd;
    mfem::VectorGridFunctionCoefficient z_gfcoeff(&z);
    blf_Rd.AddDomainIntegrator(new mfem::MixedCrossProductIntegrator(z_gfcoeff));
    blf_Rd.Assemble();
    blf_Rd.FormSystemMatrix(RT0_etdof,Rd);


    // std::cout << "M:  " <<M.NumRows()  <<" "<< M.NumCols() << "\n"; 
    // std::cout << "CT: " <<CT->NumRows() <<" "<< CT->NumCols() << "\n"; 
    // std::cout << "G:  " <<G.NumRows() <<" "<<  G.NumCols() << "\n"; 
    // std::cout << "C:  " <<C.NumRows() <<" "<<  C.NumCols() << "\n"; 
    // std::cout << "Nn: " <<Nn.NumRows() <<" "<<  Nn.NumCols() << "\n"; 
    // std::cout << "GT: " <<GT->NumRows() <<" "<<  GT->NumCols() << "\n"; 
    // std::cout << "progress2" << "\n";
    // std::cout << "N:  " <<N.NumRows()  <<" "<< N.NumCols() << "\n"; 
    // std::cout << "C:  " <<C.NumRows() <<" "<<  C.NumCols() << "\n"; 
    // std::cout << "DTn:"<<DTn->NumRows()<<" "<< DTn->NumCols() << "\n"; 
    // std::cout << "CT: " <<CT->NumRows() <<" "<< CT->NumCols() << "\n"; 
    // std::cout << "Mn: " <<Mn.NumRows() <<" "<<  Mn.NumCols() << "\n"; 
    // std::cout << "D:  " <<D.NumRows()  <<" "<< D.NumCols() << "\n"; 
    std::cout << "progress2" << "\n";

    // primal: A1*x=b1
    // [M+Rd   C^T    G] [u]   [(M-Rd)*u - C^T*z + f]
    // [C      -N      ] [w] = [          0         ]
    // [G^T            ] [p]   [          0         ]
    //
    // dual: A2*y=b2, use RT0 and ND0 for N, Rd, C, DTn, CT, Mn and D
    // [N+Rd   C   -D^T] [v]   [(N-Rd)*u - C*w + f]
    // [C^T    -M    0 ] [z] = [         0        ]
    // [D      0     0 ] [q]   [         0        ]

    mfem::SparseMatrix A1(size_p);
    mfem::SparseMatrix A2(size_d);
    AddSubmatrix(M,   A1, 0, 0); // submatrix, matrix, rowoffset, coloffset
    AddSubmatrix(*CT, A1, 0, M.NumCols());
    AddSubmatrix(G,   A1, 0, M.NumCols() + CT->NumCols());
    AddSubmatrix(C,   A1, M.NumRows(), 0);
    AddSubmatrix(*GT, A1, M.NumRows() + C.NumRows(), 0);
    AddSubmatrix(Nn,  A1, M.NumRows(), M.NumCols());
    A1.Finalize();
    AddSubmatrix(N,    A2, 0, 0);
    AddSubmatrix(C,    A2, 0, N.NumCols());
    AddSubmatrix(*DTn, A2, 0, N.NumCols() + C.NumCols());
    AddSubmatrix(*CT,  A2, N.NumRows(), 0);
    AddSubmatrix(Mn,   A2, N.NumCols(), CT->NumRows());
    AddSubmatrix(D,    A2, N.NumRows() + CT->NumRows(), 0);
    A2.Finalize();
    std::cout << "progress1" << "\n";

    // some tests with 
    
    std::cout << M.Elem(0,0) << " " << A1.Elem(0,0) << "\n"; 

    mfem::Vector vec(size_p); vec = 0.0;
    for (int i = 0; i< size_p ; i++){x(i) = i; }
    for (int i = 0; i< size_p ; i++){
        for (int j = 0; j< size_p ; j++){
            std::cout << A1(i,j) << " ";
        }
        std::cout << "\n";
    }
    // A1.Mult (x,vec);
    // for (int i = 0; i< size_p ; i++){
    //     std::cout << x(i) << " "<<i<< "\n";
    // }
    // std::cout << "-----\n";
    // for (int i = 0; i< size_p ; i++){
    //     std::cout << vec(i) << " "<<i<< "\n";
    // }
    // std::cout << A1.NumCols() << " "<< A1.NumRows();


    delete fec_CG;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;
    delete fec_ND0;
    delete fec_RT0;

    std::cout << "---------------finish MEHC---------------\n";
}


