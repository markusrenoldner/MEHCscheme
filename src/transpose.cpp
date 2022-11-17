#include "mfem.hpp"
#include <fstream>
#include <iostream>
// using namespace std;
using namespace mfem;


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
    FiniteElementCollection *fec_H1, *fec_ND, *fec_RT, *fec_DG;
    fec_H1 = new H1_FECollection(order, dim);
    fec_ND = new ND_FECollection(order, dim);
    fec_RT = new RT_FECollection(order, dim);
    fec_DG = new L2_FECollection(order, dim);
    FiniteElementSpace H1(&mesh, fec_H1);
    FiniteElementSpace ND(&mesh, fec_ND);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace DG(&mesh, fec_DG);
    
    // boundary conditions
    mfem::Array<int> H1_etdof, ND_etdof, RT_etdof, DG_etdof; // "essential true degrees of freedom"
    mfem::Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 0;
    H1.GetEssentialTrueDofs(ess_bdr, H1_etdof);
    ND.GetEssentialTrueDofs(ess_bdr, ND_etdof);
    RT.GetEssentialTrueDofs(ess_bdr, RT_etdof);
    DG.GetEssentialTrueDofs(ess_bdr, DG_etdof);
    std::cout << "progress3" << "\n";

    // Matrix M and -M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(RT_etdof,M);
    M.Finalize();
    
    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    C.Finalize();
    mfem::SparseMatrix *CT = Transpose(C);

    // Matrix G and GT
    mfem::MixedBilinearForm blf_G(&H1, &ND);
    mfem::SparseMatrix G;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(ND_etdof,H1_etdof,G);
    G.Finalize();
    mfem::SparseMatrix *GT = Transpose(G);

    std::cout << "M:  " <<M.NumRows()  <<" "<< M.NumCols() << "\n"; 
    std::cout << "C:  " <<C.NumRows() <<" "<< C.NumCols() << "\n"; 
    std::cout << "CT: " <<CT->NumRows() <<" "<< CT->NumCols() << "\n"; 

    // primal: A1*x=b1
    // [M+Rd   C^T    G] [u]   [(M-Rd)*u - C^T*z + f]
    // [C      -N      ] [w] = [          0         ]
    // [G^T            ] [p]   [          0         ]
    //
    // dual: A2*y=b2
    // [N+Rp   C   -D^T] [v]   [(N-Rp)*u - C*w + f]
    // [C^T    -M    0 ] [z] = [         0        ]
    // [D      0     0 ] [q]   [         0        ]

    int size_p = M.NumCols() + CT->NumCols() + G.NumCols();

    mfem::SparseMatrix A1(size_p);
    AddSubmatrix(M, A1, 0, 0); // submatrix, matrix, rowoffset, coloffset
    AddSubmatrix(*CT, A1, 0, M.NumCols());
    AddSubmatrix(G,  A1, 0, M.NumCols() + CT->NumCols());
    std::cout << "progress1" << "\n";


    delete fec_H1;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;

    std::cout << "---------------finish MEHC---------------\n";
}


