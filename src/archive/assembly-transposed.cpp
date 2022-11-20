#include "mfem.hpp"
#include <fstream>
#include <iostream>
// using namespace std;
using namespace mfem;


// alle submatritzen M,N,C,D,G, transponieren bevor sie in a1 und a2 eingebaut werden
// anscheinend ist das falsch


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

    // careful: mfem constructs transposed matrices
    // Matrix M and -M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix MT;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(RT_etdof,MT);
    MT.Finalize();
    mfem::SparseMatrix *M = Transpose(MT);
    mfem::SparseMatrix Mn = *M;
    Mn *= -1;

    // Matrix N and -N
    mfem::BilinearForm blf_N(&ND);
    mfem::SparseMatrix NT;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(ND_etdof,NT);
    NT.Finalize();
    mfem::SparseMatrix *N = Transpose(NT);
    mfem::SparseMatrix Nn = *N;
    Nn *= -1;
    
    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix CT;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,CT);
    CT.Finalize();
    mfem::SparseMatrix *C = Transpose(CT);

    // Matrix D and DT
    mfem::MixedBilinearForm blf_D(&RT, &DG);
    mfem::SparseMatrix DT;
    blf_D.AddDomainIntegrator(new mfem::MixedScalarDivergenceIntegrator());
    blf_D.Assemble();
    blf_D.FormRectangularSystemMatrix(RT_etdof,DG_etdof,DT);
    DT.Finalize();
    mfem::SparseMatrix *D = Transpose(DT);
    
    // Matrix G and GT
    mfem::MixedBilinearForm blf_G(&H1, &ND);
    mfem::SparseMatrix GT;
    blf_G.AddDomainIntegrator(new mfem::MixedVectorGradientIntegrator());
    blf_G.Assemble();
    blf_G.FormRectangularSystemMatrix(ND_etdof,H1_etdof,GT);
    GT.Finalize();
    mfem::SparseMatrix *G = Transpose(GT);

    // std::cout << "M:  " <<M->NumRows()  <<" "<< M->NumCols() << "\n"; 
    std::cout << "CT: " <<CT.NumRows() <<" "<< CT.NumCols() << "\n"; 
    // std::cout << "G:  " <<G->NumRows() <<" "<<  G->NumCols() << "\n"; 
    std::cout << "C:  " <<C->NumRows() <<" "<<  C->NumCols() << "\n"; 
    // std::cout << "Nn:  " <<Nn.NumRows() <<" "<<  Nn.NumCols() << "\n"; 

    // primal: A1*x=b1
    // [M+Rd   C^T    G] [u]   [(M-Rd)*u - C^T*z + f]
    // [C      -N      ] [w] = [          0         ]
    // [G^T            ] [p]   [          0         ]
    //
    // dual: A2*y=b2
    // [N+Rp   C   -D^T] [v]   [(N-Rp)*u - C*w + f]
    // [C^T    -M    0 ] [z] = [         0        ]
    // [D      0     0 ] [q]   [         0        ]


    int size_p = M->NumCols() + CT.NumCols() + G->NumCols();
    mfem::SparseMatrix A1(size_p);
    AddSubmatrix(*M, A1, 0, 0); // submatrix, matrix, rowoffset, coloffset
    AddSubmatrix(CT, A1, 0, M->NumCols());
    AddSubmatrix(*G,  A1, 0, M->NumCols() + CT.NumCols());
    AddSubmatrix(*C,  A1, M->NumRows(), 0);
    AddSubmatrix(GT, A1, M->NumRows() + C->NumRows(), 0);
    AddSubmatrix(Nn, A1, M->NumRows(), M->NumCols());
    A1.Finalize();
    std::cout << "progress1" << "\n";


    delete fec_H1;
    delete fec_ND;
    delete fec_RT;
    delete fec_DG;

    std::cout << "---------------finish MEHC---------------\n";
}


