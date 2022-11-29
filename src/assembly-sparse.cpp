#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;



// TODO: this does not work yet, try advice from wouter 



// assembly options:
// 1) submatrices sind sparse und wie wouter
// 2) submatrices sind dense
// 3) blockmatrix (maybe works like blockoperator?)
// 4) blockoperator mfem example 5


// i = 4, j = 8
void printmatrix(mfem::Matrix &mat) {
    for (int i = 0; i<mat.NumRows(); i++) {
        for (int j = 0; j<mat.NumCols(); j++) {
            std::cout << std::setprecision(1) << std::fixed;
            std::cout << mat.Elem(i,j) << " ";
        }
        std::cout <<"\n";
    }
    std::cout << "progress: matrix test\n";
}
//     int* I =  mat.GetI();
//     for (int i = 0; i< mat.Height()+1 ; ++i){
//         std::cout << I[i] << " ";
//     }
//     std::cout << "----------\n";
//     int* J = mat.GetJ();
//     for (int i = 0; i< I[mat.Height()] ; ++i){
//         std::cout << J[i] << " ";
//         if((i+1)%36 == 1){
//             std::cout << std::endl;
//         }
//     }
// }

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

    // mesh
    const char *mesh_file = "extern/mfem-4.5/data/ref-cube.mesh";
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    // FE spaces
    int order = 1;
    FiniteElementCollection *fec_RT = new RT_FECollection(order, dim);
    FiniteElementCollection *fec_ND = new ND_FECollection(order, dim);
    FiniteElementSpace RT(&mesh, fec_RT);
    FiniteElementSpace ND(&mesh, fec_ND);

    // boundary conditions
    mfem::Array<int> RT_etdof;
    mfem::Array<int> ND_etdof;

    // Matrix M and -M
    mfem::BilinearForm blf_M(&ND);
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ND_etdof,M);
    M.Finalize();
    M*=100;
    
    // Matrix Nn
    mfem::BilinearForm blf_N(&RT);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.FormSystemMatrix(RT_etdof,N);
    N.Finalize();
    mfem::SparseMatrix Nn = N;
    Nn *= -1;
    Nn.Finalize();

    // Matrix C and CT
    mfem::MixedBilinearForm blf_C(&ND, &RT);
    mfem::SparseMatrix C;
    blf_C.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
    blf_C.Assemble();
    blf_C.FormRectangularSystemMatrix(ND_etdof,RT_etdof,C);
    C.Finalize();
    mfem::SparseMatrix *CT = Transpose(C);
    CT->Finalize();

    // assemble sparsematrix A
    int size_p = M.NumCols() + CT->NumCols();
    mfem::SparseMatrix A(size_p);
    AddSubmatrix(M,   A, 0, 0); // submatrix, matrix, rowoffset, coloffset
    AddSubmatrix(*CT, A, 0, M.NumCols());
    AddSubmatrix(C,   A, M.NumRows(), 0);
    // AddSubmatrix(Nn,  A, M.NumRows(), M.NumCols());
    A.Finalize();

    // rhs
    mfem::Vector b(size_p);
    mfem::Vector x(size_p);
    x = 0.924;
    A.Mult(x,b); // set b=A*x
    x = -1.1;

    // check b
    std::cout << "----check b----\n";
    for (int j = 0; j<size_p; j++) {
        std::cout << std::setprecision(3) << std::fixed;
        std::cout << b[j]<< "\n";
    }

    // MINRES
    int iter = 200000;
    int tol = 1e-4;
    std::cout << "----minres start----\n";
    mfem::MINRES(A, b, x, 0, iter, tol, tol); 
    
    // check if x=[1,1,...1]
    for (int j = 0; j<size_p; j++) {
        std::cout << std::setprecision(3) << std::fixed;
        std::cout << x[j]<< "\n";
    }


    // assembly matrix A
    // A.AddSubMatrix(arr,arr,M);
    // mfem::Array<int> arr(2);
    // arr = {M.NumCols(),0};
    // mfem::BlockMatrix A(arr);
    // printmatrix(A);
    // A.SetBlock(0,0,&M);
    // printmatrix(A);
    // std::cout << A.Elem(1,1) << "\n";
    // AddSubmatrix(M,A,0,0);
    // mfem::Array<int> arr;
    // mfem::Array<int> arr(12);
    // arr = {0,0,0,0,0,0,0,0,0,0,0,0};
    // arr = 1;
    // A.Finalize();
    // printmatrix(A);


    delete fec_RT;
}