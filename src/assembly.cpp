#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;


// some tests on assembling A1 and A2


// TODO test options for assembly:
// 1) submatrices sind sparse und wie wouter
// 2) submatrices sind dense
// 3) blockmatrix


// void printmatrix(mfem::Matrix &mat) {
// i = 4, j = 8
void printmatrix(mfem::Matrix &mat) {
    for (int i = 0; i<mat.NumRows(); i++) {
        for (int j = 0; j<mat.NumCols(); j++) {
            std::cout << std::setprecision(1) << std::fixed;
            std::cout << mat.Elem(i,j) << " ";
            // std::cout << mat(i,j) << " ";
            // std::cout << mat.getI << " ";
        }
        std::cout <<"\n";
    }
    std::cout << "progress: matrix test\n";

    // int* I =  mat.GetI();
    // for (int i = 0; i< mat.Height()+1 ; ++i){
    //     std::cout << I[i] << " ";
    // }
    // std::cout << "----------\n";
    // int* J = mat.GetJ();
    // for (int i = 0; i< I[mat.Height()] ; ++i){
    //     std::cout << J[i] << " ";
    //     if((i+1)%36 == 1){
    //         std::cout << std::endl;
    //     }
    // }

}

// TODO check how this works in wouters file
// void AddSubmatrix(mfem::SparseMatrix submatrix, mfem::SparseMatrix matrix, int rowoffset, int coloffset) {
//     for (int r = 0; r < submatrix.NumRows(); r++) {
//         mfem::Array<int> cols;
//         mfem::Vector srow;
//         submatrix.GetRow(r, cols, srow);
//         for (int c = 0; c < submatrix.NumCols(); c++) {
//             matrix.Add(rowoffset + r, coloffset + cols[c], srow[c]);
//         }
//         cols.DeleteAll();
//     }
// }



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
    // mfem::DenseMatrix M;
    mfem::SparseMatrix M;
    blf_M.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_M.Assemble();
    blf_M.FormSystemMatrix(ND_etdof,M);
    M*=10;
    M.Finalize();

    // Matrix N and -N
    mfem::BilinearForm blf_N(&RT);
    mfem::SparseMatrix N;
    blf_N.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
    blf_N.Assemble();
    blf_N.Finalize();
    blf_N.FormSystemMatrix(RT_etdof,N);
    N.Finalize();



    // assembly matrix A
    mfem::Array<int> arr(2);
    arr = {M.NumCols(),0};
    mfem::BlockMatrix A(arr);


    // mfem::SparseMatrix A(M.NumRows()+3);

    // printmatrix(A);
    A.SetBlock(0,0,&M);
    // printmatrix(A);
    std::cout << A.Elem(1,1) << "\n";

    // AddSubmatrix(M,A,0,0);
    // mfem::Array<int> arr;
    // mfem::Array<int> arr(12);
    // arr = {0,0,0,0,0,0,0,0,0,0,0,0};
    // arr = 1;
    // A.AddSubMatrix(arr,arr,M);
    // A.Finalize();
    // printmatrix(A);


    delete fec_RT;
}