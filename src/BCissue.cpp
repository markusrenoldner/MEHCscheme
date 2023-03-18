#include <iostream>
#include "mfem.hpp"


// understanding how formlinearsystem works

// forum question: https://github.com/mfem/mfem/discussions/3544

// current understanding:
// formlinsys has to be called on a symmetric operator, otherwise it fails

// this is and example of a non-symmetric operator:


int main(int argc, char *argv[]) {

    // submatrix 1
    mfem::SparseMatrix mat1 (3,3);
    mat1.Set(0,0,1.);
    mat1.Set(0,1,1.);
    mat1.Set(1,0,1.);
    mat1.Set(1,1,1.);
    mat1.Set(2,2,1.);

    // submatrix 2
    mfem::SparseMatrix mat2 (2,2);
    mat2.Set(1,0,1.);
    mat2.Set(0,1,4.);

    // submatrix 3
    mfem::SparseMatrix mat3 (2,3);
    mat3.Set(1,0,1.);

    // blockmatrix
    mfem::Array<int> offsets (3);
    offsets[0] = 0;
    offsets[1] = 3;
    offsets[2] = 2;
    offsets.PartialSum();
    mfem::BlockOperator a(offsets);
    // mfem::BlockMatrix a(offsets);

    // set blocks
    // [mat1   0  ]
    // [mat3  mat2]
    a.SetBlock(0,0,&mat1);
    a.SetBlock(1,1,&mat2);
    a.SetBlock(1,0,&mat3);

    // vectors
    mfem::Vector b (5);
    b.Elem(0) = 11.;
    b.Elem(1) = 11.;
    b.Elem(2) = 12.;
    b.Elem(3) = 13.;
    b.Elem(4) = 14.;
    mfem::Vector x (5);
    x.Elem(0) = 2.;
    x.Elem(1) = 3.;
    x.Elem(2) = 4.;
    x.Elem(3) = 5.;
    x.Elem(4) = 6.;

    // BC
    mfem::Array <int> esstdof;
    esstdof.Append(3);

    // formlinsys
    mfem::Operator* A;
    mfem::Vector X;
    mfem::Vector B;
    a.FormLinearSystem(esstdof, x, b, A, X, B);
    // a.SetDiagonalPolicy(mfem::OperatorDiagonalPolicy::DIAG_ONE);
    // a.DiagonalPolicy = mfem::OperatorDiagonalPolicy::DIAG_ONE;
    // mfem::Operator *pt_a = &a;
    // pt_a->SetDiagonalPolicy(mfem::OperatorDiagonalPolicy::DIAG_ONE);

    // transfer the operator A to a dense matrix for plotting
    mfem::DenseMatrix dense_mat(5,5);
    mfem::Vector temp_in(5), temp_out(5);
    for(int i=0; i<5;i++) {
        temp_in  = 0.;
        temp_out = 0.;
        temp_in.Elem(i) = 1.;
        A->Mult(temp_in, temp_out);
        dense_mat.SetCol(i,temp_out);
    }

    // print
    std::cout << "A: \n";
    dense_mat.PrintMatlab(std::cout);
    std::cout << "b: ";
    b.Print(std::cout);
    std::cout << "B: ";
    B.Print(std::cout);
    std::cout << "x: ";
    x.Print(std::cout);
    std::cout << "X: ";
    X.Print(std::cout);
}