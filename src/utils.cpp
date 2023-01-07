#include "mfem.hpp"
#include <fstream>
#include <iostream>

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

void PrintVector2(mfem::Vector vec, int stride=1) {
    std::cout << std::endl<<"vec =\n";
    for (int j = 0; j<vec.Size(); j+=stride) {
        std::cout << std::setprecision(3) << std::fixed;
        if (stride != 1 ) {std::cout << vec[j]<< "\n...";}
        else {std::cout << vec[j];}
        std::cout << "\n";
    }
    std::cout << "--------------\n";
}

void PrintVector(mfem::Vector vec, int stride=1) {
    std::cout << std::endl<<"vec[::"<<stride<<"]=\n";
    for (int j = 0; j<vec.Size(); j+=stride) {
        std::cout << std::setprecision(3) << std::fixed;
        std::cout << vec[j]<< "\n";
    }
    std::cout << "--------------\n";
}

void PrintVector3(mfem::Vector vec, int stride=1, int start=0, int stop=0, int prec=3) {
    if (stop==0) { stop = vec.Size(); }
    std::cout << "vec=\n";
    for (int j = start; j<stop; j+=stride) {
        std::cout << std::setprecision(prec) << std::fixed;
        std::cout << vec[j]<< "\n";
    }
    std::cout << "\n";
}


void PrintMatrix(mfem::Matrix &mat, int prec=2) {
    for (int i = 0; i<mat.NumRows(); i++) {
        for (int j = 0; j<mat.NumCols(); j++) {
            std::cout << std::setprecision(prec) << std::fixed;
            std::cout << mat.Elem(i,j) << " ";
        }
        std::cout <<"\n";
    }
}