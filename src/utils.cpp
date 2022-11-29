#include "mfem.hpp"
#include <fstream>
#include <iostream>


void printvector(mfem::Vector vec, int stride=1) {
    std::cout << std::endl<<"vec =\n";
    for (int j = 0; j<vec.Size(); j+=stride) {
        std::cout << std::setprecision(3) << std::fixed;
        if (stride != 1 ) {std::cout << vec[j]<< "\n...";}
        else {std::cout << vec[j];}
        std::cout << "\n";
    }
}

void printmatrix(mfem::Matrix &mat) {
    for (int i = 0; i<mat.NumRows(); i++) {
        for (int j = 0; j<mat.NumCols(); j++) {
            std::cout << std::setprecision(1) << std::fixed;
            std::cout << mat.Elem(i,j) << " ";
        }
        std::cout <<"\n";
    }
}