

// a very tedious way to assemble A1 from submatrices 


// add N+Rh to A1
for (int r = 0; r < N.NumRows(); r++) { // rows of N or Rh

    mfem::Array<int> cols;
    mfem::Vector srow;
    N.GetRow(r, cols, srow);
    for (int c = 0; c < N.NumCols(); c++) { // cols of N
        A1.Add(r, cols[c], srow[c]); // add cols of N to A1
    }

    // cols.DeleteAll();
    // Rh.GetRow(r, cols, srow);
    // for (int c = 0; c < N.NumCols(); c++) { 
    //     A1.Add(r, cols[c], srow[c]);
    // }
}
std::cout << "---------------check6---------------\n";

// add C to A1
for (int r = 0; r < C.NumRows(); r++) {
    mfem::Array<int> cols;
    mfem::Vector srow;
    C.GetRow(r, cols, srow);
    for (int c = 0; c < C.NumCols(); c++) {
        A1.Add(r, N.NumCols() + cols[c], srow[c]);
    }
}
std::cout << "---------------check7---------------\n";

// add D^T to A1
// TODO transpose
for (int r = 0; r < D.NumRows(); r++) {
    mfem::Array<int> cols;
    mfem::Vector srow;
    D.GetRow(r, cols, srow);
    for (int c = 0; c < D.NumCols(); c++) {
        A1.Add(r, N.NumCols() + cols[c], srow[c]);
    }
}