#include <stdio.h>
#include "array.h"


int main() {
    double* v; 
    int n = 5;
    make_vector(v, n); 
    for (int i = 0; i < n; i++) {
        v[i] = 1.0 / (1 + i);
    } 
    print_vector("%7.3f", v, n);
    free_vector(v);

    int** M;
    int row = 3;
    int col = 6;
    make_matrix(M, row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            M[i][j] = j + i;
    }
    print_matrix("%3d ", M, row, col);

    int** T;
    make_matrix(T, col, row);
    matrix_transpose(M, row, col, T);
    print_matrix("%3d ", T, col, row);
    
    free_matrix(T);
    free_matrix(M);
}


