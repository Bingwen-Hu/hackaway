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
    make_matrix(M, 10, 10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++)
            M[i][j] = j + i;
    }
    print_matrix("%3d ", M, 10, 10);
    free_matrix(M);
}


