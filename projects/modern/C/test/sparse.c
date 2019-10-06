#include "sparse.h"


void sparse_pack(double** a, int m, int n,
        int* Ap, int* Ai, double* Ax)
{
    int nonzero = 0;
    Ap[0] = nonzero;

    // column first
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (a[i][j] != 0) {
                Ai[nonzero] = i;
                Ax[nonzero] = a[i][j];
                nonzero++;
            }
        }
        Ap[j+1] = nonzero;
    }
}

void sparse_unpack(double** a, int m, int n, 
        int* Ap, int* Ai, double* Ax)
{
    int i, j, k; // row, column, ap index
    // init
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            a[i][j] = 0.0;
        }
    }

    // iterate each column in Ap, get the row index
    // final assign the value from Ax
    for (j = 0; j < n; j++) {
        for (k = Ap[j]; k < Ap[j+1]; k++) {
            i = Ai[k];
            a[i][j] = Ax[k];
        }
    }
}
