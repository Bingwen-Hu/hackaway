#include <stdio.h>
#include "sparse.h"
#include "array.h"


int main()
{
    double** a;
    int m = 4, n = 5;
    make_matrix(a, m, n);

    // init some value
    a[0][1] = 7; a[0][4] = 1;
    a[1][1] = 4; a[1][3] = 3;
    a[2][2] = 5; a[2][3] = 1; 
    a[3][0] = 5; a[3][1] = 2;
    // count number of nonzero entries
    int nonzero = 8;    

    printf("Print the Matrix\n");
    print_matrix("%3.0f ", a, m, n);

    printf("Nonzero entries is %d\n", nonzero);

    // start CCS
    int* Ap;
    int* Ai;
    double* Ax;
    make_vector(Ap, n+1);
    make_vector(Ai, nonzero);
    make_vector(Ax, nonzero);

    sparse_pack(a, m, n, Ap, Ai, Ax);
    printf("Sparse pack: \n");
    printf("AP = ");
    print_vector("%3d ", Ap, n+1);
    printf("Ai = ");
    print_vector("%3d ", Ai, nonzero);
    printf("Ax = ");
    print_vector("%3.f ", Ax, nonzero);

    // validate
    double** b;
    make_matrix(b, m, n);
    sparse_unpack(b, m, n, Ap, Ai, Ax);

    double** c;
    make_matrix(c, m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0 ; j < n; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }

    putchar('\n');
    printf("the difference between original matrix and reconstructed one\n");
    print_matrix("%1.1f ", c, m, n);

    free_matrix(a);
    free_matrix(b);
    free_matrix(c);
    free_vector(Ai);
    free_vector(Ap);
    free_vector(Ax);
}