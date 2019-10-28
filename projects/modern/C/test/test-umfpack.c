#include <suitesparse/umfpack.h>
#include <stdio.h>
#include "array.h"
#include "sparse.h"
#include "modern.h"

int main()
{
    // init a matrix
    double** a;
    double* b;
    double* x; // for solved result
    int n = 5;
    make_matrix(a, n, n); 
    make_vector(b, n);
    make_vector(x, n);

    a[0][0] = 0; a[0][1] = 3; a[0][2] = 0; a[0][3] = 0; a[0][4] = 0;
    a[1][0] = 0; a[1][1] = 0; a[1][2] = 4; a[1][3] = 0; a[1][4] = 6;
    a[2][0] = 0; a[2][1] = -1; a[2][2] = -3; a[2][3] = 2; a[2][4] = 0;
    a[3][0] = 0; a[3][1] = 0; a[3][2] = 1; a[3][3] = 0; a[3][4] = 0;
    a[4][0] = 0; a[4][1] = 4; a[4][2] = 2; a[4][3] = 0; a[4][4] = 1;
    
    b[0] = 8; b[1] = 45; b[2] = -3; b[3] = 3; b[4] = 19;

    // solution: x = 1 2 3 4 5

    int nonzero = 12;

    int* Ap;
    int* Ai;
    double* Ax;
    make_vector(Ap, n+1);
    make_vector(Ai, nonzero);
    make_vector(Ax, nonzero);

    sparse_pack(a, n, n, Ap, Ai, Ax);

    void* symbolic;
    void* numeric;
    int status;
    status = umfpack_di_symbolic(n, n, Ap, Ai, Ax, &symbolic, NULL, NULL);

    if (status != UMFPACK_OK) {
        error("umfpack_di_symbol() fail!");
        return EXIT_FAILURE;
    }

    status = umfpack_di_numeric(Ap, Ai, Ax, symbolic, &numeric, NULL, NULL);

    if (status == UMFPACK_WARNING_singular_matrix) {
        error("matrix is singular");
        return EXIT_FAILURE;
    }   

    umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, numeric, NULL, NULL);

    print_vector("%3.1lf ", x, n);

    umfpack_di_free_symbolic(&symbolic);
    umfpack_di_free_numeric(&numeric);

    free_matrix(a);
    free_vector(b);
    free_vector(x);
    free_vector(Ap);
    free_vector(Ai);
    free_vector(Ax);

    return 0;
}
