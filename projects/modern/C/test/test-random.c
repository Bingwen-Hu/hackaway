// generate portable bitmap

#include <stdio.h>
#include <stdlib.h>
#include "array.h"
#include "random.h"

static int write_pbm(char** M, int m, int n, char* outfile)
{
    FILE* f = fopen(outfile, "w");
    if (f == NULL) {
        fprintf(stderr, "faild to open file: %s\n", outfile);
        return -1;
    }

    fprintf(f, "P1\n");
    fprintf(f, "%d %d\n", m, n);
    for (int i = 0; i < m; i++) {
        fprint_vector(f, "%d ", M[i], n);
    }

    fclose(f);
    return 0;
}


static char** make_random_matrix(int m, int n, double f)
{
    char** M;
    int i, j, k;

    make_matrix(M, m, n);
    
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            M[i][j] = 0;
        }
    }

    k = 0;
    while (k < f * m * n) {
        i = randint(m);
        j = randint(n);
        // avoid repeatedly assign one pixel to black.
        if (M[i][j] == 0) {
            M[i][j] = 1;
            k++;
        }
    }

    return M;
}

static void show_usage(char* progname)
{
    printf("Usage: %s m n s f outfile\n", progname);
    printf("    Write an mxn random bitmap to a file named `outfile`\n");
    printf("    f: fill ratio  0.0 <= f <= 1.0\n");
    printf("    s: integer >= 1: seeds the random number generator\n");
}

int main(int argc, char** argv)
{
    int m, n, s; // image is m x n, seed is s
    double f; // fill ratio
    char** M;
    char* outfile;
    char* endptr;
    int status = EXIT_FAILURE;
    
    if (argc != 6) {
        show_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    m = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || m < 1) {
        show_usage(argv[0]);
        return status;
    }

    n = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || n < 1) {
        show_usage(argv[0]);
        return status;
    }

    s = strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || s < 1) {
        show_usage(argv[0]);
        return status;
    }

    f = strtod(argv[4], &endptr);
    if (*endptr != '\0' || f > 1. || f < 0.) {
        show_usage(argv[0]);
        return status;
    }

    outfile = argv[5];

    // initialization finish

    srand(s);
    M = make_random_matrix(m, n, f);
    if (write_pbm(M, m, n, outfile) == 1){
        status = EXIT_SUCCESS;
    }
    free_matrix(M);
    return status;
}