#include <stdio.h>
#include "array.h"
#include "wavelet.h"


static void test_vector(int n) 
{
    double* v;

    make_vector(v, n);
    for (int i = 0; i < n; i++) {
        v[i] = 1.0 / (i+1);
    }
    printf("original vector: \n");
    print_vector("%8.4f ", v, n);
    puts("");

    haar_transform_vector(v, n, WT_FWD);
    printf("transformed vector:\n");
    print_vector("%8.4f ", v, n);
    puts("");

    haar_transform_vector(v, n, WT_REV);
    printf("reconstructed vector:\n");
    print_vector("%8.4f ", v, n);
    puts("");
}


int main()
{
    test_vector(8);
    return 0;
}