#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wavelet.h"
#include "modern.h"
#include "array.h"

#define SQRT1_2 sqrt(1.0/2)


// back to chapter 13 to review, Page 106 explain this function
/**
 * @brief apply haar wavelet transform on input vector forward 
 *
 * @param v, v_j
 * @param n, vector size, should be power of 2, n = 2_j
 * @return happen on the vector v
 *   @retval void 
 */
static void haar_transform_vector_forward(double* v, int n)
{
    double h = sqrt(n);
    int i, d;
    for (i = 0; i < n; i++) {
        v[i] /= h;
    }
    for (d = 1; d < n; d *= 2) {
        for (i = 0; i < n; i += 2*d) {
            double x = SQRT1_2 * (v[i] + v[i+d]);
            double y = SQRT1_2 * (v[i] - v[i+d]);
            v[i] = x;
            v[i+d] = y;
        }
    }
}

static void haar_transform_vector_reverse(double* v, int n)
{
    double h = sqrt(n);
    int i, d;
    for (d = n/2; d > 0; d /= 2) {
        for (i = 0; i < n; i += 2*d) {
            double x = SQRT1_2 * (v[i] + v[i+d]);
            double y = SQRT1_2 * (v[i] - v[i+d]);
            v[i] = x;
            v[i+d] = y;
        }
    }
    for (i = 0; i < n; i++) {
        v[i] *= h;
    }
}


static void haar_transform_matrix_forward(double** a, int m, int n)
{
    for (int i = 0; i < m; i++) {
        haar_transform_vector(a[i], n, WT_FWD);
    }
    // create a transpose matrix of a
    // apply the haar transform on new matrix's rows
    // them mapping back to a
    double** t;
    make_matrix(t, n, m);
    matrix_transpose(a, m, n, t);
    for (int j = 0; j < n; j++) {
        haar_transform_vector(t[j], m, WT_FWD);
    }
    matrix_transpose(t, n, m, a);
}

static void haar_transform_matrix_reverse(double** a, int m, int n)
{
    for (int i = 0; i < m; i++) {
        haar_transform_vector(a[i], n, WT_REV);
    }
    double** t;
    make_matrix(t, n, m);
    matrix_transpose(a, m, n, t);
    for (int j = 0; j < n; j++) {
        haar_transform_vector(t[j], m, WT_REV);
    }
    matrix_transpose(t, n, m, a);
}

/**
 * @brief apply haar wavelet transform on input vector forward or 
 *   reverse in place
 *
 * @param v, 
 * @param n, vector size 
 * @param dir, transform direction, forward of reverse 
 * @return happen on the vector v
 *   @retval void 
 */
void haar_transform_vector(double* v, int n, int dir)
{
    if (dir == WT_FWD) {
        haar_transform_vector_forward(v, n);
    } else if (dir == WT_REV) {
        haar_transform_vector_reverse(v, n);
    } else {
        error("*** error in haar_transform_vector() "
              "the third argument should be one of "
              "WT_FWD or WT_REV\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief apply haar wavelet transform on matrix forward or reverse 
 *   in place
 *
 * @param a, 
 * @param m, matrix row 
 * @param n, matrix column 
 * @param dir, transform direction, forward of reverse 
 * @return happen on the vector v
 *   @retval void 
 */
void haar_transform_matrix(double** a, int m, int n, int dir)
{
    if (dir == WT_FWD) {
        haar_transform_matrix_forward(a, m, n);
    } else if (dir == WT_REV) {
        haar_transform_matrix_reverse(a, m, n);
    } else {
        error("*** error in haar_transform_matrix() "
              "the third argument should be one of "
              "WT_FWD or WT_REV\n");
        exit(EXIT_FAILURE);
    }
}