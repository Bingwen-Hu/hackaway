#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wavelet.h"

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

}

static void haar_transform_matrix_reverse(double** a, int m, int n)
{

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

}