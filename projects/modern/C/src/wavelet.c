#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <wavelet.h>

#define SQRT1_2 sqrt(1.0/2)

static void haar_transform_vector_forward(double* v, int n)
{

}

static void haar_transform_vector_reverse(double* v, int n)
{

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